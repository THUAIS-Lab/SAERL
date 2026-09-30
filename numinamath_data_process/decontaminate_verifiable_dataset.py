#!/usr/bin/env python3
"""
对“已可验证”的本地数据集执行去数据污染（decontamination）。

设计目标:
- 不依赖项目内其他脚本，单文件可运行。
- 输入均为本地路径（重点支持 parquet 文件/目录）。
- 候选集与测试集由命令行参数角色决定，不依赖 split 概念。

核心流程:
1) 读取候选可验证数据集，抽取问题字段。
2) 读取多个测试集路径，构建 benchmark 问题池。
3) 用本地 embedding 模型做 top-k 召回。
4) 对高相似样本做重复/改写判定（启发式，或可选本地 Llama Judge）。
5) 输出去污染后样本、污染样本和统计摘要。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher


QUESTION_KEYS = (
    "question",
    "problem",
    "Problem",
    "query",
    "instruction",
    "prompt",
    "input",
)


def _maybe_tqdm(iterable, desc: str):
    """优先显示进度条；无 tqdm 时回退为原始迭代器。"""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def to_str(value: Any) -> str:
    """把任意值转成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def first_non_empty(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """按键顺序返回第一个非空字段。"""
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_question(text: str) -> str:
    """问题文本归一化，用于启发式重复判断。"""
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def heuristic_is_duplicate(
    q1: str,
    q2: str,
    ratio_threshold: float = 0.90,
    jaccard_threshold: float = 0.85,
) -> bool:
    """启发式同题判定：编辑距离 + 词集 Jaccard。"""
    n1 = normalize_question(q1)
    n2 = normalize_question(q2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    ratio = SequenceMatcher(None, n1, n2).ratio()
    if ratio >= ratio_threshold:
        return True

    t1 = set(n1.split())
    t2 = set(n2.split())
    if not t1 or not t2:
        return False
    jac = len(t1 & t2) / len(t1 | t2)
    return jac >= jaccard_threshold


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 返回文本中提取 JSON 对象。"""
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        return json.loads(text[left : right + 1])
    except Exception:
        return None


class LocalLlamaJudge:
    """本地 Llama 判重器（transformers 直连本地模型目录）。"""

    def __init__(
        self,
        model_path: str,
        device_map: str,
        torch_dtype: str,
        max_new_tokens: int,
        temperature: float,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'transformers'/'torch'. Install with: pip install transformers torch"
            ) from exc

        dtype_value: Any
        if torch_dtype == "auto":
            dtype_value = "auto"
        elif torch_dtype == "float16":
            dtype_value = torch.float16
        elif torch_dtype == "bfloat16":
            dtype_value = torch.bfloat16
        elif torch_dtype == "float32":
            dtype_value = torch.float32
        else:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype_value,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """对话式生成，返回模型文本输出。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "[SYSTEM]\n"
                f"{system_prompt}\n\n"
                "[USER]\n"
                f"{user_prompt}\n\n"
                "[ASSISTANT]\n"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()


def maybe_build_local_judge(args: argparse.Namespace) -> Optional[LocalLlamaJudge]:
    """按需构建本地 Llama Judge。"""
    if not args.use_llm_judge:
        return None
    if not args.judge_model_path:
        raise ValueError("--use-llm-judge enabled, but --judge-model-path is empty.")

    return LocalLlamaJudge(
        model_path=args.judge_model_path,
        device_map=args.judge_device_map,
        torch_dtype=args.judge_torch_dtype,
        max_new_tokens=args.judge_max_new_tokens,
        temperature=args.judge_temperature,
        trust_remote_code=args.judge_trust_remote_code,
    )


def llm_duplicate_judge(
    judge: LocalLlamaJudge,
    candidate_question: str,
    benchmark_question: str,
) -> bool:
    """LLM 判定两个题目是否同题/改写。"""
    system_prompt = (
        "You are a strict decontamination judge for math datasets. "
        "Decide if two questions are duplicates or paraphrases that test essentially the same problem."
    )
    user_prompt = (
        "Question A:\n"
        f"{candidate_question}\n\n"
        "Question B:\n"
        f"{benchmark_question}\n\n"
        "Return JSON only with this schema:\n"
        '{"duplicate_or_paraphrase": true/false, "reason": "..."}'
    )

    text = judge.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    payload = parse_json_from_text(text)
    if payload is None:
        return heuristic_is_duplicate(candidate_question, benchmark_question)
    return bool(payload.get("duplicate_or_paraphrase", False))


class EmbeddingEncoder:
    """句向量编码器。支持本地模型目录或模型名。"""

    def __init__(self, model_name: str, batch_size: int, device: Optional[str]) -> None:
        try:
            import importlib

            sentence_transformers = importlib.import_module("sentence_transformers")
            SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(arr, dtype=np.float32)


def cosine_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """对每条 query 返回 corpus top-k 相似项索引和得分。"""
    if corpus_emb.shape[0] == 0:
        return (
            np.zeros((query_emb.shape[0], 0), dtype=np.int32),
            np.zeros((query_emb.shape[0], 0), dtype=np.float32),
        )

    k = min(k, corpus_emb.shape[0])
    sims = query_emb @ corpus_emb.T
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)
    return top_idx.astype(np.int32), top_scores.astype(np.float32)


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读 JSONL。"""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    """读取 parquet 文件为记录列表。"""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pylist()
    except Exception:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(
                "Failed to read parquet. Install pyarrow or pandas."
            ) from exc

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")


def load_local_records(data_path: str) -> List[Dict[str, Any]]:
    """
    读取本地数据。

    支持:
    - 单个 parquet/json/jsonl 文件
    - 目录（自动读取目录下全部 parquet 文件）
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {data_path}")

    if p.is_dir():
        parquet_files = sorted([x for x in p.glob("*.parquet") if x.is_file()])
        if not parquet_files:
            raise ValueError(f"No parquet files found in directory: {data_path}")

        all_records: List[Dict[str, Any]] = []
        for file in parquet_files:
            all_records.extend(read_parquet(file))
        return all_records

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return read_parquet(p)
    if suffix == ".jsonl":
        return read_jsonl(p)
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON must be a list of objects: {data_path}")
        return data

    raise ValueError(f"Unsupported file type: {data_path}")


def extract_question(record: Dict[str, Any], preferred_key: str = "") -> str:
    """从记录抽取问题字段。"""
    if preferred_key:
        return to_str(record.get(preferred_key, "")).strip()
    return to_str(first_non_empty(record, QUESTION_KEYS)).strip()


@dataclass
class CandidateItem:
    """候选训练样本。"""

    source_path: str
    row_index: int
    question: str
    record: Dict[str, Any]


def load_candidates(
    source_path: str,
    question_key: str,
    max_samples: Optional[int],
) -> Tuple[List[CandidateItem], Dict[str, Any]]:
    """加载候选数据集并抽取问题字段。"""
    records = load_local_records(source_path)

    items: List[CandidateItem] = []
    for idx, rec in enumerate(records):
        question = extract_question(rec, preferred_key=question_key)
        if not question:
            continue

        items.append(
            CandidateItem(
                source_path=source_path,
                row_index=idx,
                question=question,
                record=dict(rec),
            )
        )

    if max_samples is not None:
        items = items[:max_samples]

    stats = {
        "source_path": source_path,
        "raw_records": len(records),
        "usable_candidates": len(items),
    }
    return items, stats


def load_benchmark_questions(
    benchmark_paths: Sequence[str],
    question_key: str,
    max_samples_per_file: Optional[int],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """加载多个测试集并提取问题，拼成去污染 benchmark 池。"""
    question_to_source: Dict[str, str] = {}
    by_source: Dict[str, Dict[str, int]] = {}

    for bench_path in benchmark_paths:
        records = load_local_records(bench_path)
        if max_samples_per_file is not None:
            records = records[:max_samples_per_file]

        raw_n = len(records)
        usable_n = 0
        for rec in records:
            question = extract_question(rec, preferred_key=question_key)
            if not question:
                continue
            usable_n += 1
            if question not in question_to_source:
                question_to_source[question] = bench_path

        by_source[bench_path] = {
            "raw_records": raw_n,
            "usable_questions": usable_n,
        }

    unique_questions = list(question_to_source.keys())
    question_sources = [question_to_source[q] for q in unique_questions]
    stats = {
        "benchmark_sources": by_source,
        "benchmark_questions_raw": sum(v["usable_questions"] for v in by_source.values()),
        "benchmark_questions_unique": len(unique_questions),
    }
    return unique_questions, question_sources, stats


def decontaminate(
    candidates: Sequence[CandidateItem],
    benchmark_questions: Sequence[str],
    benchmark_question_sources: Sequence[str],
    encoder: EmbeddingEncoder,
    top_k: int,
    min_similarity: float,
    judge_client: Optional[LocalLlamaJudge],
    chunk_size: int,
    topk_similarity_path: Optional[Path],
    attach_metadata: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """执行语义检索 + 判重去污染，并落盘每个候选的 top-k 相似度。"""
    if not benchmark_questions:
        raise ValueError("Benchmark question pool is empty. Please provide valid --benchmark inputs.")

    bench = list(dict.fromkeys([q for q in benchmark_questions if q.strip()]))
    if len(bench) != len(benchmark_question_sources):
        raise ValueError("benchmark_questions and benchmark_question_sources length mismatch")

    bench_emb = encoder.encode(bench)

    kept_records: List[Dict[str, Any]] = []
    dropped_records: List[Dict[str, Any]] = []

    judged_pairs = 0
    fallback_heuristic_calls = 0
    dropped = 0
    topk_similarity_records = 0

    topk_writer = None
    if topk_similarity_path is not None:
        topk_similarity_path.parent.mkdir(parents=True, exist_ok=True)
        topk_writer = topk_similarity_path.open("w", encoding="utf-8")

    try:
        candidate_list = list(candidates)
        for start in tqdm(range(0, len(candidate_list), chunk_size), desc="Decontamination"):
            chunk = candidate_list[start : start + chunk_size]
            q_texts = [c.question for c in chunk]
            q_emb = encoder.encode(q_texts)
            idx_mat, sim_mat = cosine_topk(q_emb, bench_emb, k=top_k)

            for row_i, item in tqdm(enumerate(chunk), desc="Judging candidates", leave=False):
                contaminated = False
                matched_question = ""
                matched_source = ""
                matched_similarity = 0.0
                max_checked_similarity = 0.0
                current_topk: List[Dict[str, Any]] = []

                for rank, (cand_i, sim) in enumerate(
                    zip(idx_mat[row_i].tolist(), sim_mat[row_i].tolist()),
                    start=1,
                ):
                    sim_f = float(sim)
                    bench_q = bench[cand_i]
                    bench_source = benchmark_question_sources[cand_i]

                    current_topk.append(
                        {
                            "rank": rank,
                            "similarity": sim_f,
                            "benchmark_question": bench_q,
                            "benchmark_source": bench_source,
                        }
                    )

                    if sim_f > max_checked_similarity:
                        max_checked_similarity = sim_f

                    if sim_f < min_similarity:
                        continue

                    judged_pairs += 1

                    if judge_client is not None:
                        try:
                            is_dup = llm_duplicate_judge(
                                judge=judge_client,
                                candidate_question=item.question,
                                benchmark_question=bench_q,
                            )
                        except Exception:
                            fallback_heuristic_calls += 1
                            is_dup = heuristic_is_duplicate(item.question, bench_q)
                    else:
                        is_dup = heuristic_is_duplicate(item.question, bench_q)

                    if is_dup:
                        contaminated = True
                        matched_question = bench_q
                        matched_source = bench_source
                        matched_similarity = sim_f
                        break

                if topk_writer is not None:
                    topk_payload = {
                        "source_path": item.source_path,
                        "candidate_row_index": item.row_index,
                        "candidate_question": item.question,
                        "top_k": top_k,
                        "top_k_matches": current_topk,
                        "max_similarity": max_checked_similarity,
                    }
                    topk_writer.write(json.dumps(topk_payload, ensure_ascii=False) + "\n")
                    topk_similarity_records += 1

                out_rec = dict(item.record)
                if attach_metadata:
                    out_rec["_decontam"] = {
                        "source_path": item.source_path,
                        "row_index": item.row_index,
                        "question": item.question,
                        "contaminated": contaminated,
                        "max_checked_similarity": max_checked_similarity,
                        "matched_similarity": matched_similarity,
                        "matched_benchmark_question": matched_question,
                        "matched_benchmark_source": matched_source,
                    }

                if contaminated:
                    dropped += 1
                    dropped_records.append(out_rec)
                else:
                    kept_records.append(out_rec)
    finally:
        if topk_writer is not None:
            topk_writer.close()

    stats = {
        "input": len(candidates),
        "kept": len(kept_records),
        "dropped_contaminated": dropped,
        "benchmark_questions": len(bench),
        "top_k": top_k,
        "min_similarity": min_similarity,
        "llm_judge_enabled": judge_client is not None,
        "judged_pairs": judged_pairs,
        "fallback_heuristic_calls": fallback_heuristic_calls,
        "topk_similarity_file": str(topk_similarity_path) if topk_similarity_path is not None else None,
        "topk_similarity_records": topk_similarity_records,
    }
    return kept_records, dropped_records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Decontaminate a local verifiable dataset against local benchmarks.")

    parser.add_argument(
        "--input",
        required=True,
        help="候选可验证数据集路径（本地 parquet/json/jsonl 文件，或包含 parquet 的目录）",
    )
    parser.add_argument(
        "--input-question-key",
        default="",
        help="候选集问题字段名；留空自动从常见字段中推断",
    )
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="测试集路径（可重复传多个；本地 parquet/json/jsonl 文件，或包含 parquet 的目录）",
    )
    parser.add_argument(
        "--benchmark-question-key",
        default="",
        help="测试集问题字段名；留空自动从常见字段中推断",
    )
    parser.add_argument("--benchmark-max-samples-per-file", type=int, default=None)

    parser.add_argument(
        "--embedding-model",
        required=True,
        help="embedding 模型路径（支持本地目录）或模型名",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--chunk-size", type=int, default=256)

    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="启用本地 Llama Judge；不启用则使用启发式判重。",
    )
    parser.add_argument(
        "--judge-model-path",
        default="",
        help="本地 Llama 模型目录路径（仅在 --use-llm-judge 时使用）",
    )
    parser.add_argument(
        "--judge-device-map",
        default="auto",
        help="transformers device_map 参数，如 auto/cuda:0/cpu",
    )
    parser.add_argument(
        "--judge-torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="本地 Llama Judge 推理 dtype",
    )
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-trust-remote-code", action="store_true")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kept-file", default="decontaminated_kept.jsonl")
    parser.add_argument("--dropped-file", default="decontaminated_dropped.jsonl")
    parser.add_argument(
        "--topk-similarity-file",
        default="candidate_top5_similarity.jsonl",
        help="保存候选问题与 benchmark top-k 相似度的 JSONL 文件名",
    )
    parser.add_argument("--save-dropped", action="store_true")
    parser.add_argument("--attach-metadata", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载候选可验证数据集")
    candidates, candidate_stats = load_candidates(
        source_path=args.input,
        question_key=args.input_question_key,
        max_samples=args.max_samples,
    )
    print(f"  - source={args.input}, usable_candidates={len(candidates)}")
    print(f"  - candidate_stats: {json.dumps(candidate_stats, indent=2)}")

    print("[2/4] 加载多个测试集并构建 benchmark 问题池")
    benchmark_questions, benchmark_question_sources, benchmark_stats = load_benchmark_questions(
        benchmark_paths=args.benchmark,
        question_key=args.benchmark_question_key,
        max_samples_per_file=args.benchmark_max_samples_per_file,
    )
    print(f"  - benchmark unique questions={len(benchmark_questions)}")
    print(f"  - benchmark_stats: {json.dumps(benchmark_stats, indent=2)}")

    llm_client = maybe_build_local_judge(args)
    if llm_client is not None:
        print(f"  - local Llama judge enabled: {args.judge_model_path}")
    else:
        print("  - local Llama judge disabled, using heuristic duplicate judge")

    encoder = EmbeddingEncoder(
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
    )

    topk_similarity_path = output_dir / args.topk_similarity_file

    print("[3/4] 执行去污染")
    kept_records, dropped_records, decontam_stats = decontaminate(
        candidates=candidates,
        benchmark_questions=benchmark_questions,
        benchmark_question_sources=benchmark_question_sources,
        encoder=encoder,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        judge_client=llm_client,
        chunk_size=args.chunk_size,
        topk_similarity_path=topk_similarity_path,
        attach_metadata=args.attach_metadata,
    )
    print(
        f"  - kept={len(kept_records)}, dropped_contaminated={len(dropped_records)}, "
        f"input={len(candidates)}"
    )

    print("[4/4] 写出结果")
    kept_path = output_dir / args.kept_file
    write_jsonl(kept_path, kept_records)
    print(f"  - top-k similarity saved: {topk_similarity_path}")

    dropped_path = None
    if args.save_dropped:
        dropped_path = output_dir / args.dropped_file
        write_jsonl(dropped_path, dropped_records)

    summary = {
        "input_source": args.input,
        "candidate_stats": candidate_stats,
        "benchmark_stats": benchmark_stats,
        "decontam": decontam_stats,
        "outputs": {
            "kept": str(kept_path),
            "dropped": str(dropped_path) if dropped_path else None,
        },
        "config": vars(args),
        "timestamp": int(time.time()),
    }

    summary_path = output_dir / "decontam_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
