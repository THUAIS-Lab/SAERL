#!/usr/bin/env python3
"""
Reproduction pipeline for DeepMath-103K data curation.

This script is an open reimplementation of the data-cleaning process described in:
DeepMath-103K: arXiv:2504.11456

Important:
- The paper-level curation code has not been publicly released as an end-to-end script.
- This implementation follows the described methodology and provides practical fallbacks.

中文说明:
- 本脚本是 DeepMath-103K 论文清洗流程的工程化复现版本。
- 目标是把多来源数学数据统一成“可去污染、可难度筛选、可答案一致性验证”的训练样本。
- 主流程分为: 来源加载与适配 -> 来源白名单过滤 -> Stage1 基础过滤 -> Stage2 语义去污染
	-> Stage3 难度筛选 -> Stage4 答案可验证性筛选。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# 论文和官方评测里高频使用的 benchmark；用于 Stage2 去污染候选检索。
DEFAULT_BENCHMARK_SPECS = [
	# Benchmarks explicitly used in DeepMath evaluation settings.
	"zwhe99/MATH::math500",
	"zwhe99/aime90::2024",
	"math-ai/aime25::test",
	"zwhe99/amc23::test",
	"zwhe99/simplerl-minerva-math::test",
	"zwhe99/simplerl-OlympiadBench::test",
	# Other benchmarks mentioned in the paper decontamination section.
	"zwhe99/pm-en::test",
]

QUESTION_KEYS = (
	"question",
	"problem",
	"query",
	"instruction",
	"prompt",
	"input",
)

ANSWER_KEYS = (
	"final_answer",
	"answer",
	"expected_answer",
	"solution",
	"response",
	"output",
)

TOPIC_KEYS = ("topic", "subject", "type", "category", "domain")

R1_KEYS = ("r1_solution_1", "r1_solution_2", "r1_solution_3")

# 三个主要来源数据集的名称提示，用于自动识别 schema。
MMIQC_SOURCE_HINTS = ("vivacem/mmiqc", "mmiqc")
WEBINSTRUCTSUB_SOURCE_HINTS = ("tiger-lab/webinstructsub", "webinstructsub")
NUMINAMATH_SOURCE_HINTS = ("ai-mo/numinamath-cot", "numinamath-cot", "numinamath")

OPEN_ENDED_PATTERN = re.compile(
	r"\b(prove|show that|explain|why|discuss|derive|justify|analyze|compare)\b",
	flags=re.IGNORECASE,
)

FINAL_ANSWER_PATTERNS = [
	re.compile(r"(?i)final answer\s*[:is]*\s*(.+)$"),
	re.compile(r"(?i)answer\s*[:is]*\s*(.+)$"),
]


def _maybe_tqdm(iterable: Iterable[Any], desc: str) -> Iterable[Any]:
	"""尽量使用 tqdm 展示进度；环境没有 tqdm 时退化为普通迭代。"""
	try:
		from tqdm import tqdm

		return tqdm(iterable, desc=desc)
	except Exception:
		return iterable


def chunked(items: Sequence[Any], chunk_size: int) -> Iterator[Sequence[Any]]:
	"""把大列表按固定大小分块，降低一次性编码/推理内存压力。"""
	for i in range(0, len(items), chunk_size):
		yield items[i : i + chunk_size]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
	"""读取 JSONL 文件为字典列表。"""
	records: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
	"""把字典列表写回 JSONL（每行一条样本）。"""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def first_non_empty(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
	"""按候选键顺序取第一个非空值，适配多数据源异构字段。"""
	for k in keys:
		if k in record and record[k] not in (None, ""):
			return record[k]
	return None


def to_str(value: Any) -> str:
	"""把任意值转成字符串，避免后续正则/拼接出错。"""
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	return str(value)


def normalize_question(text: str) -> str:
	"""问题文本归一化，用于去重和近重复判断。"""
	text = text.lower().strip()
	text = text.replace("\n", " ")
	text = re.sub(r"\\[a-zA-Z]+", " ", text)
	text = re.sub(r"[^a-z0-9]+", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def normalize_answer(text: str) -> str:
	"""答案归一化，弱化格式差异（latex 包装、空格、百分号等）。"""
	text = text.strip()
	text = text.replace("\\left", "").replace("\\right", "")
	text = text.replace("$", "")
	text = text.replace("\\%", "%")
	text = text.replace("%", "")
	text = text.replace(" ", "")
	text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
	text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
	text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
	text = re.sub(r"(^[=:\-]+)|([=:\-]+$)", "", text)
	return text


def parse_source_spec(spec: str, default_split: str) -> Tuple[str, str]:
	"""解析数据源描述: <dataset_or_path>::<split>。"""
	if "::" in spec:
		source, split = spec.split("::", 1)
		return source.strip(), split.strip()
	return spec.strip(), default_split


def stable_id(source_name: str, idx: int, question: str) -> str:
	"""生成稳定样本 ID，便于跨阶段追踪。"""
	key = f"{source_name}::{idx}::{question}".encode("utf-8", errors="ignore")
	return hashlib.sha1(key).hexdigest()[:16]


@dataclass
class Sample:
	"""清洗流程中的统一样本结构。"""
	sample_id: str
	source_name: str
	# 来源 schema 类型: mmiqc/webinstructsub/numinamath/generic
	source_schema: str
	# 来源中的子来源值（论文强调的来源维度）
	source_value: str
	# 统一后的问题字段
	question: str
	# 来源侧解答文本（可用于提取最终答案）
	source_solution: str = ""
	# 来源侧最终答案（原值）
	final_answer: str = ""
	topic: str = ""
	source_index: Optional[int] = None
	difficulty: Optional[float] = None
	r1_solutions: List[str] = field(default_factory=list)
	# 仅保留清洗相关列，便于追溯输入字段
	clean_columns: Dict[str, Any] = field(default_factory=dict)


def detect_source_schema(source_name: str, rec: Dict[str, Any]) -> str:
	"""根据数据集名和字段集合识别来源 schema。"""
	lower_name = source_name.lower()
	keys = set(rec.keys())

	if any(h in lower_name for h in MMIQC_SOURCE_HINTS) or {"instruction", "output", "source"}.issubset(keys):
		return "mmiqc"
	if any(h in lower_name for h in WEBINSTRUCTSUB_SOURCE_HINTS) or {"orig_question", "orig_answer", "question", "answer", "source"}.issubset(keys):
		return "webinstructsub"
	if any(h in lower_name for h in NUMINAMATH_SOURCE_HINTS) or {"source", "problem", "solution", "messages"}.issubset(keys):
		return "numinamath"
	return "generic"


def map_record_to_clean_columns(source_name: str, rec: Dict[str, Any]) -> Dict[str, Any]:
	"""
	把不同来源记录映射到统一字段。

	统一目标字段:
	- question: 问题
	- answer: 答案
	- source_value: 来源
	- source_solution: 解答文本
	- clean_columns: 清洗追溯字段
	"""
	schema = detect_source_schema(source_name, rec)

	if schema == "mmiqc":
		question = to_str(rec.get("instruction", "")).strip()
		answer = to_str(rec.get("output", "")).strip()
		source_value = to_str(rec.get("source", "")).strip()
		return {
			"schema": schema,
			"question": question,
			"answer": answer,
			"source_value": source_value,
			"source_solution": answer,
			"topic": "",
			"index": None,
			"clean_columns": {
				"question": question,
				"answer": answer,
				"source": source_value,
				"instruction": question,
				"output": answer,
			},
		}

	if schema == "webinstructsub":
		question = to_str(rec.get("question", "")).strip() or to_str(rec.get("orig_question", "")).strip()
		answer = to_str(rec.get("answer", "")).strip() or to_str(rec.get("orig_answer", "")).strip()
		source_value = to_str(rec.get("source", "")).strip()
		index_val = rec.get("index", None)
		try:
			index_val = int(index_val) if index_val is not None else None
		except Exception:
			index_val = None
		return {
			"schema": schema,
			"question": question,
			"answer": answer,
			"source_value": source_value,
			"source_solution": answer,
			"topic": "",
			"index": index_val,
			"clean_columns": {
				"question": question,
				"answer": answer,
				"source": source_value,
				"orig_question": to_str(rec.get("orig_question", "")).strip(),
				"orig_answer": to_str(rec.get("orig_answer", "")).strip(),
				"index": index_val,
			},
		}

	if schema == "numinamath":
		question = to_str(rec.get("problem", "")).strip()
		answer = to_str(rec.get("solution", "")).strip()
		source_value = to_str(rec.get("source", "")).strip()
		return {
			"schema": schema,
			"question": question,
			"answer": answer,
			"source_value": source_value,
			"source_solution": answer,
			"topic": "",
			"index": None,
			"clean_columns": {
				"question": question,
				"answer": answer,
				"source": source_value,
				"problem": question,
				"solution": answer,
			},
		}

	question = to_str(first_non_empty(rec, QUESTION_KEYS)).strip()
	answer = to_str(first_non_empty(rec, ANSWER_KEYS)).strip()
	topic = to_str(first_non_empty(rec, TOPIC_KEYS)).strip()
	source_value = to_str(rec.get("source", "")).strip()
	if not source_value:
		source_value = source_name

	return {
		"schema": schema,
		"question": question,
		"answer": answer,
		"source_value": source_value,
		"source_solution": to_str(first_non_empty(rec, ("solution", "response", "output"))).strip(),
		"topic": topic,
		"index": rec.get("index", None),
		"clean_columns": {
			"question": question,
			"answer": answer,
			"source": source_value,
		},
	}


def parse_source_value_allowlists(specs: Sequence[str]) -> Dict[str, set[str]]:
	"""
	解析来源值白名单规则。

	格式示例:
	- schema:mmiqc=math.stackexchange.com,mathoverflow.net
	- source:Vivacem/MMIQC=math.stackexchange.com
	- Vivacem/MMIQC=math.stackexchange.com
	"""
	lookup: Dict[str, set[str]] = {}
	for raw in specs:
		spec = raw.strip()
		if not spec:
			continue
		if "=" not in spec:
			raise ValueError(f"Invalid allowlist spec (missing '='): {raw}")
		key, values = spec.split("=", 1)
		key = key.strip()
		if not key:
			raise ValueError(f"Invalid allowlist spec (empty key): {raw}")
		parsed_values = {v.strip() for v in values.split(",") if v.strip()}
		if not parsed_values:
			raise ValueError(f"Invalid allowlist spec (empty values): {raw}")
		if key not in lookup:
			lookup[key] = set()
		lookup[key].update(parsed_values)
	return lookup


def load_allowlists_from_file(path: str) -> Dict[str, set[str]]:
	"""从 JSON 文件加载来源值白名单。"""
	p = Path(path)
	with p.open("r", encoding="utf-8") as f:
		obj = json.load(f)
	if not isinstance(obj, dict):
		raise ValueError("Allowlist file must be a JSON object.")
	lookup: Dict[str, set[str]] = {}
	for key, vals in obj.items():
		if not isinstance(key, str):
			continue
		if isinstance(vals, str):
			parsed = {vals.strip()} if vals.strip() else set()
		elif isinstance(vals, list):
			parsed = {to_str(v).strip() for v in vals if to_str(v).strip()}
		else:
			continue
		if parsed:
			lookup[key.strip()] = parsed
	return lookup


def merge_allowlists(*lookups: Dict[str, set[str]]) -> Dict[str, set[str]]:
	"""合并多个白名单来源（CLI + 文件），相同 key 做并集。"""
	out: Dict[str, set[str]] = {}
	for lk in lookups:
		for key, vals in lk.items():
			if key not in out:
				out[key] = set()
			out[key].update(vals)
	return out


def source_value_allowed(sample: Sample, allowlists: Dict[str, set[str]]) -> bool:
	"""判断样本是否命中白名单。优先级: source 规则 > schema 规则。"""
	if not allowlists:
		return True

	# Specific source rule has highest priority.
	for key in (f"source:{sample.source_name}", sample.source_name):
		if key in allowlists:
			return sample.source_value in allowlists[key]

	# Schema-level rule as fallback.
	schema_key = f"schema:{sample.source_schema}"
	if schema_key in allowlists:
		return sample.source_value in allowlists[schema_key]

	# No matching rule means keep.
	return True


def filter_by_source_allowlist(
	samples: Sequence[Sample],
	allowlists: Dict[str, set[str]],
) -> Tuple[List[Sample], Dict[str, Any]]:
	"""按来源白名单进行预过滤，统计过滤前后数量。"""
	if not allowlists:
		return list(samples), {"enabled": False, "input": len(samples), "kept": len(samples), "dropped": 0}

	kept: List[Sample] = []
	dropped = 0
	for s in samples:
		if source_value_allowed(s, allowlists):
			kept.append(s)
		else:
			dropped += 1

	stats = {
		"enabled": True,
		"rules": {k: sorted(list(v)) for k, v in allowlists.items()},
		"input": len(samples),
		"kept": len(kept),
		"dropped": dropped,
	}
	return kept, stats


def summarize_samples_by_source(samples: Sequence[Sample]) -> Dict[str, Any]:
	"""统计样本在 source_name/source_schema/source_value 的分布。"""
	by_source_name: Dict[str, int] = {}
	by_schema: Dict[str, int] = {}
	by_source_value: Dict[str, Dict[str, int]] = {}

	for s in samples:
		by_source_name[s.source_name] = by_source_name.get(s.source_name, 0) + 1
		by_schema[s.source_schema] = by_schema.get(s.source_schema, 0) + 1
		if s.source_name not in by_source_value:
			by_source_value[s.source_name] = {}
		by_source_value[s.source_name][s.source_value] = (
			by_source_value[s.source_name].get(s.source_value, 0) + 1
		)

	return {
		"total": len(samples),
		"by_source_name": dict(sorted(by_source_name.items(), key=lambda x: x[0])),
		"by_source_schema": dict(sorted(by_schema.items(), key=lambda x: x[0])),
		"by_source_value": {
			k: dict(sorted(v.items(), key=lambda x: x[0]))
			for k, v in sorted(by_source_value.items(), key=lambda x: x[0])
		},
	}


def summarize_records_by_source(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""统计最终记录按来源的构成（用于 summary 输出）。"""
	by_source_name: Dict[str, int] = {}
	by_source_value: Dict[str, Dict[str, int]] = {}

	for r in records:
		source_name = to_str(r.get("source_name", "unknown"))
		source_value = to_str(r.get("source", ""))
		by_source_name[source_name] = by_source_name.get(source_name, 0) + 1
		if source_name not in by_source_value:
			by_source_value[source_name] = {}
		by_source_value[source_name][source_value] = by_source_value[source_name].get(source_value, 0) + 1

	return {
		"total": len(records),
		"by_source_name": dict(sorted(by_source_name.items(), key=lambda x: x[0])),
		"by_source_value": {
			k: dict(sorted(v.items(), key=lambda x: x[0]))
			for k, v in sorted(by_source_value.items(), key=lambda x: x[0])
		},
	}


def load_from_hf(dataset_id: str, split: str) -> List[Dict[str, Any]]:
	"""从 HuggingFace 数据集加载指定 split。"""
	try:
		import datasets
	except Exception as exc:
		raise RuntimeError(
			"Missing dependency 'datasets'. Install with: pip install datasets"
		) from exc

	ds = datasets.load_dataset(dataset_id, split=split)
	return [dict(row) for row in ds]


def load_source_records(source_spec: str, default_split: str) -> Tuple[str, List[Dict[str, Any]]]:
	"""统一加载入口：支持本地 json/jsonl/parquet 与 HF 数据集。"""
	source_name, split = parse_source_spec(source_spec, default_split)
	path = Path(source_name)

	if path.exists():
		if path.suffix.lower() == ".jsonl":
			return source_name, read_jsonl(path)
		if path.suffix.lower() == ".json":
			with path.open("r", encoding="utf-8") as f:
				data = json.load(f)
			if not isinstance(data, list):
				raise ValueError(f"JSON source must be a list: {source_name}")
			return source_name, data
		if path.suffix.lower() == ".parquet":
			try:
				import datasets
			except Exception as exc:
				raise RuntimeError(
					"Missing dependency 'datasets' for parquet loading."
				) from exc
			ds = datasets.load_dataset("parquet", data_files=str(path), split="train")
			return source_name, [dict(row) for row in ds]
		raise ValueError(f"Unsupported local source format: {source_name}")

	records = load_from_hf(source_name, split)
	return source_name, records


def adapt_records(source_name: str, records: Sequence[Dict[str, Any]], require_answer: bool = True) -> List[Sample]:
	"""
	把原始记录转换成 Sample。

	require_answer=True 时只保留带答案样本（训练清洗主流程）。
	require_answer=False 主要用于 benchmark 问题池构建。
	"""
	samples: List[Sample] = []
	for idx, rec in enumerate(records):
		mapped = map_record_to_clean_columns(source_name, rec)
		source_schema = to_str(mapped.get("schema", "generic")).strip() or "generic"
		question = to_str(mapped.get("question", "")).strip()
		if not question:
			continue

		final_answer = to_str(mapped.get("answer", "")).strip()
		if require_answer and not final_answer:
			continue

		source_solution = to_str(mapped.get("source_solution", "")).strip()
		if not source_solution:
			source_solution = final_answer

		topic = to_str(mapped.get("topic", "")).strip()
		source_value = to_str(mapped.get("source_value", "")).strip() or source_name

		source_index = mapped.get("index", None)
		try:
			source_index = int(source_index) if source_index is not None else None
		except Exception:
			source_index = None

		difficulty_val = first_non_empty(rec, ("difficulty", "level"))
		difficulty: Optional[float] = None
		if difficulty_val is not None:
			try:
				difficulty = float(difficulty_val)
			except Exception:
				difficulty = None

		r1_solutions = [to_str(rec.get(k, "")).strip() for k in R1_KEYS]
		r1_solutions = [s for s in r1_solutions if s]

		samples.append(
			Sample(
				sample_id=stable_id(source_name, idx, question),
				source_name=source_name,
				source_schema=source_schema,
				source_value=source_value,
				question=question,
				source_solution=source_solution,
				final_answer=final_answer,
				topic=topic,
				source_index=source_index,
				difficulty=difficulty,
				r1_solutions=r1_solutions,
				clean_columns=mapped.get("clean_columns", {}),
			)
		)
	return samples


def stage1_basic_filter(
	samples: Sequence[Sample],
	min_question_chars: int,
	max_question_chars: int,
	min_tokens: int,
) -> Tuple[List[Sample], Dict[str, int]]:
	"""Stage1: 基础规则过滤 + 归一化精确去重。"""
	kept: List[Sample] = []
	stats = Counter()
	seen_norm = set()

	for s in samples:
		q = s.question.strip()
		if len(q) < min_question_chars:
			stats["drop_too_short"] += 1
			continue
		if len(q) > max_question_chars:
			stats["drop_too_long"] += 1
			continue
		if len(q.split()) < min_tokens:
			stats["drop_few_tokens"] += 1
			continue
		norm_q = normalize_question(q)
		if not norm_q:
			stats["drop_empty_after_norm"] += 1
			continue
		if norm_q in seen_norm:
			stats["drop_exact_duplicate"] += 1
			continue
		seen_norm.add(norm_q)
		kept.append(s)

	stats["input"] = len(samples)
	stats["kept"] = len(kept)
	return kept, dict(stats)


class EmbeddingEncoder:
	"""句向量编码器（Stage2 检索候选污染样本）。"""
	def __init__(self, model_name: str, batch_size: int, device: Optional[str]) -> None:
		try:
			sentence_transformers = importlib.import_module("sentence_transformers")
			SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
		except Exception as exc:
			raise RuntimeError(
				"Missing dependency 'sentence-transformers'. Install with: "
				"pip install sentence-transformers"
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


class OpenAICompatibleClient:
	"""兼容 OpenAI Chat Completions 协议的最小客户端。"""
	def __init__(self, base_url: str, api_key: str, timeout: int = 120) -> None:
		if not base_url:
			raise ValueError("base_url is required")
		if not api_key:
			raise ValueError("api_key is required")

		try:
			import requests
		except Exception as exc:
			raise RuntimeError(
				"Missing dependency 'requests'. Install with: pip install requests"
			) from exc

		self.requests = requests
		self.base_url = base_url.rstrip("/")
		self.api_key = api_key
		self.timeout = timeout

	def chat(
		self,
		model: str,
		system_prompt: str,
		user_prompt: str,
		temperature: float = 0.0,
		max_tokens: int = 512,
	) -> str:
		url = f"{self.base_url}/chat/completions"
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}
		payload = {
			"model": model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": temperature,
			"max_tokens": max_tokens,
		}
		response = self.requests.post(url, headers=headers, json=payload, timeout=self.timeout)
		response.raise_for_status()
		data = response.json()
		return data["choices"][0]["message"]["content"]


def cosine_topk(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
	"""基于余弦相似度做 top-k 召回。"""
	if corpus_emb.shape[0] == 0:
		return np.zeros((query_emb.shape[0], 0), dtype=np.int32), np.zeros((query_emb.shape[0], 0), dtype=np.float32)

	k = min(k, corpus_emb.shape[0])
	sims = query_emb @ corpus_emb.T
	part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
	part_scores = np.take_along_axis(sims, part, axis=1)
	order = np.argsort(-part_scores, axis=1)
	top_idx = np.take_along_axis(part, order, axis=1)
	top_scores = np.take_along_axis(part_scores, order, axis=1)
	return top_idx.astype(np.int32), top_scores.astype(np.float32)


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
	"""从 LLM 自由文本中提取 JSON 对象。"""
	left = text.find("{")
	right = text.rfind("}")
	if left < 0 or right <= left:
		return None
	try:
		return json.loads(text[left : right + 1])
	except Exception:
		return None


def heuristic_is_duplicate(q1: str, q2: str, ratio_threshold: float = 0.9, jaccard_threshold: float = 0.85) -> bool:
	"""无 LLM 时的近重复启发式判定（编辑距离 + Jaccard）。"""
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


def llm_duplicate_judge(
	client: OpenAICompatibleClient,
	model: str,
	candidate_question: str,
	benchmark_question: str,
) -> bool:
	"""LLM 语义判重：判断候选题是否与 benchmark 题同题/改写。"""
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
	text = client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
	payload = parse_json_from_text(text)
	if payload is None:
		return heuristic_is_duplicate(candidate_question, benchmark_question)
	return bool(payload.get("duplicate_or_paraphrase", False))


def stage2_decontaminate(
	candidates: Sequence[Sample],
	benchmark_questions: Sequence[str],
	encoder: EmbeddingEncoder,
	top_k: int,
	min_similarity: float,
	judge_client: Optional[OpenAICompatibleClient],
	judge_model: str,
	chunk_size: int,
) -> Tuple[List[Sample], Dict[str, Any]]:
	"""Stage2: 语义检索 + 判重去污染。"""
	if not benchmark_questions:
		return list(candidates), {"kept": len(candidates), "dropped_contaminated": 0, "note": "no benchmarks"}

	unique_bench = list(dict.fromkeys([b for b in benchmark_questions if b.strip()]))
	bench_emb = encoder.encode(unique_bench)

	kept: List[Sample] = []
	dropped = 0
	judged_pairs = 0

	candidate_list = list(candidates)
	for chunk in _maybe_tqdm(list(chunked(candidate_list, chunk_size)), desc="Stage2 decontamination"):
		q_texts = [s.question for s in chunk]
		q_emb = encoder.encode(q_texts)
		idx_mat, sim_mat = cosine_topk(q_emb, bench_emb, k=top_k)

		for row_i, sample in enumerate(chunk):
			contaminated = False
			for cand_i, sim in zip(idx_mat[row_i].tolist(), sim_mat[row_i].tolist()):
				if sim < min_similarity:
					continue
				judged_pairs += 1
				benchmark_q = unique_bench[cand_i]
				if judge_client is not None:
					try:
						is_dup = llm_duplicate_judge(
							judge_client,
							judge_model,
							sample.question,
							benchmark_q,
						)
					except Exception:
						is_dup = heuristic_is_duplicate(sample.question, benchmark_q)
				else:
					is_dup = heuristic_is_duplicate(sample.question, benchmark_q)

				if is_dup:
					contaminated = True
					break

			if contaminated:
				dropped += 1
			else:
				kept.append(sample)

	stats = {
		"input": len(candidates),
		"kept": len(kept),
		"dropped_contaminated": dropped,
		"benchmarks": len(unique_bench),
		"judged_pairs": judged_pairs,
		"top_k": top_k,
		"min_similarity": min_similarity,
	}
	return kept, stats


def parse_difficulty_score(text: str) -> Optional[float]:
	"""从模型输出中提取 1-10 难度分数。"""
	m = re.findall(r"(?<!\d)(10|[1-9])(?!\d)", text)
	if not m:
		return None
	try:
		return float(m[-1])
	except Exception:
		return None


def heuristic_difficulty(question: str) -> float:
	"""无 LLM 时的难度估计：基于长度与符号密度。"""
	tokens = question.split()
	length_score = min(5.0, math.log2(len(tokens) + 1.0))
	symbol_hits = len(re.findall(r"[=<>\\^_{}()\[\]/]", question))
	symbol_score = min(3.0, symbol_hits / 8.0)
	question_marks = 1.0 if "?" in question else 0.0
	base = 2.0 + length_score + symbol_score + question_marks
	return float(max(1.0, min(10.0, base)))


def llm_rate_difficulty(
	client: OpenAICompatibleClient,
	model: str,
	question: str,
	rounds: int,
) -> Optional[float]:
	"""LLM 多次打分并取平均，复现论文中多次评估思想。"""
	system_prompt = (
		"You are a math problem difficulty rater using AoPS-style levels from 1 to 10. "
		"Return only one integer in [1, 10]."
	)
	user_prompt = (
		"Rate this question difficulty from 1 to 10. Return only one integer.\n\n"
		f"Question:\n{question}"
	)

	votes: List[float] = []
	for _ in range(rounds):
		try:
			text = client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
			score = parse_difficulty_score(text)
			if score is not None:
				votes.append(score)
		except Exception:
			continue

	if not votes:
		return None
	return float(sum(votes) / len(votes))


def stage3_difficulty_filter(
	samples: Sequence[Sample],
	threshold: float,
	prefer_existing: bool,
	llm_client: Optional[OpenAICompatibleClient],
	difficulty_model: str,
	num_votes: int,
) -> Tuple[List[Sample], Dict[str, Any]]:
	"""Stage3: 难度评估与阈值筛选（默认保留 difficulty >= threshold）。"""
	kept: List[Sample] = []
	dropped = 0
	used_existing = 0
	used_llm = 0
	used_heuristic = 0

	for sample in _maybe_tqdm(samples, desc="Stage3 difficulty"):
		score: Optional[float] = None

		if prefer_existing and sample.difficulty is not None:
			score = float(sample.difficulty)
			used_existing += 1

		if score is None and llm_client is not None:
			score = llm_rate_difficulty(
				client=llm_client,
				model=difficulty_model,
				question=sample.question,
				rounds=num_votes,
			)
			if score is not None:
				used_llm += 1

		if score is None:
			score = heuristic_difficulty(sample.question)
			used_heuristic += 1

		sample.difficulty = float(score)
		if score >= threshold:
			kept.append(sample)
		else:
			dropped += 1

	stats = {
		"input": len(samples),
		"kept": len(kept),
		"dropped_below_threshold": dropped,
		"threshold": threshold,
		"used_existing": used_existing,
		"used_llm": used_llm,
		"used_heuristic": used_heuristic,
	}
	return kept, stats


def extract_last_boxed(text: str) -> Optional[str]:
	"""提取最后一个 \boxed{} 内容。"""
	idx = text.rfind("\\boxed{")
	if idx < 0:
		return None

	i = idx
	depth = 0
	right = None
	while i < len(text):
		if text[i] == "{":
			depth += 1
		elif text[i] == "}":
			depth -= 1
			if depth == 0:
				right = i
				break
		i += 1

	if right is None:
		return None

	boxed = text[idx : right + 1]
	boxed = re.sub(r"^\\boxed\{", "", boxed)
	if boxed.endswith("}"):
		boxed = boxed[:-1]
	return boxed.strip()


def extract_final_answer_from_solution(solution_text: str) -> Optional[str]:
	"""从解答文本中抽取最终答案（boxed/answer pattern/数字回退）。"""
	if not solution_text:
		return None

	boxed = extract_last_boxed(solution_text)
	if boxed:
		return boxed

	lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
	if lines:
		last = lines[-1]
		for pat in FINAL_ANSWER_PATTERNS:
			m = pat.search(last)
			if m:
				return m.group(1).strip()

	for pat in FINAL_ANSWER_PATTERNS:
		m_all = pat.findall(solution_text)
		if m_all:
			return to_str(m_all[-1]).strip()

	num = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", solution_text)
	if num:
		return num[-1]
	return None


def llm_filter_and_format_question(
	client: OpenAICompatibleClient,
	model: str,
	question: str,
) -> Tuple[bool, str, str]:
	"""LLM 判断题目是否可规则验证，并做格式化改写。"""
	system_prompt = (
		"You convert math questions into a standardized format for rule-based verification. "
		"Discard questions that do not have a single verifiable numeric/symbolic final answer."
	)
	user_prompt = (
		"Question:\n"
		f"{question}\n\n"
		"Return JSON only with schema:\n"
		'{"keep": true/false, "formatted_question": "...", "reason": "..."}'
	)
	text = client.chat(model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
	payload = parse_json_from_text(text)
	if payload is None:
		keep = not bool(OPEN_ENDED_PATTERN.search(question))
		return keep, question.strip(), "fallback_no_json"
	keep = bool(payload.get("keep", False))
	formatted_question = to_str(payload.get("formatted_question", question)).strip() or question.strip()
	reason = to_str(payload.get("reason", ""))
	return keep, formatted_question, reason


def heuristic_filter_and_format_question(question: str) -> Tuple[bool, str, str]:
	"""无 LLM 时的问题可验证性过滤回退逻辑。"""
	if OPEN_ENDED_PATTERN.search(question):
		return False, question, "heuristic_open_ended"
	formatted = re.sub(r"\s+", " ", question).strip()
	return True, formatted, "heuristic_keep"


def llm_generate_solutions(
	client: OpenAICompatibleClient,
	model: str,
	question: str,
	n: int,
) -> List[str]:
	"""按题目生成 n 条解答，用于答案一致性校验。"""
	system_prompt = (
		"You are a math solver. Provide complete reasoning and end with a line: "
		"Final Answer: <answer>."
	)
	outputs: List[str] = []
	for _ in range(n):
		user_prompt = f"Solve:\n{question}"
		text = client.chat(
			model=model,
			system_prompt=system_prompt,
			user_prompt=user_prompt,
			temperature=0.7,
			max_tokens=2048,
		)
		outputs.append(text)
	return outputs


def stage4_answer_verification(
	samples: Sequence[Sample],
	llm_client: Optional[OpenAICompatibleClient],
	formatter_model: str,
	generator_model: str,
	generate_missing_r1: bool,
	min_consensus_answers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	"""Stage4: 可验证题过滤 + 多解答答案一致性筛选。"""
	final_records: List[Dict[str, Any]] = []
	dropped_question = 0
	dropped_consistency = 0
	generated_solutions = 0

	for sample in _maybe_tqdm(samples, desc="Stage4 verify"):
		if llm_client is not None:
			try:
				keep_q, formatted_q, _ = llm_filter_and_format_question(
					llm_client,
					formatter_model,
					sample.question,
				)
			except Exception:
				keep_q, formatted_q, _ = heuristic_filter_and_format_question(sample.question)
		else:
			keep_q, formatted_q, _ = heuristic_filter_and_format_question(sample.question)

		if not keep_q:
			dropped_question += 1
			continue

		solutions = list(sample.r1_solutions)
		if len(solutions) < 3 and generate_missing_r1 and llm_client is not None:
			need = 3 - len(solutions)
			try:
				solutions.extend(llm_generate_solutions(llm_client, generator_model, formatted_q, need))
				generated_solutions += need
			except Exception:
				pass

		solutions = solutions[:3]

		extracted: List[str] = []

		if sample.final_answer:
			extracted.append(sample.final_answer)
		elif sample.source_solution:
			src = extract_final_answer_from_solution(sample.source_solution)
			if src:
				extracted.append(src)

		for sol in solutions:
			ans = extract_final_answer_from_solution(sol)
			if ans:
				extracted.append(ans)

		normalized = [normalize_answer(x) for x in extracted if normalize_answer(x)]

		if len(normalized) < min_consensus_answers:
			dropped_consistency += 1
			continue

		if len(set(normalized)) != 1:
			dropped_consistency += 1
			continue

		final_answer = normalized[0]
		if not final_answer:
			dropped_consistency += 1
			continue

		final_records.append(
			{
				"question": formatted_q,
				"final_answer": final_answer,
				"source_answer": sample.final_answer,
				"difficulty": float(sample.difficulty) if sample.difficulty is not None else None,
				"topic": sample.topic,
				"r1_solution_1": solutions[0] if len(solutions) > 0 else "",
				"r1_solution_2": solutions[1] if len(solutions) > 1 else "",
				"r1_solution_3": solutions[2] if len(solutions) > 2 else "",
				"source_name": sample.source_name,
				"source_schema": sample.source_schema,
				"source": sample.source_value,
				"source_index": sample.source_index,
				"sample_id": sample.sample_id,
				"clean_columns": sample.clean_columns,
			}
		)

	stats = {
		"input": len(samples),
		"kept": len(final_records),
		"dropped_question_filter": dropped_question,
		"dropped_consistency": dropped_consistency,
		"generated_missing_solutions": generated_solutions,
		"min_consensus_answers": min_consensus_answers,
	}
	return final_records, stats


def build_benchmark_questions(
	benchmark_specs: Sequence[str],
	default_split: str,
	skip_missing: bool,
) -> List[str]:
	"""构建 benchmark 问题池（用于 Stage2 召回比对）。"""
	all_questions: List[str] = []
	for spec in benchmark_specs:
		try:
			source_name, records = load_source_records(spec, default_split)
			samples = adapt_records(source_name, records, require_answer=False)
			all_questions.extend([s.question for s in samples if s.question.strip()])
		except Exception as exc:
			if skip_missing:
				print(f"[warn] Skip benchmark {spec}: {exc}", file=sys.stderr)
				continue
			raise
	return all_questions


def maybe_build_llm_client(args: argparse.Namespace) -> Optional[OpenAICompatibleClient]:
	"""根据参数决定是否启用 LLM 客户端。"""
	if args.disable_llm:
		return None
	if not args.llm_base_url or not args.llm_api_key:
		return None
	try:
		return OpenAICompatibleClient(
			base_url=args.llm_base_url,
			api_key=args.llm_api_key,
			timeout=args.llm_timeout,
		)
	except Exception as exc:
		print(f"[warn] Failed to create LLM client. Falling back to heuristics. error={exc}", file=sys.stderr)
		return None


def save_stage(path: Path, samples: Sequence[Sample]) -> None:
	"""保存阶段中间结果，便于复现实验与做来源保留率分析。"""
	records = []
	for s in samples:
		records.append(
			{
				"sample_id": s.sample_id,
				"source_name": s.source_name,
				"source_schema": s.source_schema,
				"source": s.source_value,
				"source_index": s.source_index,
				"question": s.question,
				"source_solution": s.source_solution,
				"source_answer": s.final_answer,
				"topic": s.topic,
				"difficulty": s.difficulty,
				"r1_solutions": s.r1_solutions,
				"clean_columns": s.clean_columns,
			}
		)
	write_jsonl(path, records)


def main() -> None:
	"""命令行主入口。"""
	parser = argparse.ArgumentParser(description="DeepMath-style cleaning pipeline reimplementation")

	parser.add_argument(
		"--source",
		action="append",
		required=True,
		help="Source dataset spec. Format: <hf_dataset_id>::<split> or local file path (.jsonl/.json/.parquet).",
	)
	parser.add_argument("--source-default-split", default="train")
	parser.add_argument(
		"--source-value-allowlist",
		action="append",
		default=[],
		help=(
			"Allowlist rules for source values. Format: "
			"schema:<name>=v1,v2 or source:<dataset>=v1,v2 or <dataset>=v1,v2"
		),
	)
	parser.add_argument(
		"--source-value-allowlist-file",
		default=None,
		help="JSON file for source allowlists. Example: {\"schema:mmiqc\": [\"math.stackexchange.com\"]}",
	)
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--max-samples", type=int, default=None)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--use-default-benchmarks", action="store_true")
	parser.add_argument(
		"--benchmark",
		action="append",
		default=[],
		help="Benchmark spec. Format: <hf_dataset_id>::<split> or local file path.",
	)
	parser.add_argument("--benchmark-default-split", default="test")
	parser.add_argument("--skip-missing-benchmarks", action="store_true")

	parser.add_argument("--min-question-chars", type=int, default=20)
	parser.add_argument("--max-question-chars", type=int, default=3000)
	parser.add_argument("--min-question-tokens", type=int, default=4)

	parser.add_argument(
		"--embedding-model",
		default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
	)
	parser.add_argument("--embedding-batch-size", type=int, default=128)
	parser.add_argument("--embedding-device", default=None)
	parser.add_argument("--decontam-top-k", type=int, default=5)
	parser.add_argument("--decontam-min-similarity", type=float, default=0.55)
	parser.add_argument("--decontam-chunk-size", type=int, default=256)

	parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
	parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
	parser.add_argument("--llm-timeout", type=int, default=120)
	parser.add_argument("--disable-llm", action="store_true")

	parser.add_argument("--judge-model", default="meta-llama/Llama-3.3-70B-Instruct")
	parser.add_argument("--difficulty-model", default="gpt-4o")
	parser.add_argument("--formatter-model", default="gpt-4o")
	parser.add_argument("--generator-model", default="deepseek-reasoner")

	parser.add_argument("--difficulty-threshold", type=float, default=5.0)
	parser.add_argument("--difficulty-num-votes", type=int, default=6)
	parser.add_argument("--ignore-existing-difficulty", action="store_true")

	parser.add_argument("--generate-missing-r1", action="store_true")
	parser.add_argument("--min-consensus-answers", type=int, default=4)
	parser.add_argument("--save-intermediate", action="store_true")

	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1) 加载并统一来源格式
	print("[1/7] Loading source datasets")
	all_samples: List[Sample] = []
	for source_spec in args.source:
		source_name, records = load_source_records(source_spec, args.source_default_split)
		samples = adapt_records(source_name, records)
		all_samples.extend(samples)
		print(f"  - loaded {len(samples)} samples from {source_name}")

	if args.max_samples is not None:
		all_samples = all_samples[: args.max_samples]
		print(f"  - truncated to max_samples={args.max_samples}")

	# 2) 统计输入分布，并按来源白名单做前置过滤
	input_distribution = summarize_samples_by_source(all_samples)

	if args.save_intermediate:
		save_stage(output_dir / "stage0_loaded.jsonl", all_samples)

	cli_allowlists = parse_source_value_allowlists(args.source_value_allowlist)
	file_allowlists: Dict[str, set[str]] = {}
	if args.source_value_allowlist_file:
		file_allowlists = load_allowlists_from_file(args.source_value_allowlist_file)
	merged_allowlists = merge_allowlists(cli_allowlists, file_allowlists)

	print("[2/7] Source value allowlist filtering")
	filtered_samples, source_filter_stats = filter_by_source_allowlist(all_samples, merged_allowlists)
	print(f"  - source filter kept {len(filtered_samples)} / {len(all_samples)}")

	# 3) Stage1: 基础质量过滤与去重
	print("[3/7] Stage 1 basic filtering")
	stage1_samples, stage1_stats = stage1_basic_filter(
		filtered_samples,
		min_question_chars=args.min_question_chars,
		max_question_chars=args.max_question_chars,
		min_tokens=args.min_question_tokens,
	)
	print(f"  - stage1 kept {len(stage1_samples)} / {len(filtered_samples)}")

	if args.save_intermediate:
		save_stage(output_dir / "stage1_basic_filtered.jsonl", stage1_samples)

	benchmark_specs: List[str] = list(args.benchmark)
	if args.use_default_benchmarks:
		benchmark_specs.extend(DEFAULT_BENCHMARK_SPECS)

	# 4) 构建 benchmark 问题池
	print("[4/7] Building benchmark pool")
	benchmark_questions = build_benchmark_questions(
		benchmark_specs,
		default_split=args.benchmark_default_split,
		skip_missing=args.skip_missing_benchmarks,
	)
	print(f"  - benchmark questions: {len(benchmark_questions)}")

	llm_client = maybe_build_llm_client(args)
	encoder = EmbeddingEncoder(
		model_name=args.embedding_model,
		batch_size=args.embedding_batch_size,
		device=args.embedding_device,
	)

	# 5) Stage2: 去污染
	print("[5/7] Stage 2 semantic decontamination")
	stage2_samples, stage2_stats = stage2_decontaminate(
		candidates=stage1_samples,
		benchmark_questions=benchmark_questions,
		encoder=encoder,
		top_k=args.decontam_top_k,
		min_similarity=args.decontam_min_similarity,
		judge_client=llm_client,
		judge_model=args.judge_model,
		chunk_size=args.decontam_chunk_size,
	)
	print(f"  - stage2 kept {len(stage2_samples)} / {len(stage1_samples)}")

	if args.save_intermediate:
		save_stage(output_dir / "stage2_decontaminated.jsonl", stage2_samples)

	# 6) Stage3: 难度筛选
	print("[6/7] Stage 3 difficulty filtering")
	stage3_samples, stage3_stats = stage3_difficulty_filter(
		samples=stage2_samples,
		threshold=args.difficulty_threshold,
		prefer_existing=not args.ignore_existing_difficulty,
		llm_client=llm_client,
		difficulty_model=args.difficulty_model,
		num_votes=args.difficulty_num_votes,
	)
	print(f"  - stage3 kept {len(stage3_samples)} / {len(stage2_samples)}")

	if args.save_intermediate:
		save_stage(output_dir / "stage3_difficulty_filtered.jsonl", stage3_samples)

	# 7) Stage4: 答案一致性验证
	print("[7/7] Stage 4 answer verifiability filtering")
	final_records, stage4_stats = stage4_answer_verification(
		samples=stage3_samples,
		llm_client=llm_client,
		formatter_model=args.formatter_model,
		generator_model=args.generator_model,
		generate_missing_r1=args.generate_missing_r1,
		min_consensus_answers=args.min_consensus_answers,
	)
	print(f"  - stage4 kept {len(final_records)} / {len(stage3_samples)}")

	final_path = output_dir / "deepmath_curated.jsonl"
	write_jsonl(final_path, final_records)

	# 汇总输出: 阶段统计 + 来源分布 + 最终文件路径
	summary = {
		"input_samples": len(all_samples),
		"source_filter": source_filter_stats,
		"stage1": stage1_stats,
		"stage2": stage2_stats,
		"stage3": stage3_stats,
		"stage4": stage4_stats,
		"source_distribution": {
			"stage0_input": input_distribution,
			"stage1": summarize_samples_by_source(stage1_samples),
			"stage2": summarize_samples_by_source(stage2_samples),
			"stage3": summarize_samples_by_source(stage3_samples),
			"final": summarize_records_by_source(final_records),
		},
		"final_output": str(final_path),
		"final_count": len(final_records),
		"config": vars(args),
		"timestamp": int(time.time()),
	}
	with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2, ensure_ascii=False)

	print("Done.")
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
