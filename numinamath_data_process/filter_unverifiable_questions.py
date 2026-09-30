#!/usr/bin/env python3
"""
Filter out non-verifiable math questions from source datasets.

Default sources:
- Vivacem/MMIQC
- TIGER-Lab/WebInstructSub
- AI-MO/NuminaMath-CoT

Verification logic:
1) Question-level verifiability check (heuristic by default, optional LLM formatter).
2) Extract answer candidates from source answer/solution.
3) Keep only samples with exactly one normalized answer candidate.

This script is intentionally lightweight and focuses only on verifiability filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepmath_cleaning_pipeline import (
    Sample,
    _maybe_tqdm,
    adapt_records,
    extract_final_answer_from_solution,
    heuristic_filter_and_format_question,
    llm_filter_and_format_question,
    load_source_records,
    maybe_build_llm_client,
    normalize_answer,
    to_str,
    write_jsonl,
)


DEFAULT_SOURCES = [
    "Vivacem/MMIQC",
    "TIGER-Lab/WebInstructSub",
    "AI-MO/NuminaMath-CoT",
]

# A few weak indicators for chain-of-thought style long text answers.
LONG_FORM_HINTS = re.compile(
    r"\b(let|suppose|consider|therefore|hence|because|first|second|then|proof)\b",
    flags=re.IGNORECASE,
)


def safe_name(source_name: str) -> str:
    """Convert dataset/source name to a safe filename token."""
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", source_name).strip("_")
    return name or "source"


def looks_like_direct_answer(text: str, max_len: int = 128) -> bool:
    """
    Heuristic gate for short direct answers.

    If text is too long or obviously explanation-style, we avoid using it as
    direct answer and rely on extraction logic instead.
    """
    text = text.strip()
    if not text:
        return False
    if len(text) > max_len:
        return False
    if LONG_FORM_HINTS.search(text):
        return False

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 2:
        return False
    return True


def collect_answer_candidates(sample: Sample) -> Dict[str, str]:
    """
    Build normalized answer candidates from sample fields.

    Returns a mapping:
    - key: normalized answer
    - value: representative raw answer text
    """
    normalized_to_best_raw: Dict[str, Tuple[int, str]] = {}

    def add_candidate(raw: str, priority: int) -> None:
        raw = to_str(raw).strip()
        if not raw:
            return
        norm = normalize_answer(raw)
        if not norm:
            return
        prev = normalized_to_best_raw.get(norm)
        if prev is None or priority < prev[0]:
            normalized_to_best_raw[norm] = (priority, raw)

    source_answer = to_str(sample.final_answer)
    source_solution = to_str(sample.source_solution)

    if source_answer:
        extracted = extract_final_answer_from_solution(source_answer)
        if extracted:
            add_candidate(extracted, priority=0)
        if looks_like_direct_answer(source_answer):
            add_candidate(source_answer, priority=1)

    if source_solution:
        extracted = extract_final_answer_from_solution(source_solution)
        if extracted:
            add_candidate(extracted, priority=0)

    return {norm: raw for norm, (_, raw) in normalized_to_best_raw.items()}


def verify_question(
    question: str,
    llm_client: Optional[Any],
    formatter_model: str,
) -> Tuple[bool, str, str]:
    """Check if question is verifiable and return formatted question text."""
    if llm_client is not None:
        try:
            return llm_filter_and_format_question(
                llm_client,
                formatter_model,
                question,
            )
        except Exception:
            pass

    return heuristic_filter_and_format_question(question)


def filter_one_source(
    source_spec: str,
    default_split: str,
    llm_client: Optional[Any],
    formatter_model: str,
    max_samples: Optional[int],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load one source and run verifiability-only filtering."""
    source_name, records = load_source_records(source_spec, default_split)
    samples = adapt_records(source_name, records, require_answer=False)

    if max_samples is not None:
        samples = samples[:max_samples]

    kept_records: List[Dict[str, Any]] = []
    dropped_records: List[Dict[str, Any]] = []
    stats = Counter()

    for sample in _maybe_tqdm(samples, desc=f"Verify filter {source_name}"):
        stats["input"] += 1

        keep_q, formatted_q, question_reason = verify_question(
            question=sample.question,
            llm_client=llm_client,
            formatter_model=formatter_model,
        )
        if not keep_q:
            stats["dropped_question_unverifiable"] += 1
            dropped_records.append(
                {
                    "sample_id": sample.sample_id,
                    "source_name": sample.source_name,
                    "source": sample.source_value,
                    "question": sample.question,
                    "drop_reason": "question_unverifiable",
                    "detail": question_reason,
                }
            )
            continue

        answers = collect_answer_candidates(sample)
        if not answers:
            stats["dropped_no_extractable_answer"] += 1
            dropped_records.append(
                {
                    "sample_id": sample.sample_id,
                    "source_name": sample.source_name,
                    "source": sample.source_value,
                    "question": sample.question,
                    "drop_reason": "no_extractable_answer",
                }
            )
            continue

        if len(answers) > 1:
            stats["dropped_ambiguous_answer"] += 1
            dropped_records.append(
                {
                    "sample_id": sample.sample_id,
                    "source_name": sample.source_name,
                    "source": sample.source_value,
                    "question": sample.question,
                    "drop_reason": "ambiguous_answer",
                    "answer_candidates": sorted(list(answers.values())),
                }
            )
            continue

        normalized_answer, canonical_answer = next(iter(answers.items()))
        kept_records.append(
            {
                "question": formatted_q,
                "answer": canonical_answer,
                "normalized_answer": normalized_answer,
                "source": sample.source_value,
                "source_name": sample.source_name,
                "source_schema": sample.source_schema,
                "source_index": sample.source_index,
                "sample_id": sample.sample_id,
            }
        )
        stats["kept"] += 1

    return source_name, kept_records, dropped_records, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter out non-verifiable questions from math datasets."
    )

    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source dataset spec: <hf_dataset_id>::<split> or local file path.",
    )
    parser.add_argument("--source-default-split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples-per-source", type=int, default=None)
    parser.add_argument(
        "--save-dropped",
        action="store_true",
        help="Also save dropped records with reasons for auditing.",
    )

    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--formatter-model", default="gpt-4o")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_specs = args.source or list(DEFAULT_SOURCES)
    llm_client = maybe_build_llm_client(args)

    all_stats: Dict[str, Any] = {
        "sources": {},
        "total_input": 0,
        "total_kept": 0,
        "total_dropped_question_unverifiable": 0,
        "total_dropped_no_extractable_answer": 0,
        "total_dropped_ambiguous_answer": 0,
        "config": vars(args),
        "timestamp": int(time.time()),
    }

    for spec in source_specs:
        source_name, kept, dropped, stats = filter_one_source(
            source_spec=spec,
            default_split=args.source_default_split,
            llm_client=llm_client,
            formatter_model=args.formatter_model,
            max_samples=args.max_samples_per_source,
        )

        kept_path = output_dir / f"{safe_name(source_name)}_verify_filtered.jsonl"
        write_jsonl(kept_path, kept)

        dropped_path = None
        if args.save_dropped:
            dropped_path = output_dir / f"{safe_name(source_name)}_dropped_unverifiable.jsonl"
            write_jsonl(dropped_path, dropped)

        all_stats["sources"][source_name] = {
            "stats": stats,
            "kept_file": str(kept_path),
            "dropped_file": str(dropped_path) if dropped_path else None,
        }

        all_stats["total_input"] += int(stats.get("input", 0))
        all_stats["total_kept"] += int(stats.get("kept", 0))
        all_stats["total_dropped_question_unverifiable"] += int(
            stats.get("dropped_question_unverifiable", 0)
        )
        all_stats["total_dropped_no_extractable_answer"] += int(
            stats.get("dropped_no_extractable_answer", 0)
        )
        all_stats["total_dropped_ambiguous_answer"] += int(
            stats.get("dropped_ambiguous_answer", 0)
        )

    summary_path = output_dir / "verify_filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
