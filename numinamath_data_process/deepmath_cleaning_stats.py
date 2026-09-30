#!/usr/bin/env python3
"""
Build stage-level and source-level retention statistics for deepmath cleaning outputs.

Expected files under --output-dir:
- stage0_loaded.jsonl (optional but recommended)
- stage1_basic_filtered.jsonl (optional)
- stage2_decontaminated.jsonl (optional)
- stage3_difficulty_filtered.jsonl (optional)
- deepmath_curated.jsonl (optional)
- summary.json (optional)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


STAGE_FILES = [
    ("stage0", "stage0_loaded.jsonl"),
    ("stage1", "stage1_basic_filtered.jsonl"),
    ("stage2", "stage2_decontaminated.jsonl"),
    ("stage3", "stage3_difficulty_filtered.jsonl"),
    ("final", "deepmath_curated.jsonl"),
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def build_source_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_source_name: Dict[str, int] = {}
    by_source_schema: Dict[str, int] = {}
    by_source_value: Dict[str, Dict[str, int]] = {}

    for r in records:
        source_name = to_str(r.get("source_name", "unknown"))
        source_schema = to_str(r.get("source_schema", ""))
        source_value = to_str(r.get("source", ""))

        by_source_name[source_name] = by_source_name.get(source_name, 0) + 1

        if source_schema:
            by_source_schema[source_schema] = by_source_schema.get(source_schema, 0) + 1

        if source_name not in by_source_value:
            by_source_value[source_name] = {}
        by_source_value[source_name][source_value] = by_source_value[source_name].get(source_value, 0) + 1

    return {
        "total": len(records),
        "by_source_name": dict(sorted(by_source_name.items(), key=lambda x: x[0])),
        "by_source_schema": dict(sorted(by_source_schema.items(), key=lambda x: x[0])),
        "by_source_value": {
            k: dict(sorted(v.items(), key=lambda x: x[0]))
            for k, v in sorted(by_source_value.items(), key=lambda x: x[0])
        },
    }


def flatten_source_value_counts(stage_stats: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    by_source_value = stage_stats.get("by_source_value", {})
    if not isinstance(by_source_value, dict):
        return out

    for source_name, value_map in by_source_value.items():
        if not isinstance(value_map, dict):
            continue
        for source_value, count in value_map.items():
            key = f"{source_name}::{source_value}"
            out[key] = int(count)
    return out


def build_retention_table(stage_stats_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_order = ["stage0", "stage1", "stage2", "stage3", "final"]
    stage_counts = {
        s: flatten_source_value_counts(stage_stats_map.get(s, {"by_source_value": {}}))
        for s in stage_order
    }

    all_keys = set()
    for counts in stage_counts.values():
        all_keys.update(counts.keys())

    rows: List[Dict[str, Any]] = []
    for key in sorted(all_keys):
        c0 = stage_counts["stage0"].get(key, 0)
        c1 = stage_counts["stage1"].get(key, 0)
        c2 = stage_counts["stage2"].get(key, 0)
        c3 = stage_counts["stage3"].get(key, 0)
        cf = stage_counts["final"].get(key, 0)

        rows.append(
            {
                "source_value_key": key,
                "stage0": c0,
                "stage1": c1,
                "stage2": c2,
                "stage3": c3,
                "final": cf,
                "ret_stage1_vs_stage0": round(safe_div(c1, c0), 6),
                "ret_stage2_vs_stage0": round(safe_div(c2, c0), 6),
                "ret_stage3_vs_stage0": round(safe_div(c3, c0), 6),
                "ret_final_vs_stage0": round(safe_div(cf, c0), 6),
                "ret_stage2_vs_stage1": round(safe_div(c2, c1), 6),
                "ret_stage3_vs_stage2": round(safe_div(c3, c2), 6),
                "ret_final_vs_stage3": round(safe_div(cf, c3), 6),
            }
        )
    return rows


def stage_totals(stage_stats_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for stage, stats in stage_stats_map.items():
        out[stage] = int(stats.get("total", 0))
    return out


def stage_pass_rates(totals: Dict[str, int]) -> Dict[str, float]:
    s0 = totals.get("stage0", 0)
    s1 = totals.get("stage1", 0)
    s2 = totals.get("stage2", 0)
    s3 = totals.get("stage3", 0)
    sf = totals.get("final", 0)

    return {
        "stage1_vs_stage0": round(safe_div(s1, s0), 6),
        "stage2_vs_stage1": round(safe_div(s2, s1), 6),
        "stage3_vs_stage2": round(safe_div(s3, s2), 6),
        "final_vs_stage3": round(safe_div(sf, s3), 6),
        "final_vs_stage0": round(safe_div(sf, s0), 6),
    }


def build_markdown_report(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DeepMath Cleaning Stats")
    lines.append("")

    totals = result.get("stage_totals", {})
    rates = result.get("stage_pass_rates", {})

    lines.append("## Stage Totals")
    lines.append("")
    lines.append("| Stage | Count |")
    lines.append("| --- | ---: |")
    for stage in ["stage0", "stage1", "stage2", "stage3", "final"]:
        lines.append(f"| {stage} | {int(totals.get(stage, 0))} |")

    lines.append("")
    lines.append("## Stage Pass Rates")
    lines.append("")
    lines.append("| Transition | Rate |")
    lines.append("| --- | ---: |")
    for key in [
        "stage1_vs_stage0",
        "stage2_vs_stage1",
        "stage3_vs_stage2",
        "final_vs_stage3",
        "final_vs_stage0",
    ]:
        lines.append(f"| {key} | {float(rates.get(key, 0.0)):.4f} |")

    retention_rows = result.get("source_value_retention", [])
    top_rows = sorted(retention_rows, key=lambda x: x.get("stage0", 0), reverse=True)[:50]

    lines.append("")
    lines.append("## Source Value Retention (Top 50 by stage0 count)")
    lines.append("")
    lines.append("| Source::Value | stage0 | stage1 | stage2 | stage3 | final | final/stage0 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in top_rows:
        lines.append(
            "| {k} | {s0} | {s1} | {s2} | {s3} | {sf} | {rf:.4f} |".format(
                k=r.get("source_value_key", ""),
                s0=int(r.get("stage0", 0)),
                s1=int(r.get("stage1", 0)),
                s2=int(r.get("stage2", 0)),
                s3=int(r.get("stage3", 0)),
                sf=int(r.get("final", 0)),
                rf=float(r.get("ret_final_vs_stage0", 0.0)),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute source and stage retention stats")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--markdown-path", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"

    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    stage_records: Dict[str, List[Dict[str, Any]]] = {}
    stage_stats_map: Dict[str, Dict[str, Any]] = {}

    for stage, fname in STAGE_FILES:
        rows = read_jsonl(output_dir / fname)
        stage_records[stage] = rows
        stage_stats_map[stage] = build_source_stats(rows)

    totals = stage_totals(stage_stats_map)
    rates = stage_pass_rates(totals)

    result: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "stage_totals": totals,
        "stage_pass_rates": rates,
        "source_stats_by_stage": stage_stats_map,
        "source_value_retention": build_retention_table(stage_stats_map),
        "summary_from_pipeline": {
            "source_filter": summary.get("source_filter", {}),
            "stage1": summary.get("stage1", {}),
            "stage2": summary.get("stage2", {}),
            "stage3": summary.get("stage3", {}),
            "stage4": summary.get("stage4", {}),
        },
    }

    json_path = Path(args.json_path) if args.json_path else output_dir / "source_retention_stats.json"
    md_path = Path(args.markdown_path) if args.markdown_path else output_dir / "source_retention_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md = build_markdown_report(result)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote JSON stats: {json_path}")
    print(f"Wrote Markdown report: {md_path}")


if __name__ == "__main__":
    main()
