"""Mean and max pooling of prompt and solution SAE activations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from saerl.dataset.text_stats import compute_text_stats
from saerl.sae_analysis.sae.opensae_adapter import OpenSaeRunner
from saerl.sae_analysis.sae.tokenization import (
    encode_full_and_cut,
    truncate_keep_prompt,
)


def _special_token_mask(tokenizer: object, input_ids: torch.Tensor) -> torch.Tensor:
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if not special_ids:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        try:
            mask |= input_ids == int(sid)
        except Exception:  # noqa: BLE001
            continue
    return mask


def _aggregate_sparse_span(
    feat_indices: torch.Tensor,
    feat_activations: torch.Tensor,
    feature_size: int,
) -> tuple[list[int], list[float], list[float], list[int], list[float], int]:
    """Aggregate sparse token activations into per-feature stats."""
    token_count = int(feat_indices.shape[0])
    if token_count <= 0:
        return [], [], [], [], [], 0

    idx_flat = feat_indices.reshape(-1).to(torch.int64)
    act_flat = feat_activations.reshape(-1).to(torch.float32)
    if idx_flat.numel() == 0:
        return [], [], [], [], [], token_count

    # Aggregate over sparse feature IDs without allocating the full SAE dictionary.
    present, inverse = torch.unique(idx_flat, sorted=True, return_inverse=True)
    bucket_count = int(present.numel())
    if bucket_count == 0:
        return [], [], [], [], [], token_count

    count_vals = torch.bincount(inverse, minlength=bucket_count)
    sum_vals = torch.bincount(inverse, weights=act_flat, minlength=bucket_count)
    max_vals = torch.full((bucket_count,), float("-inf"), device=act_flat.device, dtype=torch.float32)
    max_vals.scatter_reduce_(0, inverse, act_flat, reduce="amax", include_self=True)

    mean_vals = (sum_vals / float(token_count)).cpu().tolist()
    return (
        present.to(torch.int64).cpu().tolist(),
        mean_vals,
        sum_vals.cpu().tolist(),
        count_vals.to(torch.int64).cpu().tolist(),
        max_vals.cpu().tolist(),
        token_count,
    )


def _write_error(path: Path, sample_id: int, exc: Exception) -> None:
    payload = {
        "sample_id": sample_id,
        "error": str(exc),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _process_batch(
    batch: List[Dict[str, object]],
    tokenizer: object,
    runner: OpenSaeRunner,
    pad_id: int,
    max_seq_len: int,
    schema: pa.Schema,
    writer: pq.ParquetWriter,
    errors_path: Path,
) -> None:
    input_ids_list: List[torch.Tensor] = []
    cuts: List[int] = []
    full_lens: List[int] = []
    was_truncated_list: List[bool] = []
    text_stats_list: List[Dict[str, float | int]] = []
    valid_samples: List[Dict[str, object]] = []

    for sample in batch:
        try:
            problem = str(sample.get("problem", ""))
            solution = str(sample.get("solution", ""))
            full_ids, cut = encode_full_and_cut(tokenizer, problem, solution)
            full_ids, cut, was_truncated = truncate_keep_prompt(full_ids, cut, max_seq_len)

            input_ids_list.append(full_ids)
            cuts.append(int(cut))
            full_lens.append(int(full_ids.numel()))
            was_truncated_list.append(was_truncated)
            text_stats_list.append(compute_text_stats(problem, solution))
            valid_samples.append(sample)
        except Exception as exc:  # noqa: BLE001
            _write_error(errors_path, int(sample.get("sample_id", -1)), exc)

    if not input_ids_list:
        return

    max_len = max(t.size(0) for t in input_ids_list)
    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)

    for i, ids in enumerate(input_ids_list):
        seq_len = ids.size(0)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1

    encodings = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    feat_idx, feat_act, input_ids_dev, attention_mask_dev = runner.extract_sparse(encodings)
    feature_size = int(getattr(runner, "feature_size", 0) or 0)
    if feature_size <= 0:
        feature_size = int(feat_idx.max().item()) + 1 if feat_idx.numel() else 0

    special_mask = _special_token_mask(tokenizer, input_ids_dev)
    valid_mask = attention_mask_dev.to(torch.bool) & ~special_mask
    positions = torch.arange(input_ids_dev.shape[1], device=input_ids_dev.device)

    rows = []
    for idx, sample in enumerate(valid_samples):
        try:
            seq_len = int(attention_mask_dev[idx].sum().item())
            cut = cuts[idx]

            prompt_mask = valid_mask[idx] & (positions < int(cut))
            sol_mask = valid_mask[idx] & (positions >= int(cut)) & (positions < int(seq_len))

            (
                prompt_ids,
                prompt_vals,
                prompt_sum_vals,
                prompt_count_vals,
                prompt_max_vals,
                prompt_token_count,
            ) = _aggregate_sparse_span(feat_idx[idx][prompt_mask], feat_act[idx][prompt_mask], feature_size)
            (
                sol_ids,
                sol_vals,
                sol_sum_vals,
                sol_count_vals,
                sol_max_vals,
                sol_token_count,
            ) = _aggregate_sparse_span(feat_idx[idx][sol_mask], feat_act[idx][sol_mask], feature_size)

            text_stats = text_stats_list[idx]
            row = {
                "sample_id": int(sample.get("sample_id", -1)),
                "difficulty": float(sample.get("difficulty", float("nan"))),
                "prompt_len_tok": int(cut),
                "sol_len_tok": int(full_lens[idx] - cut),
                "full_len_tok": int(full_lens[idx]),
                "cut": int(cut),
                "was_truncated": bool(was_truncated_list[idx]),
                "latex_ratio": float(text_stats["latex_ratio"]),
                "digit_ratio": float(text_stats["digit_ratio"]),
                "sym_ratio": float(text_stats["sym_ratio"]),
                "prompt_token_count": int(prompt_token_count),
                "prompt_feat_ids": prompt_ids,
                "prompt_feat_vals": prompt_vals,
                "prompt_feat_sum": prompt_sum_vals,
                "prompt_feat_count": prompt_count_vals,
                "prompt_feat_max": prompt_max_vals,
                "sol_token_count": int(sol_token_count),
                "sol_feat_ids": sol_ids,
                "sol_feat_vals": sol_vals,
                "sol_feat_sum": sol_sum_vals,
                "sol_feat_count": sol_count_vals,
                "sol_feat_max": sol_max_vals,
            }
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            _write_error(errors_path, int(sample.get("sample_id", -1)), exc)

    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)
