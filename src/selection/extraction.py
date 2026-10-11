from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from saerl.selection.sae_projection import SaeDecoderProjector, SpanProjection


def gradient_sae_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("prompt_len_tok", pa.int64()),
            ("reward_mean", pa.float32()),
            ("raw_grad_vals", pa.list_(pa.float32())),
            ("topk_response_feat_ids", pa.list_(pa.int64())),
            ("topk_response_feat_mean", pa.list_(pa.float32())),
            ("zero_signal_token_topk", pa.bool_()),
        ]
    )


def _python_values(value: Any, expected: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, (list, tuple)):
        result = list(value)
    elif value is None:
        result = [None] * expected
    else:
        result = [value]
    if not isinstance(result, list):
        result = [result]
    if len(result) != expected:
        raise ValueError(f"Expected {expected} metadata values, got {len(result)}")
    return result


class StaticGradientSaeProcessor:

    def __init__(
        self,
        *,
        config: Any,
        device: torch.device | str,
        use_remove_padding: bool,
    ) -> None:
        self.config = config
        self.use_remove_padding = bool(use_remove_padding)
        self.expected_rollouts = int(config.num_rollouts)
        self.sample_id_key = str(config.get("sample_id_key", "index"))
        self.gradient_feature_topk = int(config.get("gradient_feature_topk", 128))
        self.zero_eps = float(config.get("gradient_zero_eps", 1e-12))
        resume_from_step = config.get("resume_from_step", None)
        self.resume_from_step = int(resume_from_step) if resume_from_step is not None else None
        self.log_every_micro_batches = int(config.get("log_every_micro_batches", 1))
        self.run_dir = Path(str(config.run_dir)).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.projector = SaeDecoderProjector(
            config.sae_path,
            device=device,
            dtype=torch.bfloat16,
            chunk_size=int(config.get("projection_chunk_size", 32)),
            gradient_feature_topk=self.gradient_feature_topk,
            gradient_zero_eps=self.zero_eps,
        )
        self.gradient_feature_topk = self.projector.gradient_feature_topk
        configured_hookpoint = str(config.get("hookpoint", "") or "").strip()
        self.hookpoint = configured_hookpoint or self.projector.spec.hookpoint
        manifest_path = self.run_dir / "representation_manifest.json"
        if self.rank == 0 and not manifest_path.exists():
            self._write_representation_manifest()
        self._groups: dict[str, dict[str, Any]] = {}
        self._global_step: int | None = None
        self._micro_batches_seen = 0
        self._rollouts_seen = 0
        self._tokens_seen = 0
        self._processing_seconds = 0.0

    def start_update(self, global_step: int) -> None:
        self._global_step = int(global_step)
        self._micro_batches_seen = 0
        self._rollouts_seen = 0
        self._tokens_seen = 0
        self._processing_seconds = 0.0

    def _project_rollout(
        self,
        *,
        token_gradient: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> tuple[SpanProjection, SpanProjection]:
        started = time.perf_counter()
        prompt_result = self.projector.project_and_pool(token_gradient[:prompt_len])
        response_result = self.projector.project_and_pool(token_gradient[prompt_len : prompt_len + response_len])
        self._processing_seconds += time.perf_counter() - started
        return prompt_result, response_result

    def _compact_gradients(self, gradient: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
        lengths = [int(value) for value in attention_mask.sum(dim=-1).tolist()]
        if self.use_remove_padding:
            flat = gradient.squeeze(0) if gradient.ndim == 3 and gradient.shape[0] == 1 else gradient
            if flat.ndim != 2 or int(flat.shape[0]) != sum(lengths):
                raise ValueError(
                    f"Remove-padding gradient shape {tuple(gradient.shape)} does not match token lengths {lengths}"
                )
            return list(torch.split(flat, lengths, dim=0))
        if gradient.ndim != 3 or tuple(gradient.shape[:2]) != tuple(attention_mask.shape):
            raise ValueError(
                f"Padded gradient shape {tuple(gradient.shape)} does not match mask {tuple(attention_mask.shape)}"
            )
        return [gradient[row][attention_mask[row].bool()] for row in range(int(gradient.shape[0]))]

    def _sample_ids(self, model_inputs: dict[str, Any], row_count: int) -> list[int]:
        values = _python_values(model_inputs.get(self.sample_id_key), row_count)
        if all(value is None for value in values):
            extra_infos = _python_values(model_inputs.get("extra_info"), row_count)
            values = [info.get(self.sample_id_key) if isinstance(info, dict) else None for info in extra_infos]
        if any(value is None for value in values):
            raise ValueError(
                f"Static extraction requires a numeric {self.sample_id_key!r} per prompt; "
                "add it to the dataset row or extra_info"
            )
        return [int(value) for value in values]

    def consume_micro_batch(
        self,
        gradient: torch.Tensor,
        model_inputs: dict[str, Any],
        gradient_scale: float = 1.0,
    ) -> None:
        if float(gradient_scale) != 1.0:
            gradient = gradient / float(gradient_scale)

        response_mask = model_inputs["response_mask"]
        attention_mask = model_inputs["attention_mask"]
        row_count = int(response_mask.shape[0])
        compact_gradients = self._compact_gradients(gradient, attention_mask)
        sample_ids = self._sample_ids(model_inputs, row_count)
        group_ids = _python_values(model_inputs.get("uid"), row_count)
        padding_flags = _python_values(model_inputs.get("static_gradient_is_padding"), row_count)
        if all(value is None for value in padding_flags):
            padding_flags = [False] * row_count
        if any(value is None for value in group_ids):
            raise ValueError("Static extraction requires the trainer-provided uid metadata")

        prompt_width = int(model_inputs["input_ids"].shape[1] - model_inputs["responses"].shape[1])
        prompt_lengths = attention_mask[:, :prompt_width].sum(dim=-1)
        response_lengths = response_mask.sum(dim=-1)
        valid_counts = response_lengths.clamp_min(1)
        advantages = ((model_inputs["advantages"] * response_mask).sum(dim=-1) / valid_counts).float()
        reward_values = model_inputs.get("token_level_scores", model_inputs.get("token_level_rewards"))
        rollout_rewards = (reward_values * response_mask).sum(dim=-1).float()

        for row, compact_gradient in enumerate(compact_gradients):
            if bool(padding_flags[row]):
                continue
            prompt_len = int(prompt_lengths[row].item())
            response_len = int(response_lengths[row].item())

            group_id = str(group_ids[row])
            accumulator = self._groups.get(group_id)
            prompt_projection, response_projection = self._project_rollout(
                token_gradient=compact_gradient,
                prompt_len=prompt_len,
                response_len=response_len,
            )
            if accumulator is None:
                accumulator = {
                    "sample_id": sample_ids[row],
                    "prompt_len": prompt_len,
                    "count": 0,
                    "reward_sum": torch.zeros((), device=rollout_rewards.device, dtype=torch.float32),
                    "raw_prompt_mean_sum": torch.zeros_like(prompt_projection.raw_mean),
                    "raw_prompt_peak_sum": torch.zeros_like(prompt_projection.raw_signed_peak),
                    "raw_response_mean_sum": torch.zeros_like(response_projection.raw_mean),
                    "raw_response_peak_sum": torch.zeros_like(response_projection.raw_signed_peak),
                    "topk_response_mean_sum": torch.zeros_like(response_projection.topk_mean),
                    "topk_response_support": torch.zeros_like(response_projection.topk_support),
                    "nonzero_advantage_rollouts": 0,
                }
                self._groups[group_id] = accumulator

            accumulator["count"] += 1
            accumulator["reward_sum"].add_(rollout_rewards[row])
            accumulator["raw_prompt_mean_sum"].add_(prompt_projection.raw_mean)
            accumulator["raw_prompt_peak_sum"].add_(prompt_projection.raw_signed_peak)
            accumulator["raw_response_mean_sum"].add_(response_projection.raw_mean)
            accumulator["raw_response_peak_sum"].add_(response_projection.raw_signed_peak)
            accumulator["topk_response_mean_sum"].add_(response_projection.topk_mean)
            accumulator["topk_response_support"].logical_or_(response_projection.topk_support)
            accumulator["nonzero_advantage_rollouts"] += int(abs(float(advantages[row].item())) > 1e-8)
            self._rollouts_seen += 1
            self._tokens_seen += prompt_len + response_len

        self._micro_batches_seen += 1
        if (
            self.rank == 0
            and self.log_every_micro_batches > 0
            and self._micro_batches_seen % self.log_every_micro_batches == 0
        ):
            print(
                "Static gradient-SAE progress: "
                f"step={self._global_step} micro_batches={self._micro_batches_seen} "
                f"rollouts={self._rollouts_seen} projected_tokens={self._tokens_seen}",
                flush=True,
            )

    def _build_record(self, group_id: str, item: dict[str, Any]) -> dict[str, Any]:
        count = int(item["count"])
        if count != self.expected_rollouts:
            raise RuntimeError(
                f"Prompt group {group_id!r} has {count} rollouts on rank {self.rank}, "
                f"expected {self.expected_rollouts}; keep all prompt rollouts on one DP rank"
            )
        mean = (item["topk_response_mean_sum"] / count).float()
        ids = item["topk_response_support"].nonzero().flatten()
        raw_gradient_representation = torch.cat([
            item["raw_prompt_mean_sum"] / count,
            item["raw_prompt_peak_sum"] / count,
            item["raw_response_mean_sum"] / count,
            item["raw_response_peak_sum"] / count,
        ])
        return {
            "sample_id": int(item["sample_id"]),
            "prompt_len_tok": int(item["prompt_len"]),
            "reward_mean": float((item["reward_sum"] / count).item()),
            "raw_grad_vals": raw_gradient_representation.cpu().tolist(),
            "topk_response_feat_ids": ids.cpu().tolist(),
            "topk_response_feat_mean": mean[ids].cpu().tolist(),
            "zero_signal_token_topk": bool(
                item["nonzero_advantage_rollouts"] == 0 or torch.linalg.vector_norm(mean).item() < self.zero_eps
            ),
        }

    def _representation_manifest(self) -> dict[str, Any]:
        return {
            "sae_checkpoint": str(self.projector.spec.checkpoint),
            "sae_feature_size": self.projector.spec.feature_size,
            "hookpoint": self.hookpoint,
            "gradient_token_scope": "response",
            "num_rollouts": self.expected_rollouts,
            "decoder_normalization": self.projector.decoder_normalization,
            "decoder_normalization_applied": self.projector.decoder_normalization_applied,
            "decoder_was_normalized": self.projector.decoder_was_normalized,
            "gradient_feature_topk": self.gradient_feature_topk,
            "token_topk_ranking": "absolute_value",
            "token_topk_value": "original_signed_projection",
            "token_topk_missing_value": "zero",
            "gradient_zero_eps": self.zero_eps,
            "raw_gradient_token_scope": "attention",
            "raw_gradient_token_pool": "mean_and_argmax_abs_keep_sign_per_span",
            "raw_gradient_rollout_pool": "equal_weight_arithmetic_mean_per_channel",
            "raw_gradient_representation_layout": [
                "prompt_mean", "prompt_signed_peak", "response_mean", "response_signed_peak",
            ],
            "raw_gradient_representation_size": 4 * self.projector.spec.hidden_size,
            "raw_gradient_storage_format": "parquet_list_float32_4h",
            "token_mean_denominator": "all_valid_response_tokens",
            "rollout_pool": "equal_weight_arithmetic_mean_of_signed_response_means",
            "feature_union": "response_token_union_then_rollout_union",
            "representation_layout": ["response_mean"],
            "topk_representation_size": self.projector.spec.feature_size,
            "physical_storage_format": "parquet_sparse_response_ids_and_float32_means",
        }

    def _write_representation_manifest(self) -> None:
        manifest_path = self.run_dir / "representation_manifest.json"
        staging_path = manifest_path.with_suffix(manifest_path.suffix + ".inprogress")
        staging_path.write_text(
            json.dumps(self._representation_manifest(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_path, manifest_path)

    def finish_update(self) -> dict[str, float]:
        finalize_started_at = time.perf_counter()
        records = [self._build_record(group_id, item) for group_id, item in self._groups.items()]
        final_path = self.run_dir / f"step-{self._global_step:08d}-rank-{self.rank:05d}.parquet"
        staging_path = final_path.with_suffix(final_path.suffix + ".inprogress")
        rows = len(records)
        if final_path.exists():
            if self.resume_from_step != self._global_step:
                raise FileExistsError(f"Refusing to overwrite static gradient output: {final_path}")
            print(
                f"Static gradient-SAE rank={self.rank} step={self._global_step} kept_existing={final_path}",
                flush=True,
            )
        else:
            table = pa.Table.from_pylist(records, schema=gradient_sae_schema())
            pq.write_table(table, staging_path, compression="snappy", use_dictionary=True)
            os.replace(staging_path, final_path)
            print(
                f"Static gradient-SAE rank={self.rank} step={self._global_step} wrote_rows={rows} path={final_path}",
                flush=True,
            )
        finalize_seconds = time.perf_counter() - finalize_started_at
        projected_tokens_per_second = self._tokens_seen / max(self._processing_seconds, 1e-12)
        self._groups = {}
        self._global_step = None
        return {
            "static_gradient/rows": float(rows),
            "static_gradient/optimizer_step": 0.0,
            "timing_s/static_gradient_processing": self._processing_seconds,
            "timing_s/static_gradient_processing_max": self._processing_seconds,
            "timing_s/static_gradient_finalize": finalize_seconds,
            "timing_s/static_gradient_finalize_max": finalize_seconds,
            "perf/static_gradient_projected_tokens_per_second": projected_tokens_per_second,
        }


def main() -> None:
    from saerl.verl.train import launch

    launch("extract")


if __name__ == "__main__":
    main()
