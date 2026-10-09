"""Captured layer-output gradients to signed Token-TopK response-mean SAE values.

Average all valid response tokens first, then average rollouts with equal
weight. Absolute values are applied later, after shared feature projection.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def gradient_sae_schema() -> pa.Schema:
    """Sample-level Token-TopK response means and rollout rewards."""
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("prompt_len_tok", pa.int64()),
            ("reward_mean", pa.float32()),
            ("topk_response_feat_ids", pa.list_(pa.int64())),
            ("topk_response_feat_mean", pa.list_(pa.float32())),
            ("zero_signal_token_topk", pa.bool_()),
        ]
    )


@dataclass(frozen=True)
class DecoderSpec:
    checkpoint: Path
    feature_size: int
    hidden_size: int
    hookpoint: str


@dataclass(frozen=True)
class SpanProjection:
    """Signed Token-TopK response mean and the union of token feature IDs."""

    topk_mean: torch.Tensor
    topk_support: torch.Tensor


def _resolve_safetensor(checkpoint: Path, tensor_key: str) -> Path:
    direct = checkpoint / "model.safetensors"
    if direct.is_file():
        return direct
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"No model.safetensors checkpoint under {checkpoint}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    if tensor_key in weight_map:
        return checkpoint / str(weight_map[tensor_key])
    matches = [(key, value) for key, value in weight_map.items() if str(key).endswith(tensor_key)]
    if len(matches) != 1:
        raise KeyError(f"Cannot uniquely resolve {tensor_key!r}; matches={[key for key, _ in matches]}")
    return checkpoint / str(matches[0][1])


def _select_decoder_key(keys: Iterable[str]) -> str:
    keys = list(keys)
    for candidate in ("W_dec", "decoder.weight"):
        if candidate in keys:
            return candidate
    matches = [key for key in keys if key.endswith("W_dec")]
    if len(matches) != 1:
        raise KeyError(f"Expected one OpenSAE W_dec tensor; available keys={keys}")
    return matches[0]


def _load_decoder(checkpoint: Path) -> torch.Tensor:
    if (checkpoint / "model.safetensors").is_file() or (checkpoint / "model.safetensors.index.json").is_file():
        from safetensors import safe_open

        first_file = _resolve_safetensor(checkpoint, "W_dec")
        with safe_open(first_file, framework="pt", device="cpu") as handle:
            decoder_key = _select_decoder_key(handle.keys())
        decoder_file = _resolve_safetensor(checkpoint, decoder_key)
        with safe_open(decoder_file, framework="pt", device="cpu") as handle:
            return handle.get_tensor(decoder_key)

    pt_files = sorted(checkpoint.glob("*.pt"))
    if len(pt_files) != 1:
        raise FileNotFoundError(f"Expected one .pt or safetensors checkpoint under {checkpoint}, got {pt_files}")
    state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=True, mmap=True)
    decoder_key = _select_decoder_key(state_dict.keys())
    return state_dict[decoder_key]


def read_decoder_spec(sae_path: str | Path) -> DecoderSpec:
    checkpoint = Path(sae_path).expanduser().resolve()
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    spec = DecoderSpec(
        checkpoint=checkpoint,
        feature_size=int(config.get("feature_size", 0) or 0),
        hidden_size=int(config.get("hidden_size", 0) or 0),
        hookpoint=str(config.get("input_hookpoint", "")).strip(),
    )
    if spec.feature_size < 1 or spec.hidden_size < 1 or not spec.hookpoint:
        raise ValueError(f"Invalid OpenSAE metadata in {config_path}")
    return spec


class SaeDecoderProjector:
    """Load ``W_dec`` and pool signed Token-TopK response projections."""

    def __init__(
        self,
        sae_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 32,
        gradient_feature_topk: int = 128,
        gradient_zero_eps: float = 1e-12,
        decoder_normalization: str = "runtime_l2_normalized",
    ) -> None:
        if int(chunk_size) < 1:
            raise ValueError("projection_chunk_size must be positive")
        self.spec = read_decoder_spec(sae_path)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        if int(gradient_feature_topk) < 1:
            raise ValueError("gradient_feature_topk must be positive")
        self.gradient_feature_topk = min(int(gradient_feature_topk), self.spec.feature_size)
        self.gradient_zero_eps = float(gradient_zero_eps)
        self.decoder_normalization = str(decoder_normalization)

        decoder = _load_decoder(self.spec.checkpoint)
        assert decoder.ndim == 2, f"SAE decoder must be 2D, got shape {tuple(decoder.shape)}"
        actual_feature_size, actual_hidden_size = map(int, decoder.shape)
        assert self.spec.feature_size == actual_feature_size, (
            f"SAE feature_size mismatch: config={self.spec.feature_size}, decoder={actual_feature_size}"
        )
        assert self.spec.hidden_size == actual_hidden_size, (
            f"SAE hidden_size mismatch: config={self.spec.hidden_size}, decoder={actual_hidden_size}"
        )

        decoder = decoder.float()
        decoder_norms = torch.linalg.vector_norm(decoder, dim=1, keepdim=True)
        self.decoder_row_norm_before = self._norm_stats(decoder_norms)
        self.decoder_was_normalized = bool(
            torch.allclose(decoder_norms, torch.ones_like(decoder_norms), rtol=1e-5, atol=1e-5)
        )
        if self.decoder_normalization == "runtime_l2_normalized":
            self.decoder_normalization_applied = not self.decoder_was_normalized
            if self.decoder_normalization_applied:
                decoder.div_(decoder_norms.clamp_min(1e-8))
        elif self.decoder_normalization == "checkpoint_raw":
            self.decoder_normalization_applied = False
        else:
            raise ValueError(f"Unsupported decoder_normalization={self.decoder_normalization!r}")
        self.decoder_row_norm_after = self._norm_stats(torch.linalg.vector_norm(decoder, dim=1))

        self.decoder = decoder.to(device=self.device, dtype=dtype)
        self.decoder.requires_grad_(False)

    @staticmethod
    def _norm_stats(norms: torch.Tensor) -> dict[str, float]:
        norms = norms.float()
        return {
            "min": float(norms.min().item()),
            "mean": float(norms.mean().item()),
            "max": float(norms.max().item()),
        }

    def _update_topk_pooling_(self, projected, feature_sum, support) -> None:
        # Rank absolute projections, retain their original signed values.
        _, feature_ids = torch.topk(projected.abs(), self.gradient_feature_topk, dim=-1, sorted=False)
        feature_values = projected.gather(dim=-1, index=feature_ids)
        flat_ids, flat_values = feature_ids.reshape(-1), feature_values.reshape(-1)
        valid = flat_values.abs() > self.gradient_zero_eps
        flat_ids, flat_values = flat_ids[valid], flat_values[valid]
        feature_sum.scatter_add_(0, flat_ids, flat_values)
        support[flat_ids] = True

    def project_and_pool(self, token_gradient: torch.Tensor, *, timing_stats=None) -> SpanProjection:
        """Token-TopK, then signed arithmetic mean over all valid response tokens."""
        if token_gradient.ndim != 2 or int(token_gradient.shape[0]) < 1:
            raise ValueError(f"Expected [tokens, hidden] gradient, got {tuple(token_gradient.shape)}")
        if int(token_gradient.shape[1]) != self.spec.hidden_size:
            raise ValueError(
                f"Gradient hidden size {token_gradient.shape[1]} != SAE hidden size {self.spec.hidden_size}"
            )
        gradient = token_gradient.detach()
        feature_sum = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.float32)
        support = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.bool)
        compute_seconds = 0.0
        pending_events = []
        with torch.no_grad():
            for start in range(0, int(gradient.shape[0]), self.chunk_size):
                chunk = gradient[start : start + self.chunk_size].to(self.device).float().to(self.decoder.dtype)
                if self.device.type == "cuda":
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                else:
                    started = time.perf_counter()
                projected = (chunk @ self.decoder.mT).float()
                self._update_topk_pooling_(projected, feature_sum, support)
                if self.device.type == "cuda":
                    end.record()
                    pending_events.append((begin, end))
                else:
                    compute_seconds += time.perf_counter() - started
        if pending_events:
            pending_events[-1][1].synchronize()
            compute_seconds += sum(begin.elapsed_time(end) / 1000.0 for begin, end in pending_events)
        if timing_stats is not None:
            timing_stats["projection_compute_s"] = compute_seconds
        return SpanProjection(feature_sum / float(gradient.shape[0]), support)


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
    """Project actor-captured gradients and persist one row per prompt."""

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
        self.decoder_normalization = str(config.get("decoder_normalization", "runtime_l2_normalized"))
        self.zero_eps = float(config.get("gradient_zero_eps", 1e-12))
        resume_from_step = config.get("resume_from_step", None)
        self.resume_from_step = int(resume_from_step) if resume_from_step is not None else None
        self.log_every_micro_batches = int(config.get("log_every_micro_batches", 1))
        if self.log_every_micro_batches < 0:
            raise ValueError("log_every_micro_batches must be nonnegative")
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
            decoder_normalization=self.decoder_normalization,
        )
        self.gradient_feature_topk = self.projector.gradient_feature_topk
        configured_hookpoint = str(config.get("hookpoint", "") or "").strip()
        self.hookpoint = configured_hookpoint or self.projector.spec.hookpoint
        manifest_path = self.run_dir / "representation_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = self._representation_manifest()
            if any(existing.get(key) != value for key, value in current.items()):
                raise ValueError(
                    "Cannot resume incompatible extraction output; use a new run_name. "
                    "Legacy four-channel shards remain valid selection/export inputs."
                )
        elif self.rank == 0:
            self._write_representation_manifest()
        self._groups: dict[str, dict[str, Any]] = {}
        self._global_step: int | None = None
        self._micro_batches_seen = 0
        self._rollouts_seen = 0
        self._tokens_seen = 0
        self._projection_compute_seconds = 0.0
        self._processing_seconds = 0.0

    def start_update(self, global_step: int) -> None:
        if self._groups:
            raise RuntimeError("Previous static gradient extraction update was not finalized")
        self._global_step = int(global_step)
        self._micro_batches_seen = 0
        self._rollouts_seen = 0
        self._tokens_seen = 0
        self._projection_compute_seconds = 0.0
        self._processing_seconds = 0.0

    def _project_rollout(self, *, token_gradient, prompt_len, response_len) -> SpanProjection:
        started = time.perf_counter()
        timing = {}
        result = self.projector.project_and_pool(
            token_gradient[prompt_len : prompt_len + response_len], timing_stats=timing
        )
        self._projection_compute_seconds += timing["projection_compute_s"]
        self._processing_seconds += time.perf_counter() - started
        return result

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
            sequence_len = prompt_len + response_len
            if int(compact_gradient.shape[0]) != sequence_len:
                raise ValueError(f"Compact gradient has {compact_gradient.shape[0]} tokens, expected {sequence_len}")

            group_id = str(group_ids[row])
            accumulator = self._groups.get(group_id)
            if accumulator is not None and (
                accumulator["sample_id"] != sample_ids[row] or accumulator["prompt_len"] != prompt_len
            ):
                raise RuntimeError(f"Inconsistent metadata for rollout group {group_id}")
            response_projection = self._project_rollout(
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
                    "topk_response_mean_sum": torch.zeros_like(response_projection.topk_mean),
                    "topk_response_support": torch.zeros_like(response_projection.topk_support),
                    "nonzero_advantage_rollouts": 0,
                }
                self._groups[group_id] = accumulator

            accumulator["count"] += 1
            accumulator["reward_sum"].add_(rollout_rewards[row])
            accumulator["topk_response_mean_sum"].add_(response_projection.topk_mean)
            accumulator["topk_response_support"].logical_or_(response_projection.topk_support)
            accumulator["nonzero_advantage_rollouts"] += int(abs(float(advantages[row].item())) > 1e-8)
            self._rollouts_seen += 1
            self._tokens_seen += response_len

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
        return {
            "sample_id": int(item["sample_id"]),
            "prompt_len_tok": int(item["prompt_len"]),
            "reward_mean": float((item["reward_sum"] / count).item()),
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
            "decoder_row_norm_before": self.projector.decoder_row_norm_before,
            "decoder_row_norm_after": self.projector.decoder_row_norm_after,
            "gradient_feature_topk": self.gradient_feature_topk,
            "token_topk_ranking": "absolute_value",
            "token_topk_value": "original_signed_projection",
            "token_topk_missing_value": "zero",
            "gradient_zero_eps": self.zero_eps,
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
        if self._global_step is None:
            raise RuntimeError("Static extraction update was not started")
        finalize_started_at = time.perf_counter()
        records = [self._build_record(group_id, item) for group_id, item in self._groups.items()]
        sample_ids = [int(record["sample_id"]) for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                "Duplicate sample_id values were produced on one rank; "
                "the dataset likely lacks a unique extra_info.index"
            )
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
            # The ordinary key is averaged across DP ranks by reduce_metrics;
            # its duplicate ending in "max" reports the critical-path rank.
            "timing_s/static_gradient_projection_compute": self._projection_compute_seconds,
            "timing_s/static_gradient_projection_compute_max": self._projection_compute_seconds,
            "timing_s/static_gradient_processing": self._processing_seconds,
            "timing_s/static_gradient_processing_max": self._processing_seconds,
            "timing_s/static_gradient_finalize": finalize_seconds,
            "timing_s/static_gradient_finalize_max": finalize_seconds,
            "perf/static_gradient_projected_tokens_per_second": projected_tokens_per_second,
        }
