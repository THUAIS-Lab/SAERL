"""Static layer-gradient processing and OpenSAE decoder representations.

The actor owns the backward hook and passes the captured tensor here. Each
rollout is projected once and pooled into dense and token-TopK prompt/response
branches before equal-weight rollout averaging.
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
    """Sample-level raw-gradient, dense SAE, and sparse token-TopK schema."""
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("prompt_len_tok", pa.int64()),
            ("reward_mean", pa.float32()),
            ("raw_grad_vals", pa.list_(pa.float32())),
            ("dense_grad_feat_vals", pa.list_(pa.float32())),
            ("topk_prompt_feat_ids", pa.list_(pa.int64())),
            ("topk_prompt_feat_mean", pa.list_(pa.float32())),
            ("topk_prompt_feat_signed_peak", pa.list_(pa.float32())),
            ("topk_response_feat_ids", pa.list_(pa.int64())),
            ("topk_response_feat_mean", pa.list_(pa.float32())),
            ("topk_response_feat_signed_peak", pa.list_(pa.float32())),
            ("zero_signal_dense", pa.bool_()),
            ("zero_signal_token_topk", pa.bool_()),
        ]
    )


def materialize_topk_4f(row: dict[str, Any], feature_size: int) -> torch.Tensor:
    """Restore one sparse token-TopK row to its logical four-channel vector."""
    out = torch.zeros(4 * int(feature_size), dtype=torch.float32)
    prompt_ids = torch.as_tensor(row["topk_prompt_feat_ids"], dtype=torch.long)
    response_ids = torch.as_tensor(row["topk_response_feat_ids"], dtype=torch.long)
    out[prompt_ids] = torch.as_tensor(row["topk_prompt_feat_mean"], dtype=torch.float32)
    out[feature_size + prompt_ids] = torch.as_tensor(row["topk_prompt_feat_signed_peak"], dtype=torch.float32)
    out[2 * feature_size + response_ids] = torch.as_tensor(row["topk_response_feat_mean"], dtype=torch.float32)
    out[3 * feature_size + response_ids] = torch.as_tensor(row["topk_response_feat_signed_peak"], dtype=torch.float32)
    return out


def materialize_raw_gradient_4h(row: dict[str, Any], hidden_size: int) -> torch.Tensor:
    """Restore and validate one dense raw-gradient baseline row."""
    out = torch.as_tensor(row["raw_grad_vals"], dtype=torch.float32)
    expected_size = 4 * int(hidden_size)
    if out.numel() != expected_size:
        raise ValueError(f"Expected raw gradient representation size {expected_size}, got {out.numel()}")
    return out


@dataclass(frozen=True)
class DecoderSpec:
    checkpoint: Path
    feature_size: int
    hidden_size: int
    hookpoint: str


@dataclass(frozen=True)
class SpanProjection:
    """Per-rollout token pooling results for one prompt or response span."""

    raw_mean: torch.Tensor
    raw_signed_peak: torch.Tensor
    dense_mean: torch.Tensor
    dense_signed_peak: torch.Tensor
    topk_mean: torch.Tensor
    topk_signed_peak: torch.Tensor
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
    """Load ``W_dec`` and pool dense plus token-TopK signed projections."""

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

    @staticmethod
    def _merge_signed_peak_(current: torch.Tensor, candidate: torch.Tensor) -> None:
        replace = candidate.abs() > current.abs()
        replace |= (candidate.abs() == current.abs()) & (candidate > current)
        current.copy_(torch.where(replace, candidate, current))

    @staticmethod
    def _dense_chunk_signed_peak(projected: torch.Tensor) -> torch.Tensor:
        peak_abs = projected.abs().amax(dim=0)
        has_positive_tie = (projected == peak_abs.unsqueeze(0)).any(dim=0)
        return torch.where(has_positive_tie, peak_abs, -peak_abs)

    def _update_topk_pooling_(
        self,
        projected: torch.Tensor,
        feature_sum: torch.Tensor,
        signed_peak: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        # Reuse the same OpenSAE TopK module used by activation extraction,
        # ranking abs projections while gathering values from the signed input.
        _, feature_ids = torch.topk(projected.abs(), self.gradient_feature_topk, dim=-1, sorted=False)
        feature_values = projected.gather(dim=-1, index=feature_ids)
        flat_ids = feature_ids.reshape(-1)
        flat_values = feature_values.reshape(-1)
        valid = flat_values.abs() > self.gradient_zero_eps
        flat_ids = flat_ids[valid]
        flat_values = flat_values[valid]
        if flat_ids.numel() == 0:
            return

        feature_sum.scatter_add_(0, flat_ids, flat_values)
        support[flat_ids] = True

        chunk_peak_abs = torch.zeros_like(signed_peak)
        chunk_peak_abs.scatter_reduce_(
            0,
            flat_ids,
            flat_values.abs(),
            reduce="amax",
            include_self=True,
        )
        at_chunk_peak = flat_values.abs() == chunk_peak_abs[flat_ids]
        chunk_peak = torch.full_like(signed_peak, float("-inf"))
        chunk_peak.scatter_reduce_(
            0,
            flat_ids[at_chunk_peak],
            flat_values[at_chunk_peak],
            reduce="amax",
            include_self=True,
        )
        touched = chunk_peak_abs > self.gradient_zero_eps
        candidate = torch.where(touched, chunk_peak, signed_peak)
        self._merge_signed_peak_(signed_peak, candidate)

    def project_and_pool(
        self,
        token_gradient: torch.Tensor,
        *,
        timing_stats: dict[str, float] | None = None,
    ) -> SpanProjection:
        """Project a span and update both representation versions per chunk."""
        if token_gradient.ndim != 2 or int(token_gradient.shape[0]) < 1:
            raise ValueError(f"Expected [tokens, hidden] gradient, got {tuple(token_gradient.shape)}")
        if int(token_gradient.shape[1]) != self.spec.hidden_size:
            raise ValueError(
                f"Gradient hidden size {token_gradient.shape[1]} != SAE hidden size {self.spec.hidden_size}"
            )
        gradient = token_gradient.detach()

        raw_sum = torch.zeros(self.spec.hidden_size, device=self.device, dtype=torch.float32)
        raw_signed_peak = torch.zeros_like(raw_sum)
        dense_sum = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.float32)
        dense_signed_peak = torch.zeros_like(dense_sum)
        topk_sum = torch.zeros_like(dense_sum)
        topk_signed_peak = torch.zeros_like(dense_sum)
        topk_support = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.bool)
        projection_compute_seconds = 0.0
        pending_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        with torch.no_grad():
            for start in range(0, int(gradient.shape[0]), self.chunk_size):
                raw_chunk = gradient[start : start + self.chunk_size].to(self.device).float()
                chunk = raw_chunk.to(dtype=self.decoder.dtype)
                if self.device.type == "cuda":
                    compute_start = torch.cuda.Event(enable_timing=True)
                    compute_end = torch.cuda.Event(enable_timing=True)
                    compute_start.record()
                else:
                    compute_started_at = time.perf_counter()
                raw_sum.add_(raw_chunk.sum(dim=0))
                self._merge_signed_peak_(raw_signed_peak, self._dense_chunk_signed_peak(raw_chunk))
                projected = (chunk @ self.decoder.mT).float()
                dense_sum.add_(projected.sum(dim=0))
                self._merge_signed_peak_(dense_signed_peak, self._dense_chunk_signed_peak(projected))
                self._update_topk_pooling_(projected, topk_sum, topk_signed_peak, topk_support)
                if self.device.type == "cuda":
                    compute_end.record()
                    pending_cuda_events.append((compute_start, compute_end))
                else:
                    projection_compute_seconds += time.perf_counter() - compute_started_at
        if pending_cuda_events:
            pending_cuda_events[-1][1].synchronize()
            projection_compute_seconds += sum(
                start_event.elapsed_time(end_event) / 1000.0 for start_event, end_event in pending_cuda_events
            )
        if timing_stats is not None:
            timing_stats["projection_compute_s"] = projection_compute_seconds
        token_count = float(gradient.shape[0])
        return SpanProjection(
            raw_mean=raw_sum / token_count,
            raw_signed_peak=raw_signed_peak,
            dense_mean=dense_sum / token_count,
            dense_signed_peak=dense_signed_peak,
            topk_mean=topk_sum / token_count,
            topk_signed_peak=topk_signed_peak,
            topk_support=topk_support,
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
        self.token_scope = str(config.get("gradient_token_scope", "attention"))
        if self.token_scope != "attention":
            raise ValueError("prompt/response representations require gradient_token_scope='attention'")
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
        if self.rank == 0 and not (self.run_dir / "representation_manifest.json").exists():
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

    def _project_rollout(
        self,
        *,
        token_gradient: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> tuple[SpanProjection, SpanProjection]:
        processing_started_at = time.perf_counter()
        prompt_timing: dict[str, float] = {}
        prompt_result = self.projector.project_and_pool(
            token_gradient[:prompt_len],
            timing_stats=prompt_timing,
        )
        response_timing: dict[str, float] = {}
        response_result = self.projector.project_and_pool(
            token_gradient[prompt_len : prompt_len + response_len],
            timing_stats=response_timing,
        )
        self._projection_compute_seconds += (
            prompt_timing["projection_compute_s"] + response_timing["projection_compute_s"]
        )
        self._processing_seconds += time.perf_counter() - processing_started_at
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
            sequence_len = prompt_len + response_len
            if int(compact_gradient.shape[0]) != sequence_len:
                raise ValueError(f"Compact gradient has {compact_gradient.shape[0]} tokens, expected {sequence_len}")

            group_id = str(group_ids[row])
            accumulator = self._groups.get(group_id)
            if accumulator is not None and (
                accumulator["sample_id"] != sample_ids[row] or accumulator["prompt_len"] != prompt_len
            ):
                raise RuntimeError(f"Inconsistent metadata for rollout group {group_id}")
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
                    "dense_prompt_mean_sum": torch.zeros_like(prompt_projection.dense_mean),
                    "dense_prompt_peak_sum": torch.zeros_like(prompt_projection.dense_signed_peak),
                    "dense_response_mean_sum": torch.zeros_like(response_projection.dense_mean),
                    "dense_response_peak_sum": torch.zeros_like(response_projection.dense_signed_peak),
                    "topk_prompt_mean_sum": torch.zeros_like(prompt_projection.topk_mean),
                    "topk_prompt_peak_sum": torch.zeros_like(prompt_projection.topk_signed_peak),
                    "topk_response_mean_sum": torch.zeros_like(response_projection.topk_mean),
                    "topk_response_peak_sum": torch.zeros_like(response_projection.topk_signed_peak),
                    "topk_prompt_support": torch.zeros_like(prompt_projection.topk_support),
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
            accumulator["dense_prompt_mean_sum"].add_(prompt_projection.dense_mean)
            accumulator["dense_prompt_peak_sum"].add_(prompt_projection.dense_signed_peak)
            accumulator["dense_response_mean_sum"].add_(response_projection.dense_mean)
            accumulator["dense_response_peak_sum"].add_(response_projection.dense_signed_peak)
            accumulator["topk_prompt_mean_sum"].add_(prompt_projection.topk_mean)
            accumulator["topk_prompt_peak_sum"].add_(prompt_projection.topk_signed_peak)
            accumulator["topk_response_mean_sum"].add_(response_projection.topk_mean)
            accumulator["topk_response_peak_sum"].add_(response_projection.topk_signed_peak)
            accumulator["topk_prompt_support"].logical_or_(prompt_projection.topk_support)
            accumulator["topk_response_support"].logical_or_(response_projection.topk_support)
            # with open('topk_support.txt', 'a') as f:
            #     f.write(f"Group ID: {group_id}\n")
            #     f.write(f"num of topk prompt features: {prompt_projection.topk_support.sum().item()}\n")
            #     f.write(f"num of topk response features: {response_projection.topk_support.sum().item()}\n")
            #     f.write(f"TopK Prompt Support: {prompt_projection.topk_support.cpu().numpy()}\n")
            #     f.write(f"TopK Response Support: {response_projection.topk_support.cpu().numpy()}\n\n")
            accumulator["nonzero_advantage_rollouts"] += int(abs(float(advantages[row].item())) > 1e-8)
            self._rollouts_seen += 1
            self._tokens_seen += sequence_len

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
                f"expected {self.expected_rollouts}; disable batch balancing and use prompt batches divisible by DP"
            )
        raw_gradient_channels = [
            item["raw_prompt_mean_sum"] / count,
            item["raw_prompt_peak_sum"] / count,
            item["raw_response_mean_sum"] / count,
            item["raw_response_peak_sum"] / count,
        ]
        dense_channels = [
            item["dense_prompt_mean_sum"] / count,
            item["dense_prompt_peak_sum"] / count,
            item["dense_response_mean_sum"] / count,
            item["dense_response_peak_sum"] / count,
        ]
        topk_channels = [
            item["topk_prompt_mean_sum"] / count,
            item["topk_prompt_peak_sum"] / count,
            item["topk_response_mean_sum"] / count,
            item["topk_response_peak_sum"] / count,
        ]
        raw_gradient_representation = torch.cat(raw_gradient_channels).float()
        dense_representation = torch.cat(dense_channels).float()
        topk_representation = torch.cat(topk_channels).float()
        prompt_ids = item["topk_prompt_support"].nonzero().flatten()
        response_ids = item["topk_response_support"].nonzero().flatten()
        nonzero_advantage_rollouts = int(item["nonzero_advantage_rollouts"])
        return {
            "sample_id": int(item["sample_id"]),
            "prompt_len_tok": int(item["prompt_len"]),
            "reward_mean": float((item["reward_sum"] / count).item()),
            "raw_grad_vals": raw_gradient_representation.cpu().tolist(),
            "dense_grad_feat_vals": dense_representation.cpu().tolist(),
            "topk_prompt_feat_ids": prompt_ids.cpu().tolist(),
            "topk_prompt_feat_mean": topk_channels[0][prompt_ids].float().cpu().tolist(),
            "topk_prompt_feat_signed_peak": topk_channels[1][prompt_ids].float().cpu().tolist(),
            "topk_response_feat_ids": response_ids.cpu().tolist(),
            "topk_response_feat_mean": topk_channels[2][response_ids].float().cpu().tolist(),
            "topk_response_feat_signed_peak": topk_channels[3][response_ids].float().cpu().tolist(),
            "zero_signal_dense": bool(
                nonzero_advantage_rollouts == 0 or torch.linalg.vector_norm(dense_representation).item() < self.zero_eps
            ),
            "zero_signal_token_topk": bool(
                nonzero_advantage_rollouts == 0 or torch.linalg.vector_norm(topk_representation).item() < self.zero_eps
            ),
        }

    def _representation_manifest(self) -> dict[str, Any]:
        feature_size = self.projector.spec.feature_size
        hidden_size = self.projector.spec.hidden_size
        return {
            "sae_checkpoint": str(self.projector.spec.checkpoint),
            "sae_feature_size": feature_size,
            "hookpoint": self.hookpoint,
            "gradient_token_scope": self.token_scope,
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
            "raw_gradient_token_pool": "mean_and_argmax_abs_keep_sign_per_span",
            "raw_gradient_rollout_pool": "equal_weight_arithmetic_mean_per_channel",
            "raw_gradient_representation_layout": [
                "prompt_mean",
                "prompt_signed_peak",
                "response_mean",
                "response_signed_peak",
            ],
            "raw_gradient_representation_size": 4 * hidden_size,
            "token_mean_denominator": "all_valid_tokens_in_span",
            "token_peak": "argmax_abs_keep_sign",
            "signed_peak_tie_break": "positive",
            "prompt_response_span_definition": "compact_attention=[valid_prompt||valid_response]",
            "rollout_pool": "equal_weight_arithmetic_mean_per_branch",
            "feature_union": "token_union_then_rollout_union_per_span",
            "representation_layout": [
                "prompt_mean",
                "prompt_signed_peak",
                "response_mean",
                "response_signed_peak",
            ],
            "dense_representation_size": 4 * feature_size,
            "topk_representation_size": 4 * feature_size,
            "physical_storage_format": {
                "raw_gradient": "parquet_list_float32_4h",
                "dense": "parquet_list_float32_4f",
                "token_topk": "parquet_sparse_union_ids_and_float32_values",
            },
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
