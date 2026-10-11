from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class DecoderSpec:
    checkpoint: Path
    feature_size: int
    hidden_size: int
    hookpoint: str


@dataclass(frozen=True)
class SpanProjection:

    raw_mean: torch.Tensor
    raw_signed_peak: torch.Tensor
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

        direct = checkpoint / "model.safetensors"
        if direct.is_file():
            with safe_open(direct, framework="pt", device="cpu") as handle:
                decoder_key = _select_decoder_key(handle.keys())
        else:
            index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
            decoder_key = _select_decoder_key(index["weight_map"])
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

    def __init__(
        self,
        sae_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 32,
        gradient_feature_topk: int = 128,
        gradient_zero_eps: float = 1e-12,
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
        self.decoder_normalization = "runtime_l2_normalized"

        decoder = _load_decoder(self.spec.checkpoint)
        if tuple(decoder.shape) != (self.spec.feature_size, self.spec.hidden_size):
            raise ValueError("SAE decoder shape does not match feature_size and hidden_size")

        decoder = decoder.float()
        decoder_norms = torch.linalg.vector_norm(decoder, dim=1, keepdim=True)
        self.decoder_was_normalized = bool(
            torch.allclose(decoder_norms, torch.ones_like(decoder_norms), rtol=1e-5, atol=1e-5)
        )
        self.decoder_normalization_applied = not self.decoder_was_normalized
        if self.decoder_normalization_applied:
            decoder.div_(decoder_norms.clamp_min(1e-8))

        self.decoder = decoder.to(device=self.device, dtype=dtype)
        self.decoder.requires_grad_(False)

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

    def _update_topk_pooling_(self, projected, feature_sum, support) -> None:
        _, feature_ids = torch.topk(projected.abs(), self.gradient_feature_topk, dim=-1, sorted=False)
        feature_values = projected.gather(dim=-1, index=feature_ids)
        flat_ids, flat_values = feature_ids.reshape(-1), feature_values.reshape(-1)
        valid = flat_values.abs() > self.gradient_zero_eps
        flat_ids, flat_values = flat_ids[valid], flat_values[valid]
        feature_sum.scatter_add_(0, flat_ids, flat_values)
        support[flat_ids] = True

    def project_and_pool(self, token_gradient: torch.Tensor) -> SpanProjection:
        if token_gradient.ndim != 2:
            raise ValueError(f"Expected [tokens, hidden] gradient, got {tuple(token_gradient.shape)}")
        if int(token_gradient.shape[1]) != self.spec.hidden_size:
            raise ValueError(
                f"Gradient hidden size {token_gradient.shape[1]} != SAE hidden size {self.spec.hidden_size}"
            )
        gradient = token_gradient.detach()
        raw_sum = torch.zeros(self.spec.hidden_size, device=self.device, dtype=torch.float32)
        raw_signed_peak = torch.zeros_like(raw_sum)
        topk_sum = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.float32)
        topk_support = torch.zeros(self.spec.feature_size, device=self.device, dtype=torch.bool)
        with torch.no_grad():
            for start in range(0, int(gradient.shape[0]), self.chunk_size):
                raw_chunk = gradient[start : start + self.chunk_size].to(self.device).float()
                chunk = raw_chunk.to(dtype=self.decoder.dtype)
                raw_sum.add_(raw_chunk.sum(dim=0))
                self._merge_signed_peak_(raw_signed_peak, self._dense_chunk_signed_peak(raw_chunk))
                projected = (chunk @ self.decoder.mT).float()
                self._update_topk_pooling_(projected, topk_sum, topk_support)
        token_count = max(1, int(gradient.shape[0]))
        return SpanProjection(
            raw_mean=raw_sum / token_count,
            raw_signed_peak=raw_signed_peak,
            topk_mean=topk_sum / token_count,
            topk_support=topk_support,
        )
