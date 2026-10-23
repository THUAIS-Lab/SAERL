"""OpenSAE adapter for activation extraction."""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict

import torch

os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_opensae", "0.0.0")

_OPENSAE_DTYPE_CONFIGURED = False


def _configure_opensae_dtype() -> None:
    """Resolve OpenSAE dtype settings from strings or torch.dtype values."""
    global _OPENSAE_DTYPE_CONFIGURED  # noqa: PLW0603
    if _OPENSAE_DTYPE_CONFIGURED:
        return

    try:
        from opensae.config_utils import PretrainedSaeConfig  # type: ignore
    except Exception:  # noqa: BLE001
        return

    def _safe_get_torch_dtype(self) -> torch.dtype:  # type: ignore[no-untyped-def]
        td = getattr(self, "torch_dtype", None)
        if isinstance(td, torch.dtype):
            return td
        if td is None:
            return torch.float32
        if isinstance(td, str):
            name = td.replace("torch.", "")
            dtype = getattr(torch, name, None)
            if isinstance(dtype, torch.dtype):
                return dtype
        warnings.warn(f"Unrecognized OpenSAE torch_dtype={td!r}; defaulting to float32", stacklevel=2)
        return torch.float32

    PretrainedSaeConfig.get_torch_dtype = _safe_get_torch_dtype  # type: ignore[method-assign]
    _OPENSAE_DTYPE_CONFIGURED = True


class OpenSaeRunner:
    """Adapter around OpenSAE to extract structured activations."""

    def __init__(self, model_name_or_path: str, sae_path: str, device: str, dtype: torch.dtype) -> None:
        from opensae import OpenSae  # type: ignore
        from opensae.transformer_with_sae import TransformerWithSae  # type: ignore
        from transformers import AutoModelForCausalLM

        _configure_opensae_dtype()

        try:
            self.sae = OpenSae.from_pretrained(sae_path, dtype=dtype)
        except TypeError:
            self.sae = OpenSae.from_pretrained(sae_path, torch_dtype=dtype)

        try:
            transformer = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                trust_remote_code=True,
            )
        except TypeError:
            transformer = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        self.model = TransformerWithSae(transformer, self.sae, device=device)
        self.device = device
        self.dtype = dtype
        self.feature_size = int(getattr(self.sae.config, "feature_size", 0) or 0)

    def extract_structured(self, encodings: Dict[str, torch.Tensor], tokenizer: Any) -> Any:
        """Extract structured token data with activations."""
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.inference_mode():
            return self.model.extract_data(encodings, tokenizer)

    def extract_sparse(
        self, encodings: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract sparse feature indices/activations aligned to tokens.

        Returns:
          - feature_indices: (batch, seq_len, k) int64
          - feature_activations: (batch, seq_len, k) float
          - input_ids: (batch, seq_len) int64 (on device)
          - attention_mask: (batch, seq_len) int64 (on device)
        """
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.inference_mode():
            encoder_output, _ = self.model.forward(return_features=True, use_cache=False, **encodings)
        indices = encoder_output.sparse_feature_indices
        activations = encoder_output.sparse_feature_activations
        bsz, seq_len = encodings["input_ids"].shape
        indices = indices.view(bsz, seq_len, -1)
        activations = activations.view(bsz, seq_len, -1)
        return indices, activations, encodings["input_ids"], encodings["attention_mask"]
