"""Chat tokenization and prompt/solution span boundaries."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


def build_messages(problem: str, solution: str | None) -> List[Dict[str, str]]:
    """Build chat messages for system/user/assistant."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": problem},
    ]
    if solution is not None:
        messages.append({"role": "assistant", "content": solution})
    return messages


def _ensure_chat_template(tokenizer: object) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer must support apply_chat_template.")
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("Tokenizer must provide the chat template for the SAE source model.")


def encode_full_and_cut(tokenizer: object, problem: str, solution: str) -> Tuple[torch.Tensor, int]:
    """Encode full chat and compute cut index for prompt/solution boundary."""
    _ensure_chat_template(tokenizer)
    full_messages = build_messages(problem, solution)
    prefix_messages = build_messages(problem, None)

    full_ids = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )[0]
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0]
    cut = int(prefix_ids.numel())
    return full_ids, cut


def truncate_keep_prompt(
    full_ids: torch.Tensor, cut: int, max_seq_len: int
) -> Tuple[torch.Tensor, int, bool]:
    """Preserve the prompt and truncate solution tokens to the sequence limit."""
    full_len = int(full_ids.numel())
    if full_len <= max_seq_len:
        return full_ids, cut, False
    if cut >= max_seq_len:
        logger.warning("Prompt exceeds max_seq_len; truncating prompt to max_seq_len.")
        return full_ids[:max_seq_len], max_seq_len, True
    kept = torch.cat([full_ids[:cut], full_ids[cut : cut + (max_seq_len - cut)]])
    return kept, cut, True
