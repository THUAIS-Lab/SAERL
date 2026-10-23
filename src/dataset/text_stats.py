"""Text statistics helpers."""
from __future__ import annotations

import re
from typing import Dict

LATEX_TOKENS = ["\\", "\\frac", "\\sqrt", "^", "_", "{", "}"]
SYMBOL_CHARS = set("=+-*/()[]<>|∑∫≈≠≤≥×÷")


def _ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(count) / float(total)


def compute_text_stats(problem: str, solution: str) -> Dict[str, float | int]:
    """Compute basic text statistics for a sample."""
    text = (problem or "") + "\n" + (solution or "")
    char_len_problem = len(problem or "")
    char_len_solution = len(solution or "")
    total_chars = max(len(text), 1)

    latex_count = sum(text.count(tok) for tok in LATEX_TOKENS)
    digit_count = sum(ch.isdigit() for ch in text)
    sym_count = sum(ch in SYMBOL_CHARS for ch in text)

    return {
        "char_len_problem": char_len_problem,
        "char_len_solution": char_len_solution,
        "latex_ratio": _ratio(latex_count, total_chars),
        "digit_ratio": _ratio(digit_count, total_chars),
        "sym_ratio": _ratio(sym_count, total_chars),
    }
