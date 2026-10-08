"""Select with the existing methods, defaulting to absolute selected-feature cosine."""

import argparse
import importlib
import sys

METHODS = {
    "abs-selected": "select_topk_response_mean_abs_selected_features_v1_2_3",
    "abs": "select_topk_response_mean_abs_v1_2_4",
    "signed-selected": "select_topk_response_mean_selected_features_v1_2_2",
    "signed": "select_topk_response_mean_v1_2",
    "dense": "select_dense_response_mean_v1_2",
    "mean-peak": "select_topk_response_mean_peak_v1",
    "top-k": "select_v1_3",
    "coverage": "select_v1_4",
    "raw-abs": "raw.select_raw_grad_abs",
    "raw-abs-selected": "raw.select_raw_grad_abs_selected_features",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=tuple(METHODS), default="abs-selected")
    args, rest = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *rest]
        importlib.import_module(f"saerl.selection.{METHODS[args.method]}").main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
