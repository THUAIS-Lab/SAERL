"""Feature selection without a training runtime dependency."""

import argparse
import importlib
import sys

METHODS = {
    "energy": "features.select_energy_features",
    "variance": "features.select_variance_features",
    "prevalence-variance": "features.select_prevalence_variance_features",
    "raw-energy": "raw.select_energy_features",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=tuple(METHODS), default="energy")
    args, rest = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *rest]
        importlib.import_module(f"saerl.selection.{METHODS[args.method]}").main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
