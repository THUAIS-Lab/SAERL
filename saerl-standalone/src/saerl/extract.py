"""One-pass static gradient extraction; accepts the existing Hydra overrides."""

from saerl.train import launch


def main():
    launch("extract")


if __name__ == "__main__":
    main()
