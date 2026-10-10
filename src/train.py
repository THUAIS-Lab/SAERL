"""Command entry points for GRPO training and static gradient extraction."""

from saerl.verl.launcher import launch


def main():
    launch("train")


def extract_main():
    launch("extract")


if __name__ == "__main__":
    main()
