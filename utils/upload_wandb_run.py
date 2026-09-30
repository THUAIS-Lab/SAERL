#!/usr/bin/env python3
import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path


def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sync_once(target_dir: Path) -> int:
    cmd = ["wandb", "sync", str(target_dir)]
    print(f"[{now_str()}] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=False)
    print(f"[{now_str()}] Exit code: {result.returncode}", flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "local_dir",
        type=Path,
        help="Local offline W&B run directory, e.g. wandb/offline-run-xxxx",
    )
    args = parser.parse_args()

    interval_hours = 0.1
    
    interval_seconds = max(1, int(interval_hours * 3600))

    print(
        f"[{now_str()}] Start periodic sync. dir={args.local_dir}, interval={interval_hours}h",
        flush=True,
    )
    while True:
        sync_once(args.local_dir)
        print(f"[{now_str()}] Sleeping {interval_seconds}s...", flush=True)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
