#!/usr/bin/env python3
import argparse
import datetime as dt
import time
import os
import re
import glob
import shutil
import subprocess
import sys
from tqdm import tqdm

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))

def create_hf_model(p):
    ckpt_paths = glob.glob(os.path.join(p, '*/'))
    ckpt_paths = [ckpt for ckpt in ckpt_paths if "actor" in os.listdir(ckpt)]
    steps = [extract_step_from_path(i) for i in ckpt_paths]
    print(f"Found {len(ckpt_paths)} checkpoints: {ckpt_paths}")
    if len(steps) != 0:
        max_step = max(steps)
        # ckpt_paths = [p for p in ckpt_paths if (extract_step_from_path(p) < (max_step-20))]
        print(len(ckpt_paths))
        print(ckpt_paths)
        print(max_step)

        for ckpt_path in tqdm(ckpt_paths, desc="Processing", total=len(ckpt_paths)):
            hf_model_dir = os.path.join(ckpt_path, "merged_hf_model")
            os.makedirs(hf_model_dir, exist_ok=True)
            actor_dir = os.path.join(ckpt_path, "actor")
            command = [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                actor_dir,
                "--target_dir",
                hf_model_dir,
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[{now_str()}] Merge failed for {actor_dir} "
                    f"(exit code {exc.returncode}); keeping source checkpoint.",
                    flush=True,
                )
                continue
            
            if extract_step_from_path(ckpt_path) < max_step:
                print(f"[{now_str()}] Removing source checkpoint {actor_dir}...", flush=True)
                shutil.rmtree(actor_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_dir")
    args = parser.parse_args()

    interval_hours = 0.4
    interval_seconds = max(1, int(interval_hours * 3600))

    print(
        f"[{now_str()}] Start periodic sync. dir={args.local_dir}, interval={interval_hours}h",
        flush=True,
    )
    while True:
        create_hf_model(args.local_dir)
        print(f"[{now_str()}] Sleeping {interval_seconds}s...", flush=True)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
