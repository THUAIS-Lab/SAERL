"""Two-GPU tail/resume extraction followed by one GRPO step on the selected subset."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch


def run(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "model", "sae", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        parser.error("This smoke test requires at least two visible CUDA GPUs")
    from saerl.integrations.verl.compat import verify_verl

    verify_verl()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    table = pq.read_table(args.dataset).slice(0, 4)
    if table.num_rows != 4:
        parser.error("The source dataset must have at least four rows")
    ids = [int(row["extra_info"]["index"]) for row in table.to_pylist()]
    if len(set(ids)) != 4:
        parser.error("The first four rows must have unique extra_info.index values")
    dataset = output / "input.parquet"
    pq.write_table(table, dataset)
    common = [
        f"data.train_files={dataset}",
        f"data.val_files={dataset}",
        "data.train_batch_size=3",
        "data.max_prompt_length=1024",
        "data.max_response_length=128",
        "data.dataloader_num_workers=0",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={args.model.resolve()}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.logger=[console]",
        "trainer.use_legacy_worker_impl=enable",
    ]
    extraction = [
        f"static_gradient_extraction.sae_path={args.sae.resolve()}",
        f"static_gradient_extraction.output_dir={output}",
        "static_gradient_extraction.run_name=extraction",
    ]
    run("saerl.extract", *common, *extraction)
    run_dir = output / "extraction"
    shards = sorted(run_dir.glob("step-*.parquet"))
    observed = [
        value for path in shards for value in pq.read_table(path, columns=["sample_id"])["sample_id"].to_pylist()
    ]
    if sorted(observed) != sorted(ids) or len(shards) != 4:
        raise AssertionError(f"Tail/DP extraction lost or duplicated samples: {observed}; shards={shards}")
    # Remove only the test's own generated final rank file to simulate partial completion.
    missing = run_dir / "step-00000002-rank-00001.parquet"
    retained = run_dir / "step-00000002-rank-00000.parquet"
    retained_bytes = retained.read_bytes()
    missing.unlink()
    run("saerl.extract", *common, *extraction, "static_gradient_extraction.resume_from_step=2")
    if not missing.is_file() or retained.read_bytes() != retained_bytes:
        raise AssertionError("Resume did not preserve an existing rank file and restore the missing rank")
    feature_size = json.loads((run_dir / "representation_manifest.json").read_text())["sae_feature_size"]
    features = output / "features.npy"
    run("saerl.feature_select", "--run-dir", run_dir, "--top-k", min(256, feature_size), "--output", features)
    selection = output / "selection"
    run(
        "saerl.select",
        "--run-dir",
        run_dir,
        "--feature-indices",
        features,
        "--device",
        "cpu",
        "--cosine-threshold",
        ".7",
        "--output-dir",
        selection,
    )
    run("saerl.export_subset", "--run-dir", run_dir, "--selection-dir", selection, "--dataset", dataset)
    subset = selection / "selected_plus_silent_deepmath.parquet"
    if pq.read_metadata(subset).num_rows < 2:
        raise RuntimeError("The four-prompt fixture yielded fewer than two selected/silent rows; use a diverse fixture")
    # Later overrides replace the three-prompt extraction batch with a two-prompt training batch.
    run(
        "saerl.train",
        *common,
        f"data.train_files={subset}",
        "data.train_batch_size=2",
        "algorithm.adv_estimator=grpo",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=1",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={output / 'checkpoints'}",
    )
    print("PASS: two-GPU extraction, tail IDs, partial-rank resume, selection/export, one GRPO step and checkpoint")


if __name__ == "__main__":
    main()
