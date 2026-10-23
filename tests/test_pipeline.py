"""CPU integration check for feature preparation and training-row export."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from saerl.selection.feature_selection import ExtractionRuns
from saerl.selection.sample_selection import export_subset

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def run_script(self, script, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_selection_to_curriculum_to_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = []
            for sample_id in range(24):
                row = {
                    "sample_id": sample_id, "difficulty": float(sample_id % 5),
                    "prompt_len_tok": 4, "sol_len_tok": 8, "full_len_tok": 12,
                    "latex_ratio": 0.1, "digit_ratio": 0.2, "sym_ratio": 0.3,
                    "was_truncated": False, "prompt_token_count": 4, "sol_token_count": 8,
                }
                for side, token_count in [("prompt", 4), ("sol", 8)]:
                    value = float(sample_id + 1)
                    row.update({
                        f"{side}_feat_ids": [222], f"{side}_feat_vals": [value / token_count],
                        f"{side}_feat_max": [value], f"{side}_feat_sum": [value],
                        f"{side}_feat_count": [1],
                    })
                rows.append(row)
            source = directory / "source.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            parts = []
            for part_id in range(2):
                part = directory / f"part{part_id}.parquet"
                pq.write_table(pa.Table.from_pylist(rows[part_id::2]), part)
                parts.extend(["--part-path", part])
            merged = directory / "merged"
            self.run_script(
                "materialize_meanmax.py", "--input-root", source, "--output-dir", merged,
                "--expected-rows", 24, "--model-name-or-path", "test-model", "--sae-path", "test-sae",
                "--dtype", "float32", "--batch-size", 2, "--read-batch-size", 8, "--max-seq-len", 4096,
                *parts,
            )
            self.assertEqual(pq.read_table(merged / "samples_meanmax.parquet").to_pylist(), rows)
            features, difficulty, curriculum = [directory / name for name in ["features", "difficulty", "curriculum"]]
            self.run_script("prepare_difficulty_features.py", "--samples-meanmax",
                            merged / "samples_meanmax.parquet", "--output-dir", features)
            table = pq.read_table(features / "difficulty_features.parquet")
            self.assertEqual(sum(name.startswith("f_") for name in table.column_names), 986)
            self.assertEqual(table["f_prompt_mean_222"][0].as_py(), 0.25)
            self.run_script("fit_difficulty.py", "--features-parquet", features / "difficulty_features.parquet",
                            "--backend", "elasticnet", "--clusters", 2, "--labeled-samples", 12,
                            "--output-dir", difficulty)
            train = directory / "train.parquet"
            training_ids = list(reversed(range(24)))
            pq.write_table(pa.Table.from_pylist([
                {"extra_info": {"index": sample_id}, "prompt": str(sample_id)} for sample_id in training_ids
            ]), train)
            shard = directory / "step-00000001.parquet"
            pq.write_table(pa.table({"sample_id": [1, 3, 5], "reward_mean": [1.0, 0.0, 0.0]}), shard)
            runs = ExtractionRuns([directory], [shard], 256, 1e-12, 3)
            subset = directory / "selection" / "subset.parquet"
            export_subset(runs, np.array([1, 7, 9, 11]), train, subset)
            ids_file = subset.parent / "selected_ids.npy"
            np.testing.assert_array_equal(np.load(ids_file), [1, 3, 5, 7, 9, 11])
            self.run_script("build_curriculum.py", "--base-predictions", difficulty / "predictions_clustered.parquet",
                            "--ids-npy", ids_file, "--batch-size", 4, "--mix-tail", 1,
                            "--output-dir", curriculum)
            output = curriculum / "train_order.pt"
            self.run_script("export_training_order.py", "--curriculum-pt", curriculum / "curriculum.pt",
                            "--train-parquet", subset, "--output-pt", output)
            indices = torch.load(output, weights_only=True)
            payload = torch.load(curriculum / "curriculum.pt", weights_only=False)
            subset_ids = pq.read_table(subset).column("extra_info").combine_chunks().field("index").to_pylist()
            self.assertEqual([subset_ids[row] for row in indices.tolist()], payload["indices"].tolist())
            self.assertEqual(len(indices) % 4, 0)


if __name__ == "__main__":
    unittest.main()
