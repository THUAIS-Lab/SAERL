import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from saerl.data.export_subset import _dataset_indices, _take_subset
from saerl.selection.select_topk_response_mean_v1_2 import _select_with_device
from saerl.selection.selected_features import load_projection


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", *map(str, args)], check=True, capture_output=True, text=True)


def test_abs_selection_and_subset_export_preserve_ids_and_rows(representation_run, tmp_path):
    run, indices = representation_run
    output = tmp_path / "selection"
    run_cli(
        "saerl.select",
        "--run-dir",
        run,
        "--feature-indices",
        indices,
        "--cosine-threshold",
        ".8",
        "--device",
        "cpu",
        "--output-dir",
        output,
    )
    assert pq.read_table(output / "selected_sample_ids.parquet")["sample_id"].to_pylist() == [101, 202]
    assert pq.read_table(output / "silent_sample_ids.parquet")["sample_id"].to_pylist() == [404]
    audit = pq.read_table(output / "selection_audit.parquet").to_pylist()
    redundant = next(r for r in audit if r["sample_id"] == 303)
    assert redundant["representative_sample_id"] == 101
    assert redundant["representative_cosine"] == pytest.approx(1)
    source = pa.Table.from_pylist(
        [dict(extra_info=dict(index=i), prompt=f"question {i}") for i in [404, 101, 303, 202]]
    )
    source = source.replace_schema_metadata({b"source": b"fixture"})
    dataset = tmp_path / "source.parquet"
    pq.write_table(source, dataset)
    run_cli("saerl.export_subset", "--run-dir", run, "--selection-dir", output, "--dataset", dataset)
    selected = pq.read_table(output / "selected_deepmath.parquet")
    combined = pq.read_table(output / "selected_plus_silent_deepmath.parquet")
    assert selected.equals(source.take(pa.array([1, 3])), check_metadata=True)
    assert combined.equals(source.take(pa.array([0, 1, 3])), check_metadata=True)
    with pytest.raises(subprocess.CalledProcessError):
        run_cli(
            "saerl.select", "--run-dir", run, "--feature-indices", indices, "--device", "cpu", "--output-dir", output
        )


def test_feature_selection_and_signed_cosine(representation_run, tmp_path):
    run, indices = representation_run
    features = tmp_path / "energy.npy"
    run_cli("saerl.feature_select", "--run-dir", run, "--top-k", 1, "--output", features)
    assert np.load(features).tolist() == [0]
    projection = load_projection(run, indices, input_glob="step-*.parquet", read_batch_size=2)
    active = projection.active_rows
    result, _ = _select_with_device(
        projection.values[active] / projection.strengths[active, None],
        projection.sample_ids[active],
        projection.strengths[active],
        cosine_threshold=0.8,
        block_size=2,
        requested_device="cpu",
    )
    assert set(projection.sample_ids[active[result.selected]]) == {101, 202, 303}


def test_reject_duplicate_and_missing_sample_ids(representation_run, tmp_path):
    run, indices = representation_run
    shard = run / "step-00000001-rank-00000.parquet"
    pq.write_table(pq.read_table(shard), run / "step-00000002-rank-00000.parquet")
    with pytest.raises(ValueError, match="Duplicate"):
        load_projection(run, indices, input_glob="step-*.parquet", read_batch_size=2)
    source = pa.Table.from_pylist([dict(extra_info=dict(index=7))])
    ids = _dataset_indices(source, tmp_path)
    with pytest.raises(ValueError, match="absent"):
        _take_subset(source, ids, np.array([8]), "test")


def test_all_offline_methods_are_importable_without_verl():
    import importlib

    from saerl.feature_select import METHODS as FEATURE_METHODS
    from saerl.select import METHODS

    before = "verl" in sys.modules
    for name in (*METHODS.values(), *FEATURE_METHODS.values()):
        assert callable(importlib.import_module(f"saerl.selection.{name}").main)
    assert ("verl" in sys.modules) == before
