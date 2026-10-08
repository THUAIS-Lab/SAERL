import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def representation_run(tmp_path):
    run = tmp_path / "representations"
    run.mkdir()
    (run / "representation_manifest.json").write_text(
        json.dumps(
            {
                "sae_feature_size": 3,
                "gradient_zero_eps": 1e-12,
                "representation_layout": ["prompt_mean", "prompt_signed_peak", "response_mean", "response_signed_peak"],
                "topk_representation_size": 12,
            }
        )
    )
    schema = pa.schema(
        [
            ("sample_id", pa.int64()),
            ("topk_response_feat_ids", pa.list_(pa.int64())),
            ("topk_response_feat_mean", pa.list_(pa.float32())),
            ("zero_signal_token_topk", pa.bool_()),
            ("reward_mean", pa.float32()),
        ]
    )
    rows = [
        dict(
            sample_id=303,
            topk_response_feat_ids=[0, 1],
            topk_response_feat_mean=[-3, -1],
            zero_signal_token_topk=False,
            reward_mean=1,
        ),
        dict(
            sample_id=101,
            topk_response_feat_ids=[0, 1],
            topk_response_feat_mean=[3, 1],
            zero_signal_token_topk=False,
            reward_mean=1,
        ),
        dict(
            sample_id=202,
            topk_response_feat_ids=[1],
            topk_response_feat_mean=[2],
            zero_signal_token_topk=False,
            reward_mean=1,
        ),
        dict(
            sample_id=404,
            topk_response_feat_ids=[],
            topk_response_feat_mean=[],
            zero_signal_token_topk=True,
            reward_mean=0,
        ),
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), run / "step-00000001-rank-00000.parquet")
    indices = tmp_path / "features.npy"
    np.save(indices, np.array([0, 1], dtype=np.int64))
    return run, indices


@pytest.fixture
def sae_checkpoint(tmp_path):
    import torch

    checkpoint = tmp_path / "sae"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"feature_size": 3, "hidden_size": 2, "input_hookpoint": "layers.0"})
    )
    torch.save({"W_dec": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])}, checkpoint / "weights.pt")
    return checkpoint
