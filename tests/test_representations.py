import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from saerl.extraction.representations import SaeDecoderProjector, StaticGradientSaeProcessor


def test_signed_pooling_and_chunk_independence(sae_checkpoint):
    gradients = torch.tensor([[2.0, -1.0], [-2.0, 3.0], [1.0, 0.0]])
    results = []
    for chunk in (1, 2, 32):
        projector = SaeDecoderProjector(
            sae_checkpoint,
            device="cpu",
            dtype=torch.float32,
            chunk_size=chunk,
            gradient_feature_topk=1,
            decoder_normalization="checkpoint_raw",
        )
        results.append(projector.project_and_pool(gradients))
    for result in results:
        torch.testing.assert_close(result.raw_mean, torch.tensor([1 / 3, 2 / 3]))
        torch.testing.assert_close(result.raw_signed_peak, torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(result.dense_mean, torch.tensor([1 / 3, 2 / 3, 1.0]))
        torch.testing.assert_close(result.dense_signed_peak, torch.tensor([2.0, 3.0, 1.0]))
        torch.testing.assert_close(result.topk_mean, results[0].topk_mean)
        torch.testing.assert_close(result.topk_signed_peak, results[0].topk_signed_peak)
        assert torch.equal(result.topk_support, results[0].topk_support)


def make_processor(checkpoint, run, **kwargs):
    config = OmegaConf.create(
        dict(sae_path=str(checkpoint), run_dir=str(run), num_rollouts=2, log_every_micro_batches=0, **kwargs)
    )
    return StaticGradientSaeProcessor(config=config, device="cpu", use_remove_padding=False)


def test_rollout_averaging_padding_and_incomplete_group(sae_checkpoint, tmp_path):
    processor = make_processor(sae_checkpoint, tmp_path / "run")
    processor.start_update(1)
    inputs = dict(
        input_ids=torch.ones(3, 3, dtype=torch.long),
        responses=torch.ones(3, 1, dtype=torch.long),
        attention_mask=torch.ones(3, 3),
        response_mask=torch.ones(3, 1),
        advantages=torch.tensor([[1.0], [-1.0], [0.0]]),
        token_level_scores=torch.tensor([[1.0], [0.0], [1.0]]),
        uid=np.array(["group", "group", "padding"], dtype=object),
        index=np.array([99, 99, 99]),
        static_gradient_is_padding=np.array([False, False, True]),
    )
    gradient = torch.ones(3, 3, 2)
    gradient[1] *= -1
    processor.consume_micro_batch(gradient * 8, inputs, gradient_scale=8)
    metrics = processor.finish_update()
    rows = pq.read_table(tmp_path / "run/step-00000001-rank-00000.parquet").to_pylist()
    assert metrics["static_gradient/optimizer_step"] == 0
    assert len(rows) == 1 and rows[0]["sample_id"] == 99
    assert rows[0]["reward_mean"] == 0.5
    assert rows[0]["raw_grad_vals"] == [0.0] * 8
    assert not list((tmp_path / "run").glob("*.inprogress"))
    processor.start_update(2)
    one = {key: value[:1] for key, value in inputs.items()}
    processor.consume_micro_batch(gradient[:1], one)
    with pytest.raises(RuntimeError, match="expected 2"):
        processor.finish_update()


def test_partial_rank_resume_preserves_existing_shard(sae_checkpoint, tmp_path):
    run = tmp_path / "resume"
    inputs = dict(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        responses=torch.ones(2, 1, dtype=torch.long),
        attention_mask=torch.ones(2, 3),
        response_mask=torch.ones(2, 1),
        advantages=torch.tensor([[1.0], [-1.0]]),
        token_level_scores=torch.tensor([[1.0], [0.0]]),
        uid=np.array(["group", "group"], dtype=object),
        index=np.array([99, 99]),
    )
    first = make_processor(sae_checkpoint, run)
    first.start_update(1)
    first.consume_micro_batch(torch.ones(2, 3, 2), inputs)
    first.finish_update()
    retained = run / "step-00000001-rank-00000.parquet"
    original = retained.read_bytes()
    for rank in (0, 1):
        resumed = make_processor(sae_checkpoint, run, resume_from_step=1)
        resumed.rank = rank
        resumed.start_update(1)
        resumed.consume_micro_batch(torch.full((2, 3, 2), 2.0), inputs)
        resumed.finish_update()
    assert retained.read_bytes() == original
    assert (run / "step-00000001-rank-00001.parquet").is_file()
