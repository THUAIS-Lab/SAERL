"""Run against the unmodified official dependency, with a tiny CPU actor."""

import copy
import inspect
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def official_verl():
    from saerl.integrations.verl.compat import verify_verl

    try:
        verify_verl()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_package_config_and_external_reward_loader():
    from hydra import compose, initialize_config_module
    from verl.trainer.ppo.reward import get_custom_reward_fn

    with initialize_config_module(config_module="saerl.config", version_base=None):
        config = compose(config_name="extract")
    assert config.static_gradient_extraction.enable
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.data.saerl.enable is False
    reward = get_custom_reward_fn(config)
    assert reward(data_source="openai/gsm8k", solution_str="#### 42", ground_truth="42", extra_info={}) == 1
    assert reward(data_source="zwhe99/DeepMath-103K", solution_str=r"\boxed{42}", ground_truth="42", extra_info={}) == 1


def test_extraction_preparation_and_resume_contract(sae_checkpoint, tmp_path):
    from hydra import compose, initialize_config_module

    from saerl.integrations.verl.config import prepare_static_gradient_extraction

    with initialize_config_module(config_module="saerl.config", version_base=None):
        config = compose(
            config_name="extract",
            overrides=[
                f"static_gradient_extraction.sae_path={sae_checkpoint}",
                f"static_gradient_extraction.output_dir={tmp_path}",
                "static_gradient_extraction.run_name=run",
            ],
        )
    prepare_static_gradient_extraction(config)
    assert config.trainer.total_epochs == 1
    assert config.trainer.save_freq == -1 and config.trainer.test_freq == -1
    assert config.trainer.balance_batch is False and config.data.shuffle is False
    assert config.actor_rollout_ref.static_gradient_extraction.run_dir == str(tmp_path / "run")
    assert (tmp_path / "run/resolved_config.yaml").is_file()
    with pytest.raises(FileExistsError):
        prepare_static_gradient_extraction(config)
    config.static_gradient_extraction.resume_from_step = 1
    prepare_static_gradient_extraction(config)
    assert (tmp_path / "run/resolved_config.resume-step-00000001.yaml").is_file()


@pytest.mark.parametrize("dynamic", [False, True])
def test_actor_backward_matches_upstream_without_optimizer_step(sae_checkpoint, tmp_path, monkeypatch, dynamic):
    from verl import DataProto
    from verl.workers.actor import dp_actor as upstream
    from verl.workers.config import FSDPActorConfig

    from saerl.integrations.verl import actor as adapter

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(upstream, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(adapter, "get_device_id", lambda: "cpu")

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
            self.head = torch.nn.Linear(2, 1, bias=False)

        def forward(self, ids):
            features = torch.stack((ids.float(), ids.float() % 3 + 1), dim=-1)
            hidden = self.layers[0](features)
            return self.head(hidden[:, -1, :]) / 10, hidden[:, -1, :1].square() / 20

    torch.manual_seed(7)
    model = Policy()
    baseline_model = copy.deepcopy(model)
    config = FSDPActorConfig(
        rollout_n=2,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        use_dynamic_bsz=dynamic,
        ppo_max_token_len_per_gpu=3,
        use_torch_compile=False,
        entropy_coeff=0.02,
        use_kl_loss=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step = Mock(side_effect=AssertionError("static optimizer was stepped"))
    static = adapter.StaticGradientActor(config, model, optimizer)
    baseline = upstream.DataParallelPPOActor(
        copy.deepcopy(config), baseline_model, torch.optim.SGD(baseline_model.parameters(), lr=0)
    )

    def forward(self, inputs, temperature, calculate_entropy=False):
        if hasattr(self, "_latest_static_layer_output_grad"):
            self._latest_static_layer_output_grad = None
        log_probs, entropy = self.actor_module(inputs["input_ids"])
        return entropy if calculate_entropy else None, log_probs

    static._forward_micro_batch = MethodType(forward, static)
    baseline._forward_micro_batch = MethodType(forward, baseline)
    extraction = OmegaConf.create(
        dict(sae_path=str(sae_checkpoint), run_dir=str(tmp_path / "run"), num_rollouts=2, log_every_micro_batches=0)
    )
    static.enable_static_gradient_extraction(extraction)
    captured_static, captured_baseline = [], []
    model.layers[0].register_full_backward_hook(lambda m, gi, go: captured_static.append(go[0].detach().clone()))
    baseline_model.layers[0].register_full_backward_hook(
        lambda m, gi, go: captured_baseline.append(go[0].detach().clone())
    )
    ids = torch.tensor([[1, 2, 3], [1, 2, 4]])
    with torch.no_grad():
        old_log_probs, _ = model(ids)
    data = DataProto.from_dict(
        tensors=dict(
            input_ids=ids,
            responses=ids[:, -1:],
            attention_mask=torch.ones(2, 3),
            position_ids=torch.arange(3).expand(2, 3),
            response_mask=torch.ones(2, 1),
            advantages=torch.tensor([[1.0], [-1.0]]),
            old_log_probs=old_log_probs,
            ref_log_prob=torch.zeros(2, 1),
            token_level_scores=torch.tensor([[1.0], [0.0]]),
        ),
        non_tensors=dict(uid=np.array(["group", "group"], dtype=object), index=np.array([99, 99])),
        meta_info=dict(temperature=1.0, global_steps=1),
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    expected_metrics = baseline.update_policy(copy.deepcopy(data))
    metrics = static.update_policy(data)
    assert len(captured_static) == len(captured_baseline) == 2
    for actual, expected in zip(captured_static, captured_baseline, strict=True):
        torch.testing.assert_close(actual, expected)
    assert metrics["actor/pg_loss"] == pytest.approx(expected_metrics["actor/pg_loss"])
    optimizer.step.assert_not_called()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(RuntimeError, match="forbidden"):
        static._optimizer_step()
    static._remove_static_gradient_hook()


def test_worker_preserves_dispatch_and_does_not_step_scheduler(monkeypatch):
    from verl import DataProto
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    from saerl.integrations.verl import worker as adapter

    assert issubclass(adapter.StaticGradientWorker, AsyncActorRolloutRefWorker)
    assert getattr(adapter.StaticGradientWorker.init_model, MAGIC_ATTR)
    assert getattr(adapter.StaticGradientWorker.update_actor, MAGIC_ATTR)
    scheduler = Mock()
    scheduler.get_last_lr.return_value = [0.001]
    device = SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    monkeypatch.setattr(adapter, "get_torch_device", lambda: device)
    worker = SimpleNamespace(
        _is_actor=True,
        _is_offload_param=False,
        _is_offload_optimizer=False,
        ulysses_sharding_manager=nullcontext(),
        actor=SimpleNamespace(update_policy=lambda data: {"static_gradient/optimizer_step": [0.0]}),
        actor_lr_scheduler=scheduler,
        flops_counter=SimpleNamespace(estimate_flops=lambda n, t: (1, 2)),
        config=OmegaConf.create({"actor": {"ppo_epochs": 1}}),
        world_size=1,
    )
    output = inspect.unwrap(adapter.StaticGradientWorker.update_actor)(
        worker, DataProto(meta_info={"global_token_num": [3, 3]})
    )
    assert output.meta_info["metrics"]["actor/lr"] == 0.001
    scheduler.step.assert_not_called()


def test_tail_dataloader_fixed_sampler_and_generation_metadata(tmp_path):
    from verl import DataProto

    from saerl.integrations.verl.sampler import SAERLSampler
    from saerl.integrations.verl.trainer import StaticGradientTrainer

    order = tmp_path / "positions.pt"
    torch.save(torch.tensor([3, 1, 2, 0]), order)
    data_config = OmegaConf.create(
        dict(
            saerl=dict(enable=True, processed_file=str(order)),
            gen_batch_size=None,
            train_batch_size=3,
            val_batch_size=1,
            dataloader_num_workers=0,
        )
    )
    del data_config.gen_batch_size
    sampler = SAERLSampler(list(range(4)), data_config)
    assert list(sampler) == [3, 1, 2, 0]
    trainer = object.__new__(StaticGradientTrainer)
    trainer.static_gradient_extraction = True
    trainer.config = OmegaConf.create(
        dict(data=OmegaConf.to_container(data_config), trainer=dict(total_epochs=1, total_training_steps=None))
    )
    trainer._create_dataloader(list(range(4)), [0], lambda rows: rows, sampler)
    assert list(trainer.train_dataloader) == [[3, 1, 2], [0]]
    assert trainer.total_training_steps == 2
    trainer.async_rollout_mode = True
    batch = DataProto.from_dict(
        tensors=dict(input_ids=torch.ones(2, 3), attention_mask=torch.ones(2, 3), position_ids=torch.ones(2, 3)),
        non_tensors=dict(
            index=np.array([99, 100]),
            uid=np.array(["a", "b"], dtype=object),
            static_gradient_is_padding=np.array([False, True]),
        ),
    )
    generated = trainer._get_gen_batch(batch)
    assert generated.non_tensor_batch["index"].tolist() == [99, 100]
    assert batch.non_tensor_batch["static_gradient_is_padding"].tolist() == [False, True]
