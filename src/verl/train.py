# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_ascend_device_name


def prepare_static_gradient_extraction(config) -> None:
    extraction = config.get("static_gradient_extraction", {})
    if not extraction.get("enable", False):
        return
    if str(config.algorithm.adv_estimator).lower() != "grpo":
        raise ValueError("static_gradient_extraction currently requires algorithm.adv_estimator=grpo")
    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise ValueError("Static gradient-SAE extraction currently requires actor.strategy=fsdp")
    if config.trainer.get("use_legacy_worker_impl", "auto") == "disable":
        raise ValueError("Static extraction requires trainer.use_legacy_worker_impl=enable")
    if config.actor_rollout_ref.rollout.mode == "sync":
        raise ValueError(
            "Rollout mode 'sync' has been removed. Please set "
            "`actor_rollout_ref.rollout.mode=async` to use the native server rollout."
        )

    resume_from_step = extraction.get("resume_from_step", None)
    if resume_from_step is not None:
        resume_from_step = int(resume_from_step)
        if resume_from_step < 1:
            raise ValueError("static_gradient_extraction.resume_from_step must be at least 1")

    configured_run_name = str(extraction.get("run_name", "") or "").strip()
    if resume_from_step is not None and not configured_run_name:
        raise ValueError("Static extraction resume requires static_gradient_extraction.run_name")
    run_name = configured_run_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = Path(str(extraction.output_dir)).expanduser().resolve() / run_name
    if resume_from_step is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise FileNotFoundError(f"Static extraction resume directory does not exist: {run_dir}")

    extraction.run_name = run_name
    extraction.run_dir = str(run_dir)
    extraction.resume_from_step = resume_from_step
    extraction.num_rollouts = int(config.actor_rollout_ref.rollout.n)
    config.actor_rollout_ref.static_gradient_extraction = OmegaConf.create(
        OmegaConf.to_container(extraction, resolve=True)
    )
    OmegaConf.resolve(config)

    config_name = (
        "resolved_config.yaml"
        if resume_from_step is None
        else f"resolved_config.resume-step-{resume_from_step:08d}.yaml"
    )
    (run_dir / config_name).write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    if resume_from_step is None:
        print(f"Static gradient-SAE output directory: {run_dir}", flush=True)
    else:
        print(f"Static gradient-SAE resuming run_dir={run_dir} from step={resume_from_step}", flush=True)


class SAERLTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        if not config.static_gradient_extraction.enable:
            return super().add_actor_rollout_worker(config)
        from verl.single_controller.ray import RayWorkerGroup

        from saerl.verl.worker import StaticGradientWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(StaticGradientWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return StaticGradientWorker, RayWorkerGroup

    def run(self, config):
        if not config.static_gradient_extraction.enable:
            return super().run(config)
        prepare_static_gradient_extraction(config)

        from verl.utils.fs import copy_to_local

        from saerl.verl.trainer import StaticGradientTrainer

        pprint(OmegaConf.to_container(config, resolve=True))

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = StaticGradientTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def launch(config_name):
    @hydra.main(config_path="pkg://saerl.config", config_name=config_name, version_base=None)
    def run(config):
        auto_set_ascend_device_name(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SAERLTaskRunner))

    run()


def main() -> None:
    launch("train")


if __name__ == "__main__":
    main()
