# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# TaskRunner.run adapted from the pinned upstream; see the verl integration section in README.md.
import os
import socket

import ray
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.config import validate_config

from saerl.integrations.verl.config import prepare_static_gradient_extraction
from saerl.integrations.verl.trainer import StaticGradientTrainer


class SAERLTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        if not config.static_gradient_extraction.enable:
            return super().add_actor_rollout_worker(config)
        from verl.single_controller.ray import RayWorkerGroup

        from saerl.integrations.verl.worker import StaticGradientWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(StaticGradientWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        return StaticGradientWorker, RayWorkerGroup

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        from saerl.integrations.verl.compat import verify_verl

        verify_verl()
        prepare_static_gradient_extraction(config)

        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer_cls = StaticGradientTrainer if config.static_gradient_extraction.enable else RayPPOTrainer
        trainer = trainer_cls(
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
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()
