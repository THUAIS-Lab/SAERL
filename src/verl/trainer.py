import uuid
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward_async
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


class StaticGradientTrainer(RayPPOTrainer):

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.get("gen_batch_size") or self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=False,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=self.config.data.val_batch_size or len(val_dataset),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.total_training_steps = min(
            self.config.trainer.total_training_steps or len(self.train_dataloader), len(self.train_dataloader),
        )
        with open_dict(self.config):
            for key in ("actor_rollout_ref.actor.optim", "critic.optim"):
                optimizer_config = OmegaConf.select(self.config, key)
                if optimizer_config is not None:
                    optimizer_config.total_training_steps = self.total_training_steps

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        sample_id_key = self.config.get("static_gradient_extraction", {}).get("sample_id_key", "index")
        kept_keys = {
            "data_source", "reward_model", "extra_info", "uid", sample_id_key, "static_gradient_is_padding",
        }
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=list(set(batch.non_tensor_batch) - kept_keys),
        )
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _update_actor(self, batch):
        batch.meta_info["global_steps"] = self.global_steps
        return super()._update_actor(batch)

    def _extract_batch(self, batch: DataProto) -> tuple[DataProto, dict, dict]:
        metrics, timing = {}, {}
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        original_size = len(batch)
        dp_size = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
        batch, _ = pad_dataproto_to_divisor(batch, dp_size)
        padding_flags = np.zeros(len(batch), dtype=np.bool_)
        padding_flags[original_size:] = True
        batch.non_tensor_batch["static_gradient_is_padding"] = padding_flags
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        rollout_count = self.config.actor_rollout_ref.rollout.n
        gen_batch = gen_batch.repeat(repeat_times=rollout_count, interleave=True)

        with marked_timer("step", timing):
            with marked_timer("gen", timing):
                rollout = (
                    self.async_rollout_manager.generate_sequences(gen_batch) if self.async_rollout_mode
                    else self.actor_rollout_wg.generate_sequences(gen_batch)
                )
                timing.update(rollout.meta_info.pop("timing", {}))
            batch = batch.repeat(repeat_times=rollout_count, interleave=True).union(rollout)
            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", timing):
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    reward_scores = (
                        self.reward_loop_manager.compute_rm_score(batch) if self.use_reward_loop
                        else self.rm_wg.compute_rm_score(batch)
                    )
                    batch = batch.union(reward_scores)
                if self.config.reward_model.launch_reward_fn_async:
                    reward_future = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
                else:
                    reward_tensor, reward_extra = self._compute_or_extract_reward(
                        batch, reward_fn=self.reward_fn, return_dict=False,
                    )

            if self.use_reference_policy:
                with marked_timer("ref", timing):
                    batch = batch.union(self._compute_ref_log_prob(batch))
            with marked_timer("adv", timing):
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra = ray.get(reward_future)
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra:
                    batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra.items()})
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_count,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm,
                )
            with marked_timer("update_actor", timing):
                actor_output = self._update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        return batch, metrics, timing

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        resume_step = self.config.static_gradient_extraction.get("resume_from_step") or 1
        progress = tqdm(total=self.total_training_steps, initial=resume_step - 1, desc="Gradient extraction")
        for step, batch_dict in enumerate(self.train_dataloader, start=1):
            if step < resume_step:
                continue
            if step > self.total_training_steps:
                break
            self.global_steps = step
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            batch, metrics, timing = self._extract_batch(DataProto.from_single_dict(batch_dict))
            metrics.update({"training/global_step": step, "training/epoch": 0})
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing))
            metrics.update(compute_throughout_metrics(
                batch=batch, timing_raw=timing, n_gpus=self.resource_pool_manager.get_n_gpus(),
            ))
            logger.log(data=metrics, step=step)
            progress.update(1)
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
        progress.close()
