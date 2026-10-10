# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Static gradient actor. Method adaptations: see the verl integration section in README.md.
"""

import re

import torch
from torch import nn
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor, logger


class StaticGradientActor(DataParallelPPOActor):
    """Reuse upstream forward; adapt backward consumption without updating weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_gradient_sae_processor = None
        self._static_gradient_hook_handle = None
        self._static_gradient_target_module_name = None
        self._latest_static_layer_output_grad = None

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        self._latest_static_layer_output_grad = None
        return super()._forward_micro_batch(micro_batch, temperature, calculate_entropy)

    def _optimizer_step(self):
        raise RuntimeError("optimizer.step() is forbidden in static gradient extraction mode")

    @property
    def static_gradient_extraction_enabled(self) -> bool:
        return self.static_gradient_sae_processor is not None

    def __del__(self):
        try:
            self._remove_static_gradient_hook()
        except Exception:
            pass

    def _remove_static_gradient_hook(self) -> None:
        if self._static_gradient_hook_handle is None:
            return
        try:
            self._static_gradient_hook_handle.remove()
        finally:
            self._static_gradient_hook_handle = None
            self._static_gradient_target_module_name = None
            self._latest_static_layer_output_grad = None

    def _register_static_gradient_hook(self, hookpoint: str) -> str:
        """Register the layer-output backward hook on the FSDP actor."""
        self._remove_static_gradient_hook()
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)(?:$|\.)", hookpoint)
        if layer_match is None:
            raise ValueError(f"Cannot infer transformer layer from hookpoint {hookpoint!r}")
        layer_index = int(layer_match.group(1))
        exact = re.compile(rf"(?:^|\.)layers\.{layer_index}(?:$|\._fsdp_wrapped_module$)")
        candidates = [(name, module) for name, module in self.actor_module.named_modules() if hookpoint in name]
        if not candidates:
            raise ValueError(f"No actor module matched SAE hookpoint {hookpoint!r}")
        candidates.sort(key=lambda item: (0 if exact.search(item[0]) else 1, len(item[0])))
        name, module = candidates[0]
        self._static_gradient_target_module_name = name
        self._static_gradient_hook_handle = module.register_full_backward_hook(self._on_static_layer_output_gradient)
        return name

    def _on_static_layer_output_gradient(self, _module: nn.Module, _grad_input, grad_output) -> None:
        if not grad_output or grad_output[0] is None:
            return
        # update_policy consumes this immediately after loss.backward(), so a
        # detached view is sufficient and avoids a full activation-sized copy.
        self._latest_static_layer_output_grad = grad_output[0].detach()

    def _pop_static_layer_output_gradient(self) -> torch.Tensor:
        gradient = self._latest_static_layer_output_grad
        self._latest_static_layer_output_grad = None
        if gradient is None:
            raise RuntimeError("Configured layer backward hook did not capture grad_output[0]")
        return gradient

    def enable_static_gradient_extraction(self, config) -> None:
        """Attach the layer-output hook used by static gradient-SAE mode."""
        if self.actor_optimizer is None:
            raise ValueError("Static gradient extraction can only be enabled on the actor")
        if self.static_gradient_sae_processor is not None:
            raise RuntimeError("Static gradient extraction is already enabled")
        if int(self.ulysses_sequence_parallel_size) != 1:
            raise ValueError("Static gradient-SAE extraction currently requires Ulysses SP size 1")
        from saerl.selection.gradient_sae import StaticGradientSaeProcessor

        processor = StaticGradientSaeProcessor(
            config=config,
            device=get_device_id(),
            use_remove_padding=self.use_remove_padding,
        )
        target_module_name = self._register_static_gradient_hook(processor.hookpoint)
        self.static_gradient_sae_processor = processor
        if torch.distributed.get_rank() == 0:
            print(
                "Static gradient-SAE enabled: "
                f"target={target_module_name}, "
                f"run_dir={self.static_gradient_sae_processor.run_dir}",
                flush=True,
            )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "advantages",
        ]
        if not self.static_gradient_extraction_enabled:
            select_keys.append("old_log_probs")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        if self.static_gradient_extraction_enabled:
            if "token_level_scores" in data.batch.keys():
                select_keys.append("token_level_scores")
            elif "token_level_rewards" in data.batch.keys():
                select_keys.append("token_level_rewards")
            else:
                raise ValueError("Static gradient extraction requires token-level reward scores")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        if self.static_gradient_extraction_enabled:
            for key in (
                "uid",
                self.static_gradient_sae_processor.sample_id_key,
                "extra_info",
                "static_gradient_is_padding",
            ):
                if key in data.non_tensor_batch:
                    non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if self.static_gradient_extraction_enabled:
            self.static_gradient_sae_processor.start_update(int(data.meta_info.get("global_steps", -1)))

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if self.static_gradient_extraction_enabled:
                        old_log_prob = log_prob.detach()
                    elif hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if self.static_gradient_extraction_enabled:
                        layer_output_gradient = self._pop_static_layer_output_gradient()
                        gradient_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
                        # Parameter gradients are not part of the requested
                        # representation. Release them before the dense SAE
                        # projection instead of retaining an entire PPO
                        # mini-batch's accumulation.
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        self.static_gradient_sae_processor.consume_micro_batch(
                            layer_output_gradient,
                            model_inputs,
                            gradient_scale=gradient_scale,
                        )

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                if self.static_gradient_extraction_enabled:
                    # Backward is retained to obtain exactly the training
                    # gradient, but parameters are never stepped.
                    self.actor_optimizer.zero_grad(set_to_none=True)
                else:
                    grad_norm = self._optimizer_step()
                    mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                    append_to_dict(metrics, mini_batch_metrics)
        if self.static_gradient_extraction_enabled:
            static_metrics = self.static_gradient_sae_processor.finish_update()
            append_to_dict(metrics, static_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
