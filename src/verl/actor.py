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

    def _remove_static_gradient_hook(self) -> None:
        if self._static_gradient_hook_handle is None:
            return
        self._static_gradient_hook_handle.remove()
        self._static_gradient_hook_handle = None
        self._static_gradient_target_module_name = None
        self._latest_static_layer_output_grad = None

    def _register_static_gradient_hook(self, hookpoint: str) -> str:
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
        self._latest_static_layer_output_grad = grad_output[0].detach()

    def _pop_static_layer_output_gradient(self) -> torch.Tensor:
        gradient = self._latest_static_layer_output_grad
        self._latest_static_layer_output_grad = None
        if gradient is None:
            raise RuntimeError("Configured layer backward hook did not capture grad_output[0]")
        return gradient

    def enable_static_gradient_extraction(self, config) -> None:
        if int(self.ulysses_sequence_parallel_size) != 1:
            raise ValueError("Static gradient-SAE extraction currently requires Ulysses SP size 1")
        from saerl.selection.extraction import StaticGradientSaeProcessor

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
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        batch_keys = [
            "responses", "response_mask", "input_ids", "attention_mask", "position_ids",
            "advantages", "token_level_scores",
        ]
        if self.config.use_kl_loss:
            batch_keys.append("ref_log_prob")
        non_tensor_keys = [
            key for key in (
                "multi_modal_inputs", "uid", self.static_gradient_sae_processor.sample_id_key,
                "extra_info", "static_gradient_is_padding",
            ) if key in data.non_tensor_batch
        ]
        data = data.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
        self.static_gradient_sae_processor.start_update(int(data.meta_info["global_steps"]))
        metrics = {}
        for mini_batch in data.split(self.config.ppo_mini_batch_size):
            if self.config.use_dynamic_bsz:
                micro_batches, _ = prepare_dynamic_batch(
                    mini_batch, max_token_len=self.config.ppo_max_token_len_per_gpu,
                )
            else:
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
            self.actor_optimizer.zero_grad(set_to_none=True)
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                response_mask = model_inputs["response_mask"]
                loss_scale = (
                    response_mask.shape[0] / self.config.ppo_mini_batch_size
                    if self.config.use_dynamic_bsz else 1 / self.gradient_accumulation
                )
                entropy_coeff = self.config.entropy_coeff
                calculate_entropy = self.config.calculate_entropy or entropy_coeff != 0
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                )
                policy_loss_fn = get_policy_loss_fn(self.config.policy_loss.get("loss_mode", "vanilla"))
                pg_loss, micro_metrics = policy_loss_fn(
                    old_log_prob=log_prob.detach(),
                    log_prob=log_prob,
                    advantages=model_inputs["advantages"],
                    response_mask=response_mask,
                    loss_agg_mode=self.config.loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=None,
                )
                policy_loss = pg_loss
                if calculate_entropy and entropy is not None:
                    entropy_mean = agg_loss(
                        loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode,
                    )
                    micro_metrics["actor/entropy"] = entropy_mean.detach().item()
                    if entropy_coeff != 0:
                        policy_loss -= entropy_coeff * entropy_mean
                if self.config.use_kl_loss:
                    kld = kl_penalty(
                        logprob=log_prob, ref_logprob=model_inputs["ref_log_prob"],
                        kl_penalty=self.config.kl_loss_type,
                    )
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    micro_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale
                    micro_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                loss = policy_loss * loss_scale
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                gradient = self._pop_static_layer_output_gradient()
                gradient_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.static_gradient_sae_processor.consume_micro_batch(
                    gradient, model_inputs, gradient_scale=gradient_scale,
                )
                micro_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale
                append_to_dict(metrics, micro_metrics)
        append_to_dict(metrics, self.static_gradient_sae_processor.finish_update())
        return metrics
