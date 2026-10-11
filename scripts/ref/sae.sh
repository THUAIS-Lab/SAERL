#!/bin/bash
set -x

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.experiment_name='sel_hidden_diff_sae_batch_v4mix8_run2' \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" \
    trainer.n_gpus_per_node=4 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/sel_hidden_diff_sae_batch_v4mix8_run2/" \
    data.shuffle=False \
    data.saerl.enable=enable \
    data.saerl.processed_file="/home/fit/lijuanzi1/WORK/dz/SAERL/rank_sorted/combine_curr/deepmath_sel_hidden_diff_sae_batch_v4mix8.pt"