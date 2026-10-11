

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    trainer.use_legacy_worker_impl=enable \
    static_gradient_extraction.enable=True \
    static_gradient_extraction.gradient_token_scope=attention \
    static_gradient_extraction.resume_from_step=743 \
    "$@"
