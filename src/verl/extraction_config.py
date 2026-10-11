from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf, open_dict


def prepare_static_gradient_extraction(config) -> None:
    extraction = config.get("static_gradient_extraction", {})
    if not extraction.get("enable", False):
        return
    if str(config.algorithm.adv_estimator).lower() != "grpo":
        raise ValueError("static_gradient_extraction currently requires algorithm.adv_estimator=grpo")
    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise ValueError("Static gradient-SAE extraction currently requires actor.strategy=fsdp")

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

    with open_dict(config):
        config.static_gradient_extraction.run_dir = str(run_dir)
        config.static_gradient_extraction.resume_from_step = resume_from_step
        config.static_gradient_extraction.num_rollouts = int(config.actor_rollout_ref.rollout.n)
        config.actor_rollout_ref.static_gradient_extraction = OmegaConf.create(
            OmegaConf.to_container(config.static_gradient_extraction, resolve=True)
        )

        config.critic.enable = False
        config.actor_rollout_ref.actor.ppo_epochs = 1
        config.actor_rollout_ref.actor.shuffle = False
        config.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 1
        config.trainer.balance_batch = False
        config.trainer.total_epochs = 1
        config.trainer.critic_warmup = 0
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.save_freq = -1
        config.trainer.resume_mode = "disable"
        config.trainer.use_legacy_worker_impl = "enable"
        config.data.shuffle = False

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
