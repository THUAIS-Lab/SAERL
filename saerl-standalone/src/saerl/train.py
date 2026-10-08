"""Hydra training entry point using an external, pinned official verl."""


def launch(config_name):
    from saerl.integrations.verl.compat import verify_verl

    verify_verl()
    import hydra

    @hydra.main(config_path="pkg://saerl.config", config_name=config_name, version_base=None)
    def run(config):
        import ray
        from verl.trainer.main_ppo import run_ppo
        from verl.utils.device import auto_set_ascend_device_name

        from saerl.integrations.verl.runner import SAERLTaskRunner

        auto_set_ascend_device_name(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SAERLTaskRunner))

    run()


def main():
    launch("train")


if __name__ == "__main__":
    main()
