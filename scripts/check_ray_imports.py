"""Check the installed SAERL and official verl from a real CPU Ray child process."""

import argparse
import tempfile

import ray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--temp-dir", default=tempfile.gettempdir())
    args = parser.parse_args()
    # Keep the base path short: Ray's AF_UNIX socket filename has a 107-byte limit.
    root = tempfile.mkdtemp(prefix="sr", dir=args.temp_dir)
    ray.init(num_cpus=1, include_dashboard=False, _temp_dir=root, _node_ip_address="127.0.0.1")
    try:

        @ray.remote
        def probe():
            import saerl
            from saerl.integrations.verl.compat import verify_verl
            from saerl.integrations.verl.runner import SAERLTaskRunner
            from saerl.integrations.verl.worker import StaticGradientWorker

            return str(verify_verl()), saerl.__file__, SAERLTaskRunner.__module__, StaticGradientWorker.__module__

        print(ray.get(probe.remote(), timeout=args.timeout))
        print("PASS: Ray child imports the installed SAERL package and pinned official verl")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
