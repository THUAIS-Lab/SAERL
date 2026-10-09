# Validation record

Validation used Python 3.12 in a temporary venv, with the official verl archive and SAERL wheel installed
as ordinary distributions. Third-party packages were shared from the inspected environment via
`--system-site-packages`; no existing global/editable installations were changed.
The working directory was outside both source checkouts, with no project `PYTHONPATH`.

Passed:

- Full official baseline audit: 1175 initial snapshot files matched Git blob hashes.
- Wheel and sdist builds; wheel contains only SAERL and its metadata/config resources, no verl/recipe/OpenSAE package.
- Official dependency fingerprints and adapter imports; both `saerl.__file__` and `verl.__file__` point to the temporary
  venv's installed packages.
- `python -m saerl.train --cfg job` and `python -m saerl.extract --cfg job`, composing packaged SAERL overlays with
  installed official verl YAMLs.
- Ruff and **13 pytest cases**, including fixed-input actor gradient/loss comparison with the official actor under
  static and dynamic microbatching, entropy/KL terms, unchanged model parameters, no optimizer/scheduler steps,
  dispatch metadata, fixed-position sampler, tail dataloader, extraction setup/resume, signed pooling/chunking,
  AMP unscaling, padding exclusion and missing-rank output recovery.
- CPU feature selection → selection → subset export, preserving stable IDs, source row order and Arrow metadata;
  duplicate/missing ID and overwrite rejection.
- Old and installed selection on the same four-row fixture: selected/silent IDs and the entire selection audit table
  (including representatives and cosine values) are identical.
- Installed CPU selection with imports of torch, verl, vLLM and OpenSAE explicitly blocked.

Limitations:

- CUDA is unavailable on this execution host (`nvidia-smi` cannot communicate with the driver, also checked outside
  the sandbox). Real FSDP backward,
  remove-padding/checkpointing under CUDA, two-GPU rollout, and training/checkpoint restore have not been executed.
- `scripts/gpu_smoke.py` exercises a four-prompt, two-GPU tail extraction, partial-rank resume, selection/export,
  one GRPO step and checkpoint save. Use a short, diverse fixture that produces nonzero advantages.
- Full dependency resolution in a newly provisioned CUDA environment has not been validated. Observed version
  constraints are provided separately from the source environment inventory.
- The sandbox denies network sockets. Outside the sandbox a local Ray instance starts, but importing the training
  adapters in its child process did not finish within the 150-second process limit. This check remains unverified;
  retry `scripts/check_ray_imports.py` on the actual training node. Driver-process adapter imports passed.

Reproduction:

```bash
python -m build --no-isolation
ruff check src tests scripts
# Run in the environment where the official pinned dependency and SAERL wheel are installed.
python -m pytest tests -q
python scripts/compare_legacy_selection.py --legacy-root /path/to/old-checkout \
  --run-dir /path/to/extraction --feature-indices /path/to/features.npy \
  --output-dir /path/to/new-comparison --cosine-threshold 0.7
```

Remaining GPU validation is not implied by the CPU tests or the existence of the smoke script.

## Repository root cleanup — 2026-10-09

SAERL is now the project at the repository root. The old verl source, recipes, experimental results,
baselines, upstream packaging and CI have been removed from this branch's file tree; the audited
standalone source was preserved. Original code remains available on `origin/dev`.

Rechecked after the move:

- Wheel and sdist builds from the repository root; the wheel has 53 files, all under `saerl/`
  or its distribution metadata. SAERL configs and the official dependency fingerprints are included.
- The new wheel installed without dependencies into the temporary validation venv. Both SAERL and
  the pinned official verl load from installed `site-packages`, without a local verl directory.
- **13 tests passed, 7 warnings, in 199.65 seconds**, including the official dependency integration tests.
  A shared third-party multiprocess resource-tracker destructor also logged an ignored exception after
  pytest completed with exit code 0.
- README extraction and GRPO commands passed installed CLI config composition and the official
  `validate_config` checks. The commands include the required rollout log-prob micro-batch setting;
  the same missing setting was added to the GPU smoke script.
- Ruff passed for `src`, `tests` and `scripts`.

The root cleanup does not change the GPU and Ray validation limitations above.
