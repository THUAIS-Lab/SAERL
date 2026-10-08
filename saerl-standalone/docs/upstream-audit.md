# Upstream audit and extension boundaries

Source: `origin/dev` at `a7a707729b39dbed8c6126ace675bd0ba60d4e2a`.
Initial snapshot: `be4da3e`. Official baseline:
[verl 875788ef6e218d0b98cb64fff919d9fef014f986](https://github.com/verl-project/verl/commit/875788ef6e218d0b98cb64fff919d9fef014f986), version `0.7.0.dev` (normalized `0.7.0.dev0`).

We compared the official archive's Git blob hashes with every file in the initial snapshot:
**1175 matches, zero changed/local-only files**, one upstream-only `.vscode/settings.json`.
Machine-readable evidence: `upstream-audit.json`. Core package fingerprints are shipped in
`saerl.integrations.verl/upstream.json` and checked before launching, again in the Ray TaskRunner and extraction worker.

## Changes against this exact baseline

| Original module | Changes in dev | Standalone implementation |
| --- | --- | --- |
| `trainer/main_ppo.py` | 71 added, 1 removed | Preparation + TaskRunner subclass injected through official `run_ppo` |
| `trainer/ppo/ray_trainer.py` | 83 added, 22 removed | Extraction-only Trainer; ordinary training uses upstream Trainer |
| `workers/fsdp_workers.py` | 5 added, 1 removed | Async worker subclass; backward capture setup; scheduler never stepped |
| `workers/actor/dp_actor.py` | 137 added, 6 removed | Actor subclass; original forward reused; adapted backward consumption |
| `workers/actor/static_gradient_sae.py` | 727 added | Independent representations module; equivalent Torch TopK replaces OpenSAE import |
| `trainer/ppo/saerl_sampler.py` | 14 added | Fixed-order sampler passed through official Trainer constructor |
| `utils/reward_score/__init__.py` | 9 added, 1 removed | Custom scoring function through `pkg://saerl.rewards.dispatch` |
| `utils/reward_score/deepmath.py` | 36 added | Same DeepMath scorer and fallback timeout |
| Two trainer YAMLs | 41 added | Package config composes official YAMLs through Hydra searchpath |

## Methods retained locally and why

All adapted methods originate from the official baseline above plus the audited SAERL dev changes;
copyright headers are retained. No official module or class is replaced globally.

- `SAERLTaskRunner.run`: this version accepts a TaskRunner class, but hardcodes `RayPPOTrainer` inside `run`;
  override orchestration to select the extraction trainer, while reusing upstream dataset/reward/resource helpers.
- `StaticGradientTrainer._create_dataloader`: upstream hardcodes `drop_last=True`; extraction must include the tail
  and calculate optimizer/trainer total steps from the resulting dataloader length.
- `StaticGradientTrainer.fit`: upstream exposes no hook between prompt batching, DP padding, log-prob processing
  and advantage computation. Preserve the dev loop with tail flags, step resume, and extraction log-prob behavior.
- `StaticGradientTrainer._get_gen_batch`: preserve stable ID and padding flags across rollout metadata transfer.
- `StaticGradientTrainer._update_actor`: attach current step, then delegate to the upstream method.
- `StaticGradientActor.update_policy`: upstream exposes no callback immediately after each microbatch's backward.
  Preserve dev loss calculation/scaling, consume the layer-output gradient, clear parameter gradients, and skip stepping.
  Hook helpers are SAERL additions; `_forward_micro_batch` delegates directly to upstream.
- `StaticGradientWorker.init_model`: call upstream initialization, then construct the specialized actor over the exact
  same model/config/optimizer. The checkpoint manager keeps the same underlying optimizer/model objects.
- `StaticGradientWorker.update_actor`: preserve upstream dispatch, profiling, sharding and offload lifecycle; omit
  LR scheduler stepping. Async rollout methods are inherited from the official async worker.

## Behavior preserved

Loss includes configured entropy/KL terms. AMP scaling is undone before projection, while ordinary microbatch loss
scaling is preserved. Token alignment, BF16 decoder projection, runtime decoder row L2 normalization, absolute TopK
ranking with signed values, positive signed-peak tie breaking, all-token mean denominator and equal-weight rollout
averaging retain the existing definitions. Rank files and representation schema remain unchanged.

Sample IDs come from `index`/`extra_info.index` by default. The fixed-order sampler consumes dataset positions;
export matches stable IDs and retains original source row order and Arrow metadata.

Old plotting/analysis, baselines, evaluation and data-cleaning files remain in the outer comparison repository.
Historical imports through `recipe.*`, manual `sys.path` injection, and nonexistent `grad_baseline` paths were removed
from the migrated core algorithms. Only the old project needs its external OpenSAE editable installation.

The original environment used external editable paths for both verl and OpenSAE. The validation environment installs
the unmodified official archive into a separate venv; `inspected-environment.json` records observed package versions,
not a clean resolver lock or proof of GPU compatibility. See `validation.md` for actual checks and outstanding GPU checks.
