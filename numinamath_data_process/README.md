# DeepMath-103K Cleaning Reproduction

This folder contains a practical reimplementation of the data curation process described in:

- Paper: DeepMath-103K (arXiv:2504.11456)
- Paper link: https://arxiv.org/abs/2504.11456
- Official repo: https://github.com/zwhe99/DeepMath
- Dataset: https://huggingface.co/datasets/zwhe99/DeepMath-103K

## Why this folder exists

The official repository publicly provides training/evaluation code and dataset release, but does not expose a complete end-to-end script for the paper's Section 3 curation pipeline (source analysis -> decontamination -> difficulty filtering -> answer verification).

This folder provides an open, configurable reproduction.

## What is implemented

Script: `deepmath_cleaning_pipeline.py`

### Source-specific schema adapters

The script now has explicit adapters for the three source datasets mentioned in the paper:

1. `Vivacem/MMIQC`
- Input columns used: `instruction`, `output`, `source`
- Standardized fields:
  - `question <- instruction`
  - `answer <- output`
  - `source <- source`

2. `TIGER-Lab/WebInstructSub`
- Input columns used: `orig_question`, `orig_answer`, `question`, `answer`, `source`, `index`
- Standardized fields:
  - `question <- question` (fallback to `orig_question`)
  - `answer <- answer` (fallback to `orig_answer`)
  - `source <- source`
  - keep `index` for traceability

3. `AI-MO/NuminaMath-CoT`
- Input columns used: `source`, `problem`, `solution`, `messages`
- Standardized fields:
  - `question <- problem`
  - `answer <- solution`
  - `source <- source`

Only cleaning-related columns are retained in stage/final outputs, and they always include:
- `question`
- `answer` (stored as `source_answer`; final normalized answer stored as `final_answer`)
- `source`

Stage mapping to the paper:

1. Stage 1 - Source analysis and collection (practical subset)
- Load multiple sources (HF datasets or local files).
- Apply basic filtering (length/token checks) and exact dedup after normalization.

2. Stage 2 - Data decontamination
- Build benchmark pool from configured benchmark datasets.
- Use embedding retrieval with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- For each candidate, retrieve top-k similar benchmark questions (default k=5).
- Judge duplicate/paraphrase via LLM judge (if configured) or heuristic fallback.
- Drop candidates flagged as contaminated.

3. Stage 3 - Difficulty filtering
- Prefer existing `difficulty` field if present.
- Otherwise score with LLM for multiple rounds (default 6 votes) and average.
- Fallback to heuristic difficulty if no LLM is configured.
- Keep only samples with difficulty >= threshold (default 5.0).

4. Stage 4 - Answer verifiability filtering
- Question filtering/formatting: remove open-ended non-verifiable problems.
- Collect source answer and three R1 solutions (or generate missing ones if enabled).
- Extract final answer strings and normalize.
- Keep sample only when all normalized answers are identical and enough answers are available.

## Related code found in this workspace/repo

These are relevant to dataset usage or answer checking, not full paper curation:

- `examples/data_preprocess/deepmath.py`
- `examples/data_preprocess/deepmath_filtered.py`
- `verl/utils/reward_score/math_dapo.py`

These scripts are mostly for training-format conversion and reward/answer normalization, rather than full dataset construction.

## Quick start

Install dependencies (example):

```bash
pip install datasets sentence-transformers requests numpy
```

Run a small dry-run:

```bash
python data_process/deepmath_cleaning_pipeline.py \
  --source "zwhe99/DeepMath-103K::train" \
  --use-default-benchmarks \
  --skip-missing-benchmarks \
  --max-samples 200 \
  --output-dir data_process/output_dry \
  --save-intermediate
```

Run with source-value allowlists (recommended for paper-like source subset control):

```bash
python data_process/deepmath_cleaning_pipeline.py \
  --source "Vivacem/MMIQC::train" \
  --source "TIGER-Lab/WebInstructSub::train" \
  --source "AI-MO/NuminaMath-CoT::train" \
  --source-value-allowlist "schema:mmiqc=math.stackexchange.com" \
  --source-value-allowlist "schema:webinstructsub=mathstackexchange" \
  --source-value-allowlist "schema:numinamath=math" \
  --use-default-benchmarks \
  --skip-missing-benchmarks \
  --output-dir data_process/output_with_allowlist \
  --save-intermediate
```

You can also provide a JSON allowlist file:

```json
{
  "schema:mmiqc": ["math.stackexchange.com"],
  "schema:webinstructsub": ["mathstackexchange"],
  "schema:numinamath": ["math"]
}
```

Then run with:

```bash
python data_process/deepmath_cleaning_pipeline.py \
  --source "Vivacem/MMIQC::train" \
  --source "TIGER-Lab/WebInstructSub::train" \
  --source "AI-MO/NuminaMath-CoT::train" \
  --source-value-allowlist-file data_process/source_allowlist.json \
  --output-dir data_process/output_with_allowlist_file \
  --save-intermediate
```

Run with LLM judge/scorer (OpenAI-compatible API):

```bash
python data_process/deepmath_cleaning_pipeline.py \
  --source "zwhe99/DeepMath-103K::train" \
  --use-default-benchmarks \
  --skip-missing-benchmarks \
  --llm-base-url "$OPENAI_BASE_URL/v1" \
  --llm-api-key "$OPENAI_API_KEY" \
  --judge-model "meta-llama/Llama-3.3-70B-Instruct" \
  --difficulty-model "gpt-4o" \
  --formatter-model "gpt-4o" \
  --generator-model "deepseek-reasoner" \
  --generate-missing-r1 \
  --output-dir data_process/output_full \
  --save-intermediate
```

Build stage/source retention statistics after a run:

```bash
python data_process/deepmath_cleaning_stats.py \
  --output-dir data_process/output_with_allowlist
```

This writes:
- `source_retention_stats.json`
- `source_retention_report.md`

The report includes:
- Stage pass rates.
- Source-value retention from stage0/stage1/stage2/stage3/final.

## Remove non-verifiable questions only

If you only want to remove non-verifiable questions from:

- `Vivacem/MMIQC`
- `TIGER-Lab/WebInstructSub`
- `AI-MO/NuminaMath-CoT`

use the dedicated script:

```bash
python data_process/filter_unverifiable_questions.py \
  --output-dir data_process/output_verify_only
```

This script defaults to those three datasets and runs a verifiability-only filter:

1. drop open-ended/non-verifiable questions;
2. extract canonical answer from source answer/solution;
3. keep only samples with exactly one normalized answer candidate.

Outputs:

- `*_verify_filtered.jsonl`: kept verifiable samples
- `verify_filter_summary.json`: per-source and global statistics

Optional: save dropped samples with reason codes:

```bash
python data_process/filter_unverifiable_questions.py \
  --output-dir data_process/output_verify_only \
  --save-dropped
```

Optional: run with OpenAI-compatible formatter model for question verifiability:

```bash
python data_process/filter_unverifiable_questions.py \
  --output-dir data_process/output_verify_only_llm \
  --llm-base-url "$OPENAI_BASE_URL/v1" \
  --llm-api-key "$OPENAI_API_KEY" \
  --formatter-model "gpt-4o"
```

## Decontaminate an already-verifiable local dataset

If you already have a local verifiable dataset and now only need decontamination,
use:

Script: `data_process/decontaminate_verifiable_dataset.py`

Example (local candidate + multiple local test sets + local embedding model path):

```bash
python data_process/decontaminate_verifiable_dataset.py \
  --input /path/to/verifiable_dataset.jsonl \
  --benchmark /path/to/testset_math500.jsonl \
  --benchmark /path/to/testset_aime.jsonl \
  --benchmark /path/to/testset_amc.jsonl \
  --embedding-model /path/to/paraphrase-multilingual-MiniLM-L12-v2 \
  --top-k 5 \
  --min-similarity 0.55 \
  --output-dir data_process/output_decontam \
  --save-dropped \
  --attach-metadata
```

Outputs:

- `decontaminated_kept.jsonl`: 去污染后保留样本
- `decontaminated_dropped.jsonl`: 被判定为污染的样本（使用 `--save-dropped` 时）
- `decontam_summary.json`: 统计与参数

Optional: use local Llama Judge for duplicate judgment:

```bash
python data_process/decontaminate_verifiable_dataset.py \
  --input /path/to/verifiable_dataset.jsonl \
  --benchmark /path/to/testset_1.jsonl \
  --embedding-model /path/to/paraphrase-multilingual-MiniLM-L12-v2 \
  --use-llm-judge \
  --judge-model-path /path/to/llama-judge-model \
  --judge-device-map auto \
  --judge-torch-dtype bfloat16 \
  --output-dir data_process/output_decontam_llm
```

## Notes

- This is a reproduction, not an official release from DeepMath authors.
- Full-scale runs can be expensive; start with `--max-samples` for validation.
- If your benchmark IDs differ, pass custom `--benchmark <dataset::split>` entries.
- The stage and final outputs include `clean_columns`, containing only source fields relevant to cleaning and traceability.
- For complete retention analytics, enable `--save-intermediate` so `stage0_loaded.jsonl` to `stage3_difficulty_filtered.jsonl` are available.
