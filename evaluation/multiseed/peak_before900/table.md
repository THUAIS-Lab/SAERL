# Per-dataset peaks at step ≤ 900

AIME: pass128; other datasets: mean128.

| Dataset | GRPO mean ± SD | SAERL mean ± SD | Δ (pp) | Δ 95% CI |
|---|---:|---:|---:|---:|
| aime | 31.11 ± 1.92 | 36.67 ± 3.33 | +5.56 | [-1.27, 12.38] |
| amc | 53.65 ± 0.65 | 55.31 ± 0.94 | +1.67 | [-0.25, 3.59] |
| gsm8k | 81.34 ± 0.79 | 85.84 ± 0.21 | +4.50 | [2.70, 6.31] |
| math | 71.82 ± 0.59 | 72.75 ± 0.38 | +0.93 | [-0.28, 2.15] |
| minerva | 26.16 ± 0.86 | 28.48 ± 0.55 | +2.31 | [0.57, 4.06] |
| olympiad | 34.96 ± 0.43 | 36.00 ± 0.49 | +1.04 | [-0.01, 2.09] |
| Macro-average | 49.84 ± 0.31 | 52.51 ± 0.69 | +2.67 | [1.21, 4.13] |

Seeds: [0, 42, 999]. Unsuffixed seed: 42. Difference CI: Welch t. Method CIs: Student t. All CIs are nominal, pointwise 95%, assume independent approximately normal run scores (paired: run differences), and are unstable with three runs. No evaluation-sample bootstrap is computed from aggregate counts. Macro-average weights datasets equally within each training run. Each dataset may peak at a different checkpoint; this is an average of peaks, not the performance of one checkpoint. Test-set peak selection can be optimistic; these intervals do not correct selection bias. Metric, source table and source row are in per_seed.csv.
