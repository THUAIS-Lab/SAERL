#!/usr/bin/env python3
"""Summarize checkpoint metrics across training seeds (requires numpy, scipy).

Uses per-dataset peaks at step <= 900 from Markdown summaries. AIME uses
pass128; the other five datasets use mean128. Unsuffixed rows map to seed 42.
Outputs use percentage points. Requires numpy and scipy.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import t


def summarize(values):
    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half)


def difference(a, b, paired):
    """SAERL minus baseline; Welch by default, paired only by design."""
    a, b = np.asarray(a), np.asarray(b)
    if paired:
        return summarize(b-a)
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    se = np.sqrt(va+vb)
    df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if se else float('inf')
    half = float(t.ppf(.975, df)*se) if se else 0.0
    mean = float(b.mean()-a.mean())
    return dict(mean=mean, ci95_low=mean-half, ci95_high=mean+half)


def read_peak_table(path):
    """Read full-precision peaks directly from the supplied Markdown table."""
    text = path.read_text()
    if not re.search(r'max step\s*≤\s*900', text):
        raise ValueError(f'{path}: expected max step ≤ 900 metadata')
    header, rows = None, {}
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells[0] == 'Method':
            header = cells
            continue
        if all(re.fullmatch(r':?-+:?', c) for c in cells):
            continue
        if header is None or len(cells) != len(header):
            raise ValueError(f'{path}: malformed table row: {line}')
        if cells[0] in rows:
            raise ValueError(f'{path}: duplicate method {cells[0]}')
        values = [float(v) for v in cells[1:]]
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError(f'{path}: invalid accuracy')
        rows[cells[0]] = dict(zip(header[1:], values))
    if not rows:
        raise ValueError(f'{path}: empty table')
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'final_results/Qwen2.5-Math-1.5B'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mean-table', type=Path, default=base/'mean128/acc_mean128_before_900.md')
    p.add_argument('--pass-table', type=Path, default=base/'pass128/acc_pass128_before_900.md')
    p.add_argument('--saerl-tag', default='sel_v1_plus_silent_v4mix8')
    p.add_argument('--baseline-tag', default='grpo')
    p.add_argument('--runs-json', type=Path, help='Explicit complete mapping: {method: {seed: exact Markdown row name}}; overrides tag naming')
    p.add_argument('--seeds', nargs='+', type=int, default=[0,42,999])
    p.add_argument('--unsuffixed-seed', type=int, default=42)
    p.add_argument('--paired', action='store_true', help='Use only for an explicitly matched experimental design')
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'peak_before900')
    args = p.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        p.error('Need at least two distinct seeds')
    args.datasets = ['aime','amc','gsm8k','math','minerva','olympiad']
    tables = {'mean128': read_peak_table(args.mean_table),
              'pass128': read_peak_table(args.pass_table)}
    runs = {
        method: {str(seed): tag + ('' if seed == args.unsuffixed_seed else f'_seed{seed}')
                 for seed in args.seeds}
        for method, tag in [('GRPO', args.baseline_tag), ('SAERL', args.saerl_tag)]}
    if args.runs_json:
        try:
            runs = json.loads(args.runs_json.read_text())
        except (OSError, ValueError) as exc:
            p.error(f'Cannot read --runs-json: {exc}')
    if not isinstance(runs, dict) or set(runs) != {'GRPO', 'SAERL'}:
        p.error('Run mapping must contain exactly GRPO and SAERL')
    selected = []
    for method, seed_runs in runs.items():
        if not isinstance(seed_runs, dict) or set(seed_runs) != {str(s) for s in args.seeds}:
            p.error(f'{method}: run mapping must contain exactly the --seeds keys')
        for seed, name in seed_runs.items():
            if not isinstance(name, str) or not name.strip():
                p.error(f'{method} seed {seed}: run name must be a nonempty string')
            if name in selected:
                p.error(f'Run {name!r} is assigned more than once')
            selected.append(name)
            for metric, table in tables.items():
                if name not in table:
                    p.error(f'{method} seed {seed}: run {name!r} missing from {metric} table')
    scores, raw = {}, []
    for method, tag in [('GRPO', args.baseline_tag), ('SAERL', args.saerl_tag)]:
        scores[method] = {}
        for dataset in args.datasets:
            metric = 'pass128' if dataset == 'aime' else 'mean128'
            path = args.pass_table if dataset == 'aime' else args.mean_table
            values = []
            for seed in args.seeds:
                row_name = runs[method][str(seed)]
                value = 100 * tables[metric][row_name][dataset]
                values.append(value)
                raw.append(dict(method=method,dataset=dataset,metric=metric,seed=seed,
                                selection='per-dataset peak at step <= 900',
                                accuracy_pct=value,source=str(path.resolve()),source_row=row_name))
            scores[method][dataset] = values
        scores[method]['Macro-average'] = np.mean(
            [scores[method][d] for d in args.datasets], axis=0).tolist()
    summary = []
    lines = ['# Per-dataset peaks at step ≤ 900\n\nAIME: pass128; other datasets: mean128.\n', '| Dataset | GRPO mean ± SD | SAERL mean ± SD | Δ (pp) | Δ 95% CI |',
             '|---|---:|---:|---:|---:|']
    for dataset in args.datasets + ['Macro-average']:
        a, b = scores['GRPO'][dataset], scores['SAERL'][dataset]
        sa, sb, delta = summarize(a), summarize(b), difference(a,b,args.paired)
        summary.append(dict(dataset=dataset,GRPO=sa,SAERL=sb,delta=delta,
                            seed_scores={'GRPO':a,'SAERL':b}))
        lines.append(f"| {dataset} | {sa['mean']:.2f} ± {sa['sd']:.2f} | {sb['mean']:.2f} ± {sb['sd']:.2f} | {delta['mean']:+.2f} | [{delta['ci95_low']:.2f}, {delta['ci95_high']:.2f}] |")
    notes = (f"\nSeeds: {args.seeds}. Unsuffixed seed: {args.unsuffixed_seed}. "
             f"Difference CI: {'paired Student t' if args.paired else 'Welch t'}. "
             "Method CIs: Student t. All CIs are nominal, pointwise 95%, assume independent "
             "approximately normal run scores (paired: run differences), and are unstable with three runs. "
             "No evaluation-sample bootstrap is computed from aggregate counts. "
             "Macro-average weights datasets equally within each training run. "
             "Each dataset may peak at a different checkpoint; this is an average of peaks, "
             "not the performance of one checkpoint. Test-set peak selection can be optimistic; "
             "these intervals do not correct selection bias. "
             "Metric, source table and source row are in per_seed.csv.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'per_seed.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    (args.output_dir/'summary.json').write_text(json.dumps(dict(
        seeds=args.seeds,runs=runs,selection='per-dataset peak at step <= 900',
        metrics={d: 'pass128' if d == 'aime' else 'mean128' for d in args.datasets},
        paired=args.paired,notes=notes,results=summary),indent=2)+'\n')
    run_lines = ['\n## Selected runs\n', '| Method | Seed | Run |', '|---|---:|---|']
    for method in ['GRPO', 'SAERL']:
        for seed in args.seeds:
            run_lines.append(f'| {method} | {seed} | {runs[method][str(seed)]} |')
    report = '\n'.join(lines)+'\n'+notes+'\n'.join(run_lines)+'\n'
    (args.output_dir/'table.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
