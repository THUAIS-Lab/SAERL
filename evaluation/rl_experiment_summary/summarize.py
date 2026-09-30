"""Rebuild inventory and all-run peak tables from final_results (stdlib only)."""
from pathlib import Path
import csv
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1] / 'final_results'
OUT = Path(__file__).resolve().parent
DATASETS = ['aime', 'amc', 'gsm8k', 'math', 'minerva', 'olympiad']
rows = []
curves = {}
for path in sorted(ROOT.rglob('*.json')):
    if path.parent.name == 'avg128':
        continue  # Derived averages, not additional experiments.
    dataset, run = path.stem.split('_', 1)
    rel = path.relative_to(ROOT)
    model = rel.parts[0]
    setting = 'math8k128' if 'math8k_limr' in rel.parts else ('deepmath512' if path.parent.name.endswith('512') else 'deepmath128')
    metric = 'pass' if path.parent.name.startswith('pass') else 'mean'
    cap = 900 if setting == 'deepmath128' else 300
    records = json.loads(path.read_text())
    points = [(int(re.search(r'global_step_(\d+)', x['model_path'])[1]), x['accuracy']) for x in records]
    assert len(points) == len(dict(points)), f'Duplicate steps: {path}'
    assert all(abs(x['accuracy'] - x['correct']/x['total']) < 1e-10 for x in records), path
    eligible = [(s, a) for s, a in points if s <= cap]
    peak_step, peak = min(eligible, key=lambda x: (-x[1], x[0]))
    row = dict(model=model, setting=setting, run=run, dataset=dataset, metric=metric,
               cap=cap, records=len(points), min_step=min(dict(points)), max_step=max(dict(points)),
               checkpoints_in_window=len(eligible), peak_step=peak_step, peak_accuracy=peak,
               accuracy_at_cap=dict(points).get(cap, ''), source=str(rel),
               checkpoint_parent=records[0]['model_path'].split('/global_step_')[0])
    rows.append(row)
    curves[model, setting, run, dataset, metric] = dict(points)
with (OUT/'inventory.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
index = {(r['model'],r['setting'],r['run'],r['dataset'],r['metric']):r for r in rows}
groups = sorted({(r['model'], r['setting']) for r in rows})
lines = ['# 全部已归档 RL 运行：原始 JSON 重算', '',
         '分数为百分数。AIME 列使用 pass（代码为每题 8 次中至少一次正确），其余五列使用 mean。', '',
         'Mixed Avg 为上述六项峰值的等权平均；Mean Avg 为六集 mean 峰值的等权平均。各峰值可来自不同 checkpoint。', '',
         'DeepMath128 窗口 ≤900；DeepMath512、Math8k 窗口 ≤300。NA 不参与凑齐六集均值。run 后缀不推断为 seed。', '']
for model, setting in groups:
    runs = sorted({r['run'] for r in rows if (r['model'],r['setting']) == (model,setting)})
    lines += [f'## {model} / {setting}（{len(runs)} 个运行标签）', '',
              '| Run | AIME pass | AMC mean | GSM8K mean | MATH mean | Minerva mean | Olympiad mean | Mixed Avg | Mean Avg | 六集共同 step 的 mean 宏均值峰值（step） | 归档边界 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for run in runs:
        mixed, means, notes = [], [], []
        for ds in DATASETS:
            key = (model,setting,run,ds)
            m = index.get((*key, 'pass' if ds == 'aime' else 'mean'))
            a = index.get((*key, 'mean'))
            mixed.append(m['peak_accuracy']*100 if m else None)
            means.append(a['peak_accuracy']*100 if a else None)
            if not a: notes.append(ds+' 缺失')
            elif a['max_step'] < a['cap']: notes.append(f"{ds}仅至{a['max_step']}")
        avg = lambda xs: f'{statistics.mean(xs):.2f}' if all(x is not None for x in xs) else 'NA'
        fmt = lambda x: f'{x:.2f}' if x is not None else 'NA'
        cs = [curves.get((model,setting,run,ds,'mean'),{}) for ds in DATASETS]
        common = set.intersection(*(set(c) for c in cs))
        cap = 900 if setting == 'deepmath128' else 300
        common = [s for s in common if s <= cap]
        shared = 'NA'
        if common:
            s = min(common, key=lambda s: (-statistics.mean(c[s] for c in cs),s))
            shared = f'{statistics.mean(c[s] for c in cs)*100:.2f} ({s})'
        lines.append('| '+' | '.join([run,*map(fmt,mixed),avg(mixed),avg(means),shared,'；'.join(notes) or '窗口覆盖完整'])+' |')
    lines.append('')
(OUT/'all_runs.md').write_text('\n'.join(lines)+'\n')
print(f'{len(rows)} source JSONs; {len(set((r["model"],r["setting"],r["run"]) for r in rows))} run labels; inventory.csv and all_runs.md generated.')
