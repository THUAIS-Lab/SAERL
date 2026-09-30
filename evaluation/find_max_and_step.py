"""find max acc before xxx step"""
import os
import json
from pathlib import Path
import re
import sys
import glob
import argparse

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))

def generate_markdown_table(data, max_step, output_path):
    """
    生成 Markdown 表格文件
    data: dict[method][dataset] = max_acc
    """
    datasets = sorted({ds for method_dict in data.values() for ds in method_dict})
    methods = sorted(data.keys())

    if not datasets or not methods:
        print("No data to write to Markdown table.")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        # 文件标题（可选）
        f.write(f"# {output_path.split('/')[-3]}\n\n")
        f.write(f"# {output_path.split('/')[-2]}\n\n")
        if max_step is not None:
            f.write(f"# Accuracy Summary (max step ≤ {max_step})\n\n")
        else:
            f.write(f"# Accuracy Summary (all steps)\n\n")

        # 表头
        header = "| Method | " + " | ".join(datasets) + " |\n"
        f.write(header)

        # 分隔线
        separator = "| --- | " + " | ".join("---" for _ in datasets) + " |\n"
        f.write(separator)

        # 数据行
        for m in methods:
            cells = []
            for ds in datasets:
                acc = data[m].get(ds)
                if acc is not None:
                    # cells.append(f"{acc:.4f}")
                    cells.append(f"{acc}")
                else:
                    cells.append("")   # 缺失留空
            row = "| " + m + " | " + " | ".join(cells) + " |\n"
            f.write(row)

    print(f"Markdown table has been saved to: {output_path}")

def main(folder, max_step=None, output_path=None):
    jsons = glob.glob(os.path.join(folder, '*.json'))
    jsons = sorted(jsons)

    table_data = dict()  # method -> dataset -> max_acc

    for file in jsons:
        with open(file, 'r') as f:
            data = json.load(f)
        
        dataset = os.path.basename(file).split('_')[0]
        method = os.path.basename(file).replace(f'{dataset}_', '').replace('.json', '')
        
        acc = []
        steps = []
        for i in data:
            acc.append(i['accuracy'])
            steps.append(extract_step_from_path(i['model_path']))
        
        max_acc = 0
        step = 0
        
        assert len(acc) == len(steps)
        if max(steps) < max_step:
            if 'dapo' not in method and 'gradalign' not in method:
                raise ValueError(f"Exceed existing steps: {method}: {max(steps)}")
        
        for a, s in zip(acc, steps):
            if max_step is not None and s > max_step:
                continue
            if a > max_acc:
                max_acc = a
                step = s

        table_data.setdefault(method, {})[dataset] = max_acc

        # print(f"=== {dataset} - {method} ===")
        # print(f"Max Accuracy: {max_acc}, Step: {step}")
        # print('\n')
    generate_markdown_table(table_data, max_step, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing JSON files")
    parser.add_argument("--max-step", default=None, type=int, help="Maximum step to consider")
    args = parser.parse_args()
    main(args.folder, args.max_step, os.path.join(args.folder, f"acc_{args.folder.split('/')[-1]}_before_{args.max_step}.md"))
