"""find the min step to target accuracy"""
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

def generate_markdown_table(data, target_accuracy, output_path):
    """
    生成 Markdown 表格文件
    data: dict[method][dataset] = min_step
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
        f.write(f"# {target_accuracy}\n\n")
        

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
                step = data[m].get(ds)
                if step is not None:
                    cells.append(f"{step}")
                else:
                    cells.append("")   # 缺失留空
            row = "| " + m + " | " + " | ".join(cells) + " |\n"
            f.write(row)

    print(f"Markdown table has been saved to: {output_path}")

def main(folder, target_accuracy, output_path):
    jsons = glob.glob(os.path.join(folder, '*.json'))
    jsons = sorted(jsons)

    table_data = dict()  # method -> dataset -> step

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
        
        min_step = 10000
        for a, s in zip(acc, steps):
            if a >= target_accuracy[dataset] and s < min_step:
                min_step = s

        table_data.setdefault(method, {})[dataset] = min_step

    generate_markdown_table(table_data, target_accuracy, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing JSON files")
    args = parser.parse_args()

    if "Qwen2.5-Math-1.5B" in args.folder:
        if "mean128" in args.folder:
            target_accuracy = {
                'aime': 1,
                'amc': 0.521874,
                'gsm8k': 0.811410158,
                'math': 0.71324,
                'minerva': 0.250919117,
                'olympiad': 0.345468052
            }
        elif "pass128" in args.folder:
            target_accuracy = {
                'aime': 0.3,
                'amc': 1,
                'gsm8k': 1,
                'math': 1,
                'minerva': 1,
                'olympiad': 1
            }
        else:
            raise ValueError(f"Unknown folder: {args.folder}")
    elif "Qwen2.5-Math-7B" in args.folder:
        if "mean128" in args.folder:
            target_accuracy = {
                'aime': 1,
                'amc': 0.665625,
                'gsm8k': 0.888172858,
                'math': 0.77525,
                'minerva': 0.328125,
                'olympiad': 0.394316493
            }
        elif "pass128" in args.folder:
            target_accuracy = {
                'aime': 0.466666666,
                'amc': 1,
                'gsm8k': 1,
                'math': 1,
                'minerva': 1,
                'olympiad': 1
            }
        else:
            raise ValueError(f"Unknown folder: {args.folder}")
    else:
        raise ValueError(f"Unknown folder: {args.folder}")
    
    main(args.folder, target_accuracy, os.path.join(args.folder, f"acc_speedup_ablation_{args.folder.split('/')[-1]}.md"))
