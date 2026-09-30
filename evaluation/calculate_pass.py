"""generate results from eval_*.json"""
import json
import glob
import os
import re

def extract_step_from_path(model_path: str):
    m = re.search(r"deepmath(\d+)", model_path)
    return int(m.group(1))

def print_existing_files(jsons):
    print(len([p for p in jsons if 'math' in os.path.basename(p)]))
    print(len([p for p in jsons if 'amc' in os.path.basename(p)]))
    print(len([p for p in jsons if 'aime' in os.path.basename(p)]))
    print(len([p for p in jsons if 'minerva' in os.path.basename(p)]))
    print(len([p for p in jsons if 'olympiad' in os.path.basename(p)]))
    print(len([p for p in jsons if 'gsm8k' in os.path.basename(p)]))

def combine_results(result_dir, type, out_dir):
    assert 'pass' in out_dir, "please check the output dir, it should contain 'pass'!"
    result_jsons = glob.glob(os.path.join(result_dir, '*.json'))
    result_jsons = sorted(result_jsons)
    print_existing_files(result_jsons)

    datasets = ["aime", "amc", "gsm8k", "math", "minerva", "olympiad"]
    dataset_to_total = {
        'aime': 240, 
        'amc': 320, 
        'gsm8k': 10552, 
        'math': 4000, 
        'minerva': 2176, 
        'olympiad': 5384
    }

    results = dict()
    for dataset in datasets:
        results[dataset] = []

    for p in result_jsons:
        dataset = os.path.basename(p).split('_')[0]
        assert dataset in datasets, "false dataset name, please check!"
        with open(p, 'r') as f:
            data = json.load(f)
        metrics = data['metrics']
        assert metrics['total'] == dataset_to_total[dataset], f"{metrics['total']}, {dataset_to_total[dataset]}"
        assert type in metrics['model_path']
        
        responses = data['results']
        assert len(responses) == metrics['total'], f"{len(responses)}, {metrics['total']}"
        correct_qs = set()
        pass_num = 0
        for r in responses:
            if int(r['correct']) == 1 and r['question'][0]['content'] not in correct_qs:
                pass_num += 1
                correct_qs.add(r['question'][0]['content'])
        best_metrics = {
            "accuracy": pass_num / (metrics['total'] // 8),
            "correct": pass_num,
            "total": metrics['total'] // 8,
            "model_path": metrics['model_path'],
            "timestamp": metrics['timestamp']
        }
        results[dataset].append(best_metrics)



    for dataset in datasets:
        if results[dataset] != []:
            fname = os.path.join(out_dir, f"{dataset}_{type}.json")
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            with open(fname, "w") as fp:
                json.dump(results[dataset], fp, indent=2)
            print(f"Results saved to {fname}")

if __name__ == "__main__":
    path = "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/"
    out_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/Qwen2.5-Math-1.5B/pass128"
    assert 'pass' in out_dir, "please check the output dir, it should contain 'pass'!"
    all_result_dirs = glob.glob(os.path.join(path, '*/'))
    ###############################################
    """for specific run"""
    run_name = 'irds'
    all_result_dirs = [p for p in all_result_dirs if run_name in p]
    print(all_result_dirs)
    ###############################################
    for result_dir in all_result_dirs:
        type = result_dir.split('/')[-2]
        os.makedirs(out_dir, exist_ok=True)
        print(result_dir, type, out_dir)
        combine_results(result_dir, type, out_dir)