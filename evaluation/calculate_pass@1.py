"""generate results from eval_*.json"""
import json
import glob
import os

##################Config##################
result_dir = "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/sorted_deepmath128/grpo/"

type = "grpo"
out_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/qwen7b/pass128"
##########################################

def print_existing_files(jsons):
    print(len([p for p in jsons if 'math' in os.path.basename(p)]))
    print(len([p for p in jsons if 'amc' in os.path.basename(p)]))
    print(len([p for p in jsons if 'aime' in os.path.basename(p)]))
    print(len([p for p in jsons if 'minerva' in os.path.basename(p)]))
    print(len([p for p in jsons if 'olympiad' in os.path.basename(p)]))
    print(len([p for p in jsons if 'gsm8k' in os.path.basename(p)]))

if __name__ == "__main__":
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
        qs = set()
        pass_num = 0
        for r in responses:
            if r['question'][0]['content'] not in qs:
                pass_num += int(r['correct'])
                qs.add(r['question'][0]['content'])
        pass_metrics = {
            "accuracy": pass_num / (metrics['total'] // 8),
            "correct": pass_num,
            "total": metrics['total'] // 8,
            "model_path": metrics['model_path'],
            "timestamp": metrics['timestamp']
        }
        results[dataset].append(pass_metrics)



    for dataset in datasets:
        if results[dataset] != []:
            fname = os.path.join(out_dir, f"{dataset}_{type}.json")
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            with open(fname, "w") as fp:
                json.dump(results[dataset], fp, indent=2)
            print(f"Results saved to {fname}")