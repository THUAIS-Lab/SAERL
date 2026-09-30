import os
import json
import torch

OUTPUT_PATH = "/WORK/PUBLIC/lijuanzi1_work/dz/data/rank_sorted_deepmath/deepmath_difficulty_mean128.pt" # 输出文件路径
result_dir = "/WORK/PUBLIC/lijuanzi1_work/dz/result/deepmath_difficulty_proxy_128/"
files = os.listdir(result_dir)
files = sorted(files)
assert len(files) == 103022, f"Expected 103022 files, but found {len(files)}"
for i in range(103022):
    assert f"{i:06d}.json" in files, f"Missing file: {i:06d}.json"

results = []
for file in files:
    with open(os.path.join(result_dir, file), 'r') as f:
        data = json.load(f)

    assert len(data) == 128, f"Expected 128 records in each file, but found {len(data)} in {file}"
    assert len(set([item['index'] for item in data])) == 1, f"Expected all records in {file} to have the same index, but found multiple indices"
    assert len(set([item['question'][0]['content'] for item in data])) == 1, f"Expected all records in {file} to have the same question, but found multiple questions"
    assert len(set([item['ground_truth'] for item in data])) == 1, f"Expected all records in {file} to have the same ground truth, but found multiple ground truths"
    assert f"{data[0]['index']:06d}.json" == file, f"Expected index in data to match filename, but found mismatch in {file}"

    correct_count = sum([item['correct'] for item in data])
    accuracy = correct_count / len(data)
    results.append({
        "index": data[0]['index'],
        "accuracy": accuracy
    })

# accuracy from high to low, aka difficulty from easy to hard
sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

final_indices = [item['index'] for item in sorted_results]
final_accs = [item['accuracy'] for item in sorted_results]

save_data = {
    "indices": final_indices,              # List[int]
    "accuracy": final_accs                 # List[float]
}
torch.save(save_data, OUTPUT_PATH)
print(f"Saved {len(final_indices)} records to {OUTPUT_PATH}")
print("Top 5 indices (Easiest):", final_indices[:5], final_accs[:5])
print("Bottom 5 indices (Hardest):", final_indices[-5:], final_accs[-5:])