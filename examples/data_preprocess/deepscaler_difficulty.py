"""transform deepscaler difficulty into rank.pt & train.parquet"""
# run this code, and then run check_length.py, done
import pandas as pd
import torch
import os

###############################Config##################################

#######################################################################

def process_parquet(indices, problems, ground_truths):
    data = []
    split = "train"
    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    for problem, idx, answer in zip(problems, indices, ground_truths):
        question = problem + " " + instruction_following
        data.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx},
        })
    return data

def main():
    p = "/data/daiz/data/lime-nlp_DeepScaleR_Difficulty/Qwen2.5-Math-7B--deepscaler--difficulty.csv"
    df = pd.read_csv(p)
    print(df.columns.tolist())

    indices = df['Unnamed: 0'].tolist()
    assert indices == sorted(indices)
    ground_truth = df['ground_truth'].tolist()
    problem = df['problem'].tolist()
    solved_percentage = df['solved_percentage'].tolist()

    train_dataset = process_parquet(indices=indices, problems=problem, ground_truths=ground_truth)
    train_dataset = pd.DataFrame(train_dataset)
    train_dataset.to_parquet(os.path.join(os.path.dirname(p), "train.parquet"))
    results_list = []
    for idx, acc in zip(indices, solved_percentage):
        results_list.append(
            {
                "index": idx,
                "accuracy": acc
            }
        )
    # accuracy from high to low, aka difficulty from easy to hard
    sorted_results = sorted(results_list, key=lambda x: x['accuracy'], reverse=True)

    final_indices = [item['index'] for item in sorted_results]
    final_accs = [item['accuracy'] for item in sorted_results]

    save_data = {
        "indices": final_indices,              # List[int]
        "accuracy": final_accs                 # List[float]
    }

    OUTPUT_PATH = "/data/daiz/data/lime-nlp_DeepScaleR_Difficulty/rank_deepscaler_difficulty.pt"
    torch.save(save_data, OUTPUT_PATH)
    print(f"Saved {len(final_indices)} records to {OUTPUT_PATH}")
    print("Top 5 indices (Easiest):", final_indices[:5], final_accs[:5])
    print("Bottom 5 indices (Hardest):", final_indices[-5:], final_accs[-5:])

if __name__ == '__main__':
    main()