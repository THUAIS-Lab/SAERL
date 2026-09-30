from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(
    "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train_difficulty_replaced_100_mean128.parquet"
)
OUTPUT_PATH = Path(__file__).with_name("difficulty.png")
BATCH_OUTPUT_PATH = Path(__file__).with_name("difficulty_batch_mean.png")
BATCH_SIZE = 128


data = pd.read_parquet(DATA_PATH, columns=["extra_info"])
difficulty = data["extra_info"].map(lambda info: info["difficulty"])

plt.figure(figsize=(12, 4))
plt.plot(difficulty.index, difficulty, linewidth=0.8)
plt.xlabel("Index")
plt.ylabel("Difficulty")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)

batch_mean = difficulty.groupby(difficulty.index // BATCH_SIZE).mean()

plt.figure(figsize=(12, 4))
plt.plot(batch_mean.index, batch_mean, linewidth=0.8)
plt.xlabel("Batch Index")
plt.ylabel("Mean Difficulty")
plt.tight_layout()
plt.savefig(BATCH_OUTPUT_PATH, dpi=200)

print(f"Saved to {OUTPUT_PATH}")
print(f"Saved to {BATCH_OUTPUT_PATH}")
