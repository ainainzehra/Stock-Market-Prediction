"""
Balances the preprocessed_data.parquet to an exact 50/50 class split by
undersampling the majority class (Down = 0).

Current distribution:  Down ~68%  |  Up ~32%
After balancing:       Down  50%  |  Up  50%
"""
import pandas as pd
import numpy as np

RANDOM_SEED = 42
INPUT_PATH  = "datasets/preprocessed_data.parquet"
OUTPUT_PATH = "datasets/balanced_preprocessed_data.parquet"

print("Loading preprocessed data...")
df = pd.read_parquet(INPUT_PATH)

print(f"  Original shape : {df.shape}")
print(f"  Target distribution (before):")
orig_counts = df['Target'].value_counts().sort_index()
for cls, cnt in orig_counts.items():
    print(f"    Class {cls}: {cnt:>10,}  ({cnt/len(df)*100:.2f}%)")

df_minority = df[df['Target'] == 1]   # Up   (~32%)
df_majority = df[df['Target'] == 0]   # Down (~68%)

n_minority = len(df_minority)
n_majority = len(df_majority)
print(f"\n  Minority class (Up=1)  : {n_minority:,}")
print(f"  Majority class (Down=0): {n_majority:,}")
print(f"  Undersampling majority to {n_minority:,} rows...")

df_majority_sampled = df_majority.sample(n=n_minority, random_state=RANDOM_SEED)

df_balanced = pd.concat([df_minority, df_majority_sampled], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\n  Balanced shape : {df_balanced.shape}")
print(f"  Target distribution (after):")
new_counts = df_balanced['Target'].value_counts().sort_index()
for cls, cnt in new_counts.items():
    print(f"    Class {cls}: {cnt:>10,}  ({cnt/len(df_balanced)*100:.2f}%)")

assert new_counts[0] == new_counts[1], "Classes are not equal — something went wrong!"
print("\n  Class balance verified: exactly 50/50")

print(f"\nSaving to {OUTPUT_PATH}...")
df_balanced.to_parquet(OUTPUT_PATH, index=False)

size_mb = __import__('os').path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Done! Saved {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print(f"  Total rows: {len(df_balanced):,}")