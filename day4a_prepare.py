"""
day4a_prepare.py — Day 4, Part A: Prepare multi-checkpoint data
Run: python day4a_prepare.py

What this does:
  Reloads the .mat files ONE more time (last time we need to — everything
  after this uses the cache) to extract features AND Qdlin curves at
  5 checkpoints (10, 30, 50, 70, 100) per cell instead of just 1.

Why: Day 3's LSTM struggled partly because it only saw ONE point in time —
there was no real "sequence" for it to learn from. Giving LSTM/GRU a
genuine 5-step sequence (how features evolve as the battery ages) gives
them an actual temporal pattern to find, which is what they're built for.
"""

import os
import sys
import numpy as np

DATA_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

sys.path.insert(0, "src")
from mat_loader import load_all_mat_batches
from features   import build_multi_checkpoint_dataset

CHECKPOINTS = [10, 30, 50, 70, 100]

print("=" * 55)
print("  DAY 4a — PREPARE MULTI-CHECKPOINT DATA")
print("=" * 55)

cells = load_all_mat_batches(DATA_DIR)

print(f"\nExtracting features at checkpoints {CHECKPOINTS}...")
data = build_multi_checkpoint_dataset(cells, checkpoints=CHECKPOINTS)

out_path = os.path.join(PROC_DIR, "multi_checkpoint_data.npz")
np.savez_compressed(out_path, **data)

print(f"\n[✓] Saved → {out_path}")
if len(data["cell_ids"]) == 0:
    print("[!] WARNING: 0 cells were successfully extracted — something is wrong.")
    print("    Check that your .mat files are in data/raw/ and loading correctly.")
else:
    print(f"    feature_seq shape : {data['feature_seq'].shape}  (cells, checkpoints, features)")
    print(f"    qdlin_final shape : {data['qdlin_final'].shape}")
    print(f"    cycle_life shape  : {data['cycle_life'].shape}")
    print(f"    feature_names     : {list(data['feature_names'])}")

print("\n[DAY 4a COMPLETE] Next: day4b_ensemble.py — CNN+LSTM+GRU ensemble")