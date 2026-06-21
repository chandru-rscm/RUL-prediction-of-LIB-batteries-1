"""
day2_features.py — Day 2: Feature Engineering
Run: python day2_features.py

What this does:
  1. Load dataset (same as Day 1)
  2. Extract features at cycle 100 for every cell:
     - Delta-Q(V) stats (Nature paper baseline feature)
     - IC curve peaks (height, voltage, width) — computed via gradient
     - Internal resistance trend
     - Charge time trend
     - Capacity fade / SOH
  3. Save feature table to data/processed/features.csv
  4. Plot feature correlations with cycle_life (sanity check)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/raw"
OUT_DIR  = "outputs"
PROC_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

sys.path.insert(0, "src")
from mat_loader import load_all_mat_batches
from features   import build_feature_dataframe, compute_dqdv, V_GRID, build_sequence_cache

CHECKPOINT = 100   # cycle at which we evaluate (matches Nature paper)
EARLY      = 10    # early cycle for Delta-Q(V)

print("=" * 55)
print("  DAY 2 — FEATURE ENGINEERING")
print("=" * 55)

# ── load data ────────────────────────────────────────
cells = load_all_mat_batches(DATA_DIR)

# ── extract features ─────────────────────────────────
print(f"\nExtracting features at checkpoint cycle={CHECKPOINT}, early cycle={EARLY}...")
features_df = build_feature_dataframe(cells, checkpoint=CHECKPOINT, early=EARLY)

# ── save ──────────────────────────────────────────────
csv_path = os.path.join(PROC_DIR, "features.csv")
features_df.to_csv(csv_path, index=False)
print(f"[✓] Features saved → {csv_path}")

# ── cache raw sequences for Day 3+ LSTM/CNN models ────
# (do this now, while `cells` is already in memory — avoids
#  reloading the huge .mat files again in later scripts)
print("\nBuilding sequence cache for Day 3+ (LSTM/CNN inputs)...")
seq_cache = build_sequence_cache(cells, checkpoint=CHECKPOINT, curve_len=CHECKPOINT)
seq_path = os.path.join(PROC_DIR, "sequences.npz")
np.savez_compressed(seq_path, **seq_cache)
print(f"[✓] Sequence cache saved → {seq_path}")
print(f"    qdlin_curve shape    : {seq_cache['qdlin_curve'].shape}")
print(f"    capacity_curve shape : {seq_cache['capacity_curve'].shape}")
print(f"\nFeature columns: {list(features_df.columns)}")
print(f"\nFirst 5 rows:")
print(features_df.head().to_string(index=False))

# ─────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.0)
feature_cols = [c for c in features_df.columns if c not in ("cell_id", "cycle_life")]

# ── Plot 1: correlation of each feature with cycle_life ──
corrs = features_df[feature_cols + ["cycle_life"]].corr()["cycle_life"].drop("cycle_life")
corrs = corrs.dropna().sort_values()

fig1, ax1 = plt.subplots(figsize=(9, max(4, len(corrs) * 0.35)))
colors = ["#E86540" if v < 0 else "#0F6E56" for v in corrs.values]
ax1.barh(corrs.index, corrs.values, color=colors)
ax1.axvline(0, color="white", linewidth=0.8)
ax1.set_xlabel("Correlation with cycle_life")
ax1.set_title(f"Feature Correlation with RUL (target)\nat checkpoint cycle {CHECKPOINT}")
plt.tight_layout()
corr_path = os.path.join(OUT_DIR, "day2_feature_correlations.png")
plt.savefig(corr_path, dpi=150, bbox_inches="tight")
print(f"\n[✓] Correlation plot saved → {corr_path}")

# ── Plot 2: scatter of top 4 most-correlated features vs cycle_life ──
top_feats = corrs.abs().sort_values(ascending=False).head(4).index.tolist()

fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
fig2.suptitle("Top 4 Features vs Cycle Life", fontsize=14, fontweight="bold")

for ax, feat in zip(axes.flat, top_feats):
    valid = features_df[[feat, "cycle_life"]].dropna()
    ax.scatter(valid[feat], valid["cycle_life"], color="#534AB7", alpha=0.6,
               s=30, edgecolors="white", linewidths=0.3)
    ax.set_xlabel(feat)
    ax.set_ylabel("cycle_life")
    r = valid[feat].corr(valid["cycle_life"])
    ax.set_title(f"r = {r:.3f}")

plt.tight_layout()
scatter_path = os.path.join(OUT_DIR, "day2_top_features_scatter.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"[✓] Top features scatter saved → {scatter_path}")

# ── Plot 3: example IC curves (computed via gradient) for a few cells ──
fig3, ax3 = plt.subplots(figsize=(9, 5))
sample_ids = list(cells.keys())[:6]
cmap = matplotlib.colormaps["plasma"].resampled(len(sample_ids))

for idx, cid in enumerate(sample_ids):
    cell = cells[cid]
    cyc  = cell["cycles"].get(str(CHECKPOINT), {})
    qdlin = cyc.get("Qdlin", np.array([]))
    if len(qdlin) == 0:
        continue
    dqdv = compute_dqdv(qdlin)
    ax3.plot(V_GRID, dqdv, color=cmap(idx), linewidth=1.3, label=f"{cid} (life={cell['cycle_life']})")

ax3.set_xlabel("Voltage (V)")
ax3.set_ylabel("dQ/dV (computed via gradient)")
ax3.set_title(f"IC Curves at Cycle {CHECKPOINT} — Computed from Qdlin\n(your own signal processing, not dataset-provided)")
ax3.legend(fontsize=8)
ax3.invert_xaxis()
plt.tight_layout()
ic_path = os.path.join(OUT_DIR, "day2_computed_ic_curves.png")
plt.savefig(ic_path, dpi=150, bbox_inches="tight")
print(f"[✓] Computed IC curves saved → {ic_path}")

print("\n── FEATURE STATS SUMMARY ─────────────────────")
print(features_df[feature_cols].describe().T[["mean", "std", "min", "max"]].to_string())

print("\n── TOP 5 FEATURES MOST CORRELATED WITH RUL ───")
print(corrs.abs().sort_values(ascending=False).head(5).to_string())

print("\n[DAY 2 COMPLETE] Next: day3_baseline.py — Elastic Net baseline + single LSTM")