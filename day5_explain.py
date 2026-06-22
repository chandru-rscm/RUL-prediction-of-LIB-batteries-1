"""
day5_explain.py — Day 5: Explainability + Multi-threshold + Dynamic re-prediction

What this does and why (recap of the project's actual contributions beyond
the reference papers):

  1. SHAP explainability  — WHICH of our 17 features actually drives the
     prediction, not just a black-box number. Neither reference paper has this.

  2. Multi-threshold prediction — instead of one "cycles to 80% EOL" number,
     predict cycles-to-90%, cycles-to-85%, cycles-to-80%. Useful for real
     deployment decisions (e.g. "reassign to second-life at 85%", discussed
     earlier in the project).

  3. Dynamic re-prediction — re-run the model at checkpoints beyond cycle 100
     (200, 300, 400, 600, 800...) and show the prediction sharpening as more
     real aging data becomes available. This is the project's core claim:
     static models (the reference papers) predict once and freeze; ours
     keeps improving across the battery's life.

Uses Day 2's cached features.csv (fast — no .mat reload) for parts 1-2.
Part 3 needs the raw cells again (one more ~6 min .mat reload) since it
extracts features at NEW checkpoints (200, 300, 400, 600, 800) that Day 2
never computed.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from explain import get_feature_columns, train_elasticnet, compute_shap_values, evaluate_threshold_models
from dynamic import build_dynamic_checkpoint_features, evaluate_dynamic_predictions

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE_FRAC = 0.2
DYNAMIC_CHECKPOINTS = [100, 200, 300, 400, 600, 800]  # 100 = same starting point as papers

print("=" * 55)
print("  DAY 5 — EXPLAINABILITY + MULTI-THRESHOLD + DYNAMIC")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# PART 1 — SHAP explainability (uses Day 2's features.csv)
# ─────────────────────────────────────────────────────────
print("\n--- Part 1: SHAP explainability ---")
df = pd.read_csv("data/processed/features.csv")
feature_cols = get_feature_columns(df)
print(f"Using {len(feature_cols)} features (dropped dead/id/target columns)")

from sklearn.model_selection import train_test_split
X = df[feature_cols].fillna(0).values
y = df["cycle_life"].values
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=TEST_SIZE_FRAC, random_state=RANDOM_STATE
)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

model, scaler = train_elasticnet(X_train, y_train, RANDOM_STATE)
shap_values, importance_df = compute_shap_values(model, scaler, X_train, X_test, feature_cols)

print("\nTop 5 features by SHAP importance:")
print(importance_df.head(5).to_string(index=False))

# Plot: SHAP bar chart
fig, ax = plt.subplots(figsize=(8, 5))
top10 = importance_df.head(10).iloc[::-1]
ax.barh(top10["feature"], top10["mean_abs_shap"], color="#4C72B0")
ax.set_xlabel("Mean |SHAP value| (impact on prediction, in cycles)")
ax.set_title("Which features actually drive the RUL prediction?")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/day5_shap_importance.png", dpi=150)
plt.close()
print(f"[✓] SHAP plot saved → {OUTPUT_DIR}/day5_shap_importance.png")

importance_df.to_csv(f"{OUTPUT_DIR}/day5_shap_importance.csv", index=False)

# ─────────────────────────────────────────────────────────
# PART 2 — Multi-threshold prediction (90% / 85% / 80%)
# ─────────────────────────────────────────────────────────
print("\n--- Part 2: Multi-threshold prediction (90% / 85% / 80%) ---")
metrics_df, threshold_models, idx_train_t, idx_test_t, targets = evaluate_threshold_models(
    df, feature_cols, thresholds=(0.90, 0.85, 0.80),
    test_size_frac=TEST_SIZE_FRAC, random_state=RANDOM_STATE,
)
print(metrics_df.to_string(index=False))
metrics_df.to_csv(f"{OUTPUT_DIR}/day5_threshold_metrics.csv", index=False)

# Plot: predicted cycle counts to each threshold, for a handful of test cells
fig, ax = plt.subplots(figsize=(9, 5))
n_show = min(10, len(idx_test_t))
show_idx = idx_test_t[:n_show]
x_pos = np.arange(n_show)
width = 0.25
for i, (thresh, color) in enumerate(zip([0.90, 0.85, 0.80], ["#55A868", "#DD8452", "#C44E52"])):
    vals = targets[thresh][show_idx]
    ax.bar(x_pos + (i - 1) * width, vals, width=width, label=f"{int(thresh*100)}% SOH", color=color)
ax.set_xticks(x_pos)
ax.set_xticklabels(df["cell_id"].values[show_idx], rotation=45, ha="right")
ax.set_ylabel("Cycles to reach threshold")
ax.set_title("Multi-threshold prediction: when does each cell hit 90% / 85% / 80% SOH?")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/day5_multi_threshold.png", dpi=150)
plt.close()
print(f"[✓] Multi-threshold plot saved → {OUTPUT_DIR}/day5_multi_threshold.png")

# ─────────────────────────────────────────────────────────
# PART 3 — Dynamic re-prediction (needs raw cells, one more .mat reload)
# ─────────────────────────────────────────────────────────
print("\n--- Part 3: Dynamic re-prediction (reloading raw cells, ~6 min) ---")
try:
    from mat_loader import load_all_mat_batches
    from features import extract_cell_features  # real signature: (cell, checkpoint=100, early=10) -> dict | None

    cells = load_all_mat_batches("data/raw")
    print(f"Loaded {len(cells)} cells")

    checkpoint_data = build_dynamic_checkpoint_features(
        cells, DYNAMIC_CHECKPOINTS, extract_cell_features, early=10
    )
    # (build_dynamic_checkpoint_features already prints per-checkpoint
    #  survivorship + error diagnostics above, verbose=True by default)

    # Use the SAME test cells as Day 3/Part 1-2 for a fair, consistent comparison.
    # We identify them by position in features.csv (idx_test), mapped to cell_id.
    test_cell_ids = set(df["cell_id"].values[idx_test])

    dyn_metrics, per_cell_preds = evaluate_dynamic_predictions(checkpoint_data, test_cell_ids, RANDOM_STATE)
    print("\nDynamic re-prediction accuracy across checkpoints:")
    print(dyn_metrics.to_string(index=False))
    dyn_metrics.to_csv(f"{OUTPUT_DIR}/day5_dynamic_metrics.csv", index=False)

    # Plot: error% vs checkpoint — should trend down (the project's core claim)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dyn_metrics["checkpoint"], dyn_metrics["rmse_pct"], marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("Checkpoint cycle (when the re-prediction happens)")
    ax.set_ylabel("RMSE % error")
    ax.set_title("Dynamic re-prediction: error shrinks with more real aging data")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/day5_dynamic_error_trend.png", dpi=150)
    plt.close()
    print(f"[✓] Dynamic trend plot saved → {OUTPUT_DIR}/day5_dynamic_error_trend.png")

    # Plot: a few individual cells' predictions converging toward true value.
    # IMPORTANT: not every cell survives every checkpoint (shorter-lived cells
    # drop out early) — pick cells that cover the WIDEST checkpoint range so
    # the convergence story is genuinely visible, and only ever draw a line
    # across checkpoints that cell actually has data for (no implied merging).
    fig, ax = plt.subplots(figsize=(9, 5))
    candidates = sorted(per_cell_preds.items(), key=lambda kv: -len(kv[1]))
    shown = 0
    for cid, preds_by_cp in candidates:
        if len(preds_by_cp) < 4:
            continue
        cps = sorted(preds_by_cp.keys())
        true_val = preds_by_cp[cps[0]][0]
        pred_vals = [preds_by_cp[cp][1] for cp in cps]
        line, = ax.plot(cps, pred_vals, marker="o", alpha=0.85,
                         label=f"{cid} (true={true_val:.0f}, last cp={cps[-1]})")
        ax.axhline(true_val, linestyle="--", alpha=0.25, color=line.get_color())
        shown += 1
        if shown >= 6:
            break
    if shown == 0:
        ax.text(0.5, 0.5, "No cell survived enough checkpoints to plot convergence",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Checkpoint cycle")
    ax.set_ylabel("Predicted total cycle life")
    ax.set_title("Individual cells: prediction converging toward true value over time\n(lines end where that cell's real data ends)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/day5_dynamic_per_cell.png", dpi=150)
    plt.close()
    print(f"[✓] Per-cell convergence plot saved → {OUTPUT_DIR}/day5_dynamic_per_cell.png")

except ImportError as e:
    print(f"\n[SKIPPED] Part 3 needs extract_cell_features(cell, checkpoint=..., early=...) in src/features.py")
    print(f"          ImportError: {e}")
except FileNotFoundError as e:
    print(f"\n[SKIPPED] Part 3 needs .mat files in data/raw/: {e}")

print("\n[DAY 5 COMPLETE] Next: Day 6 — degradation mode classification + second-life recommendation")