"""
day3_baseline.py — Day 3: Baseline Models
Run: python day3_baseline.py

What this does:
  1. Load features.csv (18 scalar features) + sequences.npz (curve shapes)
  2. Train/test split BY CELL (same split used for both models — fair comparison)
  3. Baseline 1 — Elastic Net on scalar features (reproduces Nature paper's approach)
  4. Baseline 2 — Single LSTM on the capacity-fade curve shape (deep learning baseline)
  5. Compare both: MAE, RMSE, RMSE% (matches paper's "X% error" convention), R²
  6. Plot predicted vs actual for both models

These numbers are what Day 4's ensemble model needs to BEAT.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROC_DIR = "data/processed"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, "src")
from models import SimpleLSTM, SimpleMLP

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("=" * 55)
print("  DAY 3 — BASELINE MODELS")
print("=" * 55)

# ─────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────
features_path = os.path.join(PROC_DIR, "features.csv")
sequences_path = os.path.join(PROC_DIR, "sequences.npz")

if not os.path.exists(features_path) or not os.path.exists(sequences_path):
    raise FileNotFoundError(
        "features.csv or sequences.npz not found in data/processed/.\n"
        "Run 'python day2_features.py' first."
    )

features_df = pd.read_csv(features_path)
seq_data    = np.load(sequences_path)

print(f"Features loaded : {features_df.shape}")
print(f"Sequences loaded: qdlin_curve={seq_data['qdlin_curve'].shape}, "
      f"capacity_curve={seq_data['capacity_curve'].shape}")

# align sequences with features_df by cell_id (safety — order should already match)
seq_cell_ids = list(seq_data["cell_ids"])
seq_index = {cid: i for i, cid in enumerate(seq_cell_ids)}
features_df = features_df[features_df["cell_id"].isin(seq_index)].reset_index(drop=True)

align_idx = [seq_index[cid] for cid in features_df["cell_id"]]
capacity_curves = seq_data["capacity_curve"][align_idx]   # (N, 100)
qdlin_curves    = seq_data["qdlin_curve"][align_idx]       # (N, 1000)
cycle_life_seq  = seq_data["cycle_life"][align_idx]

assert np.array_equal(cycle_life_seq, features_df["cycle_life"].values), \
    "Mismatch between features.csv and sequences.npz cycle_life — alignment bug!"

print(f"Aligned dataset : {len(features_df)} cells\n")

# ─────────────────────────────────────────────────────
#  TRAIN / TEST SPLIT (shared by both models — fair comparison)
# ─────────────────────────────────────────────────────
indices = np.arange(len(features_df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)

print(f"Train cells: {len(train_idx)}   Test cells: {len(test_idx)}\n")

y_all = features_df["cycle_life"].values.astype(np.float32)

# ─────────────────────────────────────────────────────
#  METRIC HELPER (matches papers' "X% error" reporting convention)
# ─────────────────────────────────────────────────────
def report_metrics(y_true, y_pred, label):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_pct = rmse / np.mean(y_true) * 100          # RMSE relative to mean (paper-style)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2    = r2_score(y_true, y_pred)

    print(f"── {label} ──")
    print(f"  MAE              : {mae:.1f} cycles")
    print(f"  RMSE             : {rmse:.1f} cycles")
    print(f"  RMSE %  (paper-style 'X% error') : {rmse_pct:.2f}%")
    print(f"  MAPE             : {mape:.2f}%")
    print(f"  R²               : {r2:.3f}")
    print()
    return {"label": label, "mae": mae, "rmse": rmse, "rmse_pct": rmse_pct,
            "mape": mape, "r2": r2}


# ─────────────────────────────────────────────────────
#  BASELINE 1 — ELASTIC NET (on scalar features)
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  BASELINE 1 — ELASTIC NET")
print("=" * 55)

# drop dead columns (always 0/NaN — single IC peak chemistry, see Day 2 notes)
drop_cols = ["cell_id", "cycle_life", "ic_peak2_height", "ic_peak2_voltage", "ic_peak2_width"]
feature_cols = [c for c in features_df.columns if c not in drop_cols]

X = features_df[feature_cols].values.astype(np.float64)
y = y_all

nan_mask = np.isnan(X).any(axis=1)
if nan_mask.any():
    print(f"[!] {nan_mask.sum()} rows have NaN features — filling with column median")
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

enet = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-3, 1, 30),
    cv=5, max_iter=10000, random_state=RANDOM_STATE,
)
enet.fit(X_train_s, y_train)

y_pred_enet = enet.predict(X_test_s)
y_pred_enet = np.clip(y_pred_enet, 1, None)  # cycle_life can't be negative

enet_metrics = report_metrics(y_test, y_pred_enet, "Elastic Net")
print(f"  Best alpha: {enet.alpha_:.4f}   Best l1_ratio: {enet.l1_ratio_:.2f}")

# feature importance (coefficient magnitude)
coef_importance = pd.Series(np.abs(enet.coef_), index=feature_cols).sort_values(ascending=False)
print(f"\n  Top 5 features by Elastic Net coefficient magnitude:")
print(coef_importance.head(5).to_string())
print()


# ─────────────────────────────────────────────────────
#  BASELINE 2 — SINGLE LSTM (on the SAME engineered features as ElasticNet)
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  BASELINE 2 — NEURAL NETWORK (MLP)")
print("=" * 55)

# NOTE: we initially tried feeding this LSTM the raw Qdlin voltage-capacity
# curve directly (1000 points), hoping it would learn its own features.
# With only 110 training examples, it couldn't reliably do this — a
# diagnostic confirmed PyTorch training itself was fine (a tiny network
# trained on a single pre-extracted feature reached R²=0.95 easily), so
# the bottleneck was specifically "rediscovering Day 2's feature engineering
# from raw signal with too little data" — a well-known, expected deep
# learning limitation, not a bug. This is exactly why feature engineering
# (Day 2) matters, and why Day 4's ensemble will combine engineered
# features with raw-signal learning rather than relying on either alone.
#
# So: this LSTM uses the SAME 18 engineered features as Elastic Net —
# a fair, reliable linear-vs-neural-network comparison on equal footing.

# So: this network uses the SAME 18 engineered features, SAME
# StandardScaler-transformed values as Elastic Net — a fair,
# reliable linear-vs-neural-network comparison on equal footing.
# A plain MLP is used here (not the LSTM) since there's no time
# dimension in this input — SimpleLSTM is reserved for genuine
# multi-timestep sequence tasks in Day 4.

y_train_raw = y_all[train_idx]
y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())

X_mlp_train = torch.tensor(X_train_s.astype(np.float32))
X_mlp_test  = torch.tensor(X_test_s.astype(np.float32))
y_mlp_train = torch.tensor(((y_all[train_idx] - y_mean) / y_std).astype(np.float32))
y_mlp_test  = torch.tensor(((y_all[test_idx]  - y_mean) / y_std).astype(np.float32))

model = SimpleMLP(n_features=X_mlp_train.shape[1], hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=80)
loss_fn = nn.MSELoss()

EPOCHS = 500
train_losses = []

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(X_mlp_train)
    loss = loss_fn(pred, y_mlp_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())
    train_losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{EPOCHS}  train_loss={loss.item():.4f}  lr={current_lr:.5f}")

model.eval()
with torch.no_grad():
    y_pred_lstm_scaled = model(X_mlp_test).numpy()
    y_pred_lstm = y_pred_lstm_scaled * y_std + y_mean   # inverse-transform back to real cycle units
    y_pred_lstm = np.clip(y_pred_lstm, 1, None)

print()
lstm_metrics = report_metrics(y_test, y_pred_lstm, "Neural Network")


# ─────────────────────────────────────────────────────
#  COMPARISON
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  COMPARISON — Day 3 baselines vs Nature paper benchmark")
print("=" * 55)
comparison_df = pd.DataFrame([enet_metrics, lstm_metrics]).set_index("label")
print(comparison_df.to_string())
print(f"\n  Nature paper benchmark (reference): ~9.1% error (their RMSE-style metric)")
print(f"  Day 4's ensemble model needs to beat the better of these two baselines.")

comparison_df.to_csv(os.path.join(PROC_DIR, "day3_baseline_metrics.csv"))

# ─────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Day 3 — Predicted vs Actual Cycle Life", fontsize=14, fontweight="bold")

for ax, y_pred, label, color in zip(
    axes, [y_pred_enet, y_pred_lstm], ["Elastic Net", "Neural Network"], ["#534AB7", "#0F6E56"]
):
    ax.scatter(y_test, y_pred, color=color, alpha=0.7, s=40, edgecolors="white", linewidths=0.5)
    lims = [min(y_test.min(), y_pred.min()) - 50, max(y_test.max(), y_pred.max()) + 50]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual cycle life")
    ax.set_ylabel("Predicted cycle life")
    rmse_pct = enet_metrics["rmse_pct"] if label == "Elastic Net" else lstm_metrics["rmse_pct"]
    ax.set_title(f"{label}\nRMSE% = {rmse_pct:.2f}%")
    ax.legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "day3_baseline_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n[✓] Comparison plot saved → {plot_path}")

# LSTM training loss curve
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(train_losses, color="#0F6E56", linewidth=1)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Train loss (MSE, scaled target)")
ax2.set_title("Neural Network Training Loss Curve")
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "day3_lstm_training_loss.png")
plt.savefig(loss_path, dpi=150, bbox_inches="tight")
print(f"[✓] LSTM loss curve saved → {loss_path}")

print("\n[DAY 3 COMPLETE] Next: day4_ensemble.py — CNN+LSTM+GRU ensemble + dynamic prediction")