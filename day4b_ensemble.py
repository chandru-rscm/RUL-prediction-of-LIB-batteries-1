"""
day4b_ensemble.py — Day 4, Part B: CNN+LSTM+GRU Ensemble (prediction-level)
Run: python day4b_ensemble.py

REDESIGNED APPROACH — matches the IEEE reference paper's actual method:
  Train CNN, LSTM, GRU, and Elastic Net INDEPENDENTLY (each predicts
  cycle_life on its own from a different view of the data), then combine
  their predictions via a learned weighted average — NOT one giant joint
  model. With only 110 training examples, a single large multi-branch
  network has too many parameters to learn well together; four small,
  independently-trained models are each easier to train, and their
  combination benefits from genuinely uncorrelated errors (the actual
  point of ensembling).
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

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROC_DIR = "data/processed"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, "src")
from models import CNNBranch, SequenceBranch

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("=" * 55)
print("  DAY 4b — CNN+LSTM+GRU ENSEMBLE (prediction-level)")
print("=" * 55)

# ─────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────
mc_path = os.path.join(PROC_DIR, "multi_checkpoint_data.npz")
feat_path = os.path.join(PROC_DIR, "features.csv")

if not os.path.exists(mc_path):
    raise FileNotFoundError("multi_checkpoint_data.npz not found. Run 'python day4a_prepare.py' first.")
if not os.path.exists(feat_path):
    raise FileNotFoundError("features.csv not found. Run 'python day2_features.py' first.")

mc_data = np.load(mc_path, allow_pickle=True)
features_df = pd.read_csv(feat_path)

cell_ids_mc = list(mc_data["cell_ids"])
cycle_life  = mc_data["cycle_life"].astype(np.float32)
feature_seq = mc_data["feature_seq"]
qdlin_final = mc_data["qdlin_final"]

# align features_df rows to the SAME cell order as multi_checkpoint_data
feat_lookup = {cid: i for i, cid in enumerate(features_df["cell_id"])}
align_idx = [feat_lookup[cid] for cid in cell_ids_mc if cid in feat_lookup]
keep_mask = np.array([cid in feat_lookup for cid in cell_ids_mc])

cell_ids    = np.array(cell_ids_mc)[keep_mask]
cycle_life  = cycle_life[keep_mask]
feature_seq = feature_seq[keep_mask]
qdlin_final = qdlin_final[keep_mask]
features_aligned = features_df.iloc[align_idx].reset_index(drop=True)

print(f"Loaded & aligned: {len(cell_ids)} cells")

indices = np.arange(len(cell_ids))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train cells: {len(train_idx)}   Test cells: {len(test_idx)}\n")

y_all = cycle_life
y_train_raw = y_all[train_idx]
y_mean, y_std = float(y_train_raw.mean()), float(y_train_raw.std())


def report_metrics(y_true, y_pred, label):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_pct = rmse / np.mean(y_true) * 100
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2    = r2_score(y_true, y_pred)
    print(f"── {label} ──")
    print(f"  RMSE % : {rmse_pct:.2f}%   MAE: {mae:.1f}   R²: {r2:.3f}")
    return {"label": label, "mae": mae, "rmse": rmse, "rmse_pct": rmse_pct,
            "mape": mape, "r2": r2}


# ─────────────────────────────────────────────────────
#  PREPARE INPUTS (shared across all folds)
# ─────────────────────────────────────────────────────
drop_cols = ["cell_id", "cycle_life", "ic_peak2_height", "ic_peak2_voltage", "ic_peak2_width"]
feature_cols = [c for c in features_aligned.columns if c not in drop_cols]
X_enet_all = features_aligned[feature_cols].values.astype(np.float64)
nan_mask = np.isnan(X_enet_all).any(axis=1)
if nan_mask.any():
    col_medians = np.nanmedian(X_enet_all, axis=0)
    inds = np.where(np.isnan(X_enet_all))
    X_enet_all[inds] = np.take(col_medians, inds[1])

DOWNSAMPLE = 5
qdlin_ds_all = np.nan_to_num(qdlin_final[:, ::DOWNSAMPLE], nan=0.0)
n_features = feature_seq.shape[-1]


def train_small_model(model, X_train, y_train_scaled, epochs=300, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train).squeeze(-1)
        loss = loss_fn(pred, y_train_scaled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())
    return model


# ─────────────────────────────────────────────────────
#  STEP 1 — OUT-OF-FOLD PREDICTIONS on the TRAIN set
#  (the textbook-correct way to fit a stacking meta-learner —
#  each training example's prediction comes from a model that
#  never saw it, so the combiner sees honest, non-overfit signal)
# ─────────────────────────────────────────────────────
print("Generating out-of-fold predictions (5-fold) for honest stacking...")
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_enet = np.zeros(len(train_idx))
oof_cnn  = np.zeros(len(train_idx))
oof_lstm = np.zeros(len(train_idx))
oof_gru  = np.zeros(len(train_idx))

train_idx_arr = np.array(train_idx)

for fold_i, (fold_tr, fold_val) in enumerate(kf.split(train_idx_arr)):
    print(f"  Fold {fold_i+1}/{N_FOLDS}...")
    tr_global = train_idx_arr[fold_tr]
    val_global = train_idx_arr[fold_val]

    # --- Elastic Net ---
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_enet_all[tr_global])
    Xval = sc.transform(X_enet_all[val_global])
    fold_enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=np.logspace(-3, 1, 15),
                              cv=3, max_iter=10000, random_state=RANDOM_STATE)
    fold_enet.fit(Xtr, y_all[tr_global])
    oof_enet[fold_val] = fold_enet.predict(Xval)

    # --- shared normalization stats (fit on fold-train only) ---
    qm, qs = qdlin_ds_all[tr_global].mean(), qdlin_ds_all[tr_global].std()
    qs = qs if qs > 1e-6 else 1.0
    fm = feature_seq[tr_global].mean(axis=(0, 1), keepdims=True)
    fs = feature_seq[tr_global].std(axis=(0, 1), keepdims=True)
    fs = np.where(fs > 1e-6, fs, 1.0)
    ym, ys = y_all[tr_global].mean(), y_all[tr_global].std()

    qdlin_tr_t  = torch.tensor(((qdlin_ds_all[tr_global] - qm) / qs).astype(np.float32))
    qdlin_val_t = torch.tensor(((qdlin_ds_all[val_global] - qm) / qs).astype(np.float32))
    feat_tr_t   = torch.tensor(np.nan_to_num((feature_seq[tr_global] - fm) / fs, nan=0.0).astype(np.float32))
    feat_val_t  = torch.tensor(np.nan_to_num((feature_seq[val_global] - fm) / fs, nan=0.0).astype(np.float32))
    y_tr_scaled = torch.tensor(((y_all[tr_global] - ym) / ys).astype(np.float32))

    # --- CNN ---
    m = train_small_model(CNNBranch(seq_len=qdlin_ds_all.shape[1], out_dim=1), qdlin_tr_t, y_tr_scaled)
    m.eval()
    with torch.no_grad():
        oof_cnn[fold_val] = m(qdlin_val_t).squeeze(-1).numpy() * ys + ym

    # --- LSTM ---
    m = train_small_model(SequenceBranch(n_features, 24, 1, "lstm"), feat_tr_t, y_tr_scaled)
    m.eval()
    with torch.no_grad():
        oof_lstm[fold_val] = m(feat_val_t).squeeze(-1).numpy() * ys + ym

    # --- GRU ---
    m = train_small_model(SequenceBranch(n_features, 24, 1, "gru"), feat_tr_t, y_tr_scaled)
    m.eval()
    with torch.no_grad():
        oof_gru[fold_val] = m(feat_val_t).squeeze(-1).numpy() * ys + ym

# fit the meta-learner on HONEST out-of-fold predictions
oof_matrix = np.column_stack([oof_enet, oof_cnn, oof_lstm, oof_gru])
meta = LinearRegression(positive=True)
meta.fit(oof_matrix, y_all[train_idx])
weights = meta.coef_ / meta.coef_.sum() if meta.coef_.sum() > 0 else np.ones(4) / 4
print(f"\nLearned weights (from honest out-of-fold fit) —")
print(f"  ElasticNet: {weights[0]:.2f}  CNN: {weights[1]:.2f}  LSTM: {weights[2]:.2f}  GRU: {weights[3]:.2f}")

# ─────────────────────────────────────────────────────
#  STEP 2 — retrain each base model on the FULL train set,
#  predict on the held-out TEST set, apply learned weights
# ─────────────────────────────────────────────────────
print("\nRetraining base models on full train set for final test predictions...")

print("  1/4: Elastic Net...")
scaler = StandardScaler()
X_enet_train = scaler.fit_transform(X_enet_all[train_idx])
X_enet_test  = scaler.transform(X_enet_all[test_idx])
enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     alphas=np.logspace(-3, 1, 30), cv=5, max_iter=10000,
                     random_state=RANDOM_STATE)
enet.fit(X_enet_train, y_all[train_idx])
pred_enet_test = enet.predict(X_enet_test)

q_mean, q_std = float(qdlin_ds_all[train_idx].mean()), float(qdlin_ds_all[train_idx].std())
q_std = q_std if q_std > 1e-6 else 1.0
f_mean = feature_seq[train_idx].mean(axis=(0, 1), keepdims=True)
f_std  = feature_seq[train_idx].std(axis=(0, 1), keepdims=True)
f_std  = np.where(f_std > 1e-6, f_std, 1.0)

qdlin_train_t = torch.tensor(((qdlin_ds_all[train_idx] - q_mean) / q_std).astype(np.float32))
qdlin_test_t  = torch.tensor(((qdlin_ds_all[test_idx]  - q_mean) / q_std).astype(np.float32))
feat_train_t  = torch.tensor(np.nan_to_num((feature_seq[train_idx] - f_mean) / f_std, nan=0.0).astype(np.float32))
feat_test_t   = torch.tensor(np.nan_to_num((feature_seq[test_idx]  - f_mean) / f_std, nan=0.0).astype(np.float32))
y_train_scaled = torch.tensor(((y_all[train_idx] - y_mean) / y_std).astype(np.float32))

print("  2/4: CNN...")
cnn_model = train_small_model(CNNBranch(seq_len=qdlin_ds_all.shape[1], out_dim=1), qdlin_train_t, y_train_scaled, epochs=400)
cnn_model.eval()
with torch.no_grad():
    pred_cnn_test = cnn_model(qdlin_test_t).squeeze(-1).numpy() * y_std + y_mean

print("  3/4: LSTM...")
lstm_model = train_small_model(SequenceBranch(n_features, 24, 1, "lstm"), feat_train_t, y_train_scaled, epochs=400)
lstm_model.eval()
with torch.no_grad():
    pred_lstm_test = lstm_model(feat_test_t).squeeze(-1).numpy() * y_std + y_mean

print("  4/4: GRU...")
gru_model = train_small_model(SequenceBranch(n_features, 24, 1, "gru"), feat_train_t, y_train_scaled, epochs=400)
gru_model.eval()
with torch.no_grad():
    pred_gru_test = gru_model(feat_test_t).squeeze(-1).numpy() * y_std + y_mean

print()
y_test_raw = y_all[test_idx]
m_enet = report_metrics(y_test_raw, pred_enet_test, "Elastic Net (base)")
m_cnn  = report_metrics(y_test_raw, pred_cnn_test,  "CNN (base)")
m_lstm = report_metrics(y_test_raw, pred_lstm_test, "LSTM (base)")
m_gru  = report_metrics(y_test_raw, pred_gru_test,  "GRU (base)")

# apply the HONEST weights (learned from out-of-fold data) to test predictions
test_preds = np.column_stack([pred_enet_test, pred_cnn_test, pred_lstm_test, pred_gru_test])
y_pred_ensemble = meta.predict(test_preds)
y_pred_ensemble = np.clip(y_pred_ensemble, 1, None)

print()
ensemble_metrics = report_metrics(y_test_raw, y_pred_ensemble, "FINAL ENSEMBLE")

# ─────────────────────────────────────────────────────
#  COMPARISON
# ─────────────────────────────────────────────────────
day3_path = os.path.join(PROC_DIR, "day3_baseline_metrics.csv")
print("\n" + "=" * 55)
print("  COMPARISON — all models")
print("=" * 55)
all_results = pd.DataFrame([m_enet, m_cnn, m_lstm, m_gru, ensemble_metrics]).set_index("label")
print(all_results[["rmse_pct", "mae", "r2"]].to_string())

if os.path.exists(day3_path):
    day3_df = pd.read_csv(day3_path, index_col=0)
    best_day3 = day3_df["rmse_pct"].min()
    print(f"\n  Day 3 best (Elastic Net alone): {best_day3:.2f}% error")
    print(f"  Day 4 final ensemble           : {ensemble_metrics['rmse_pct']:.2f}% error")
    if ensemble_metrics["rmse_pct"] < best_day3:
        print(f"  -> Ensemble BEATS the Day 3 baseline by {best_day3 - ensemble_metrics['rmse_pct']:.2f} points!")
    else:
        print(f"  -> Ensemble did not beat Day 3 baseline (gap: {ensemble_metrics['rmse_pct'] - best_day3:.2f} points)")

# ─────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test_raw, y_pred_ensemble, color="#534AB7", alpha=0.7, s=45, edgecolors="white", linewidths=0.5)
lims = [min(y_test_raw.min(), y_pred_ensemble.min()) - 50, max(y_test_raw.max(), y_pred_ensemble.max()) + 50]
ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual cycle life")
ax.set_ylabel("Predicted cycle life")
ax.set_title(f"Day 4 — Final Ensemble\nRMSE% = {ensemble_metrics['rmse_pct']:.2f}%")
ax.legend(fontsize=9)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "day4_ensemble_predictions.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n[✓] Prediction plot saved → {plot_path}")

fig2, ax2 = plt.subplots(figsize=(8, 5))
labels = ["Elastic Net", "CNN", "LSTM", "GRU", "Ensemble"]
values = [m_enet["rmse_pct"], m_cnn["rmse_pct"], m_lstm["rmse_pct"], m_gru["rmse_pct"], ensemble_metrics["rmse_pct"]]
colors = ["#534AB7", "#0F6E56", "#BA7517", "#D85A30", "#1E40AF"]
ax2.bar(labels, values, color=colors)
ax2.set_ylabel("RMSE % error")
ax2.set_title("Day 4 — Base Models vs Final Ensemble")
ax2.axhline(9.84, color="gray", linestyle="--", linewidth=1, label="Day 3 Elastic Net benchmark")
ax2.legend(fontsize=9)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "day4_model_comparison.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
print(f"[✓] Model comparison plot saved → {bar_path}")

all_results.to_csv(os.path.join(PROC_DIR, "day4_ensemble_metrics.csv"))

print("\n[DAY 4b COMPLETE] Next: Day 5 — SHAP explainability + multi-threshold prediction")