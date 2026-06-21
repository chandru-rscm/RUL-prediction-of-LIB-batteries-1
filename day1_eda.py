"""
day1_eda.py — Day 1: Exploratory Data Analysis
Run: python day1_eda.py

What this does:
  1. Load dataset (MIT or NASA)
  2. Print data structure so you understand what's inside each cell
  3. Plot: degradation curves, voltage curves, cycle life distribution
  4. Save plots to outputs/

Change DATA_SOURCE and DATA_DIR below to match your setup.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm  # noqa: F401 (kept for compatibility)
import seaborn as sns

# ── point this to wherever your .mat files are ──────
DATA_DIR = "data/raw"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, "src")
from mat_loader  import load_all_mat_batches
from data_loader import get_summary_df, get_capacity_df, get_cycle

# ─────────────────────────────────────────────────────
#  MOCK DATA (used when no real pkl files present yet)
#  Remove this block once you have real data
# ─────────────────────────────────────────────────────
def _make_mock_cells(n=40, seed=42):
    """Simulates realistic battery degradation for testing."""
    rng  = np.random.default_rng(seed)
    cells = {}
    for i in range(n):
        life   = int(rng.integers(400, 1500))
        cycles = {}
        cap    = 1.1

        for c in range(1, life + 1):
            # realistic voltage curve shape
            v_pts = 100
            cap   = max(0.85, cap - rng.uniform(0.0003, 0.0008))
            V     = np.linspace(3.6, 2.0, v_pts) + rng.normal(0, 0.005, v_pts)
            Qd    = np.linspace(cap, 0, v_pts)

            cycles[str(c)] = {
                "V":  V,
                "Qd": Qd,
                "I":  np.full(v_pts, -4.0),
                "T":  30 + rng.uniform(-2, 5, v_pts),
                "t":  np.linspace(0, cap / 4.0 * 3600, v_pts),
            }

        # summary arrays
        q_list = np.array([
            float(cycles[str(c)]["Qd"][-1]) if float(cycles[str(c)]["Qd"][0]) > 0
            else 1.1 * (1 - c / life * 0.2)
            for c in range(1, life + 1)
        ])
        q_list = np.linspace(1.1, 0.87, life) + rng.normal(0, 0.003, life)

        cells[f"mock_c{i}"] = {
            "cycles":     cycles,
            "cycle_life": life,
            "summary": {
                "QDischarge": q_list,
                "QCharge":    q_list + rng.uniform(0.01, 0.03, life),
                "IR":         np.linspace(0.015, 0.030, life) + rng.normal(0, 0.0005, life),
                "Tmax":       30 + rng.uniform(0, 5, life),
                "chargetime": np.linspace(360, 480, life),
            },
        }
    return cells


# ─────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────
print("=" * 55)
print("  DAY 1 — EDA")
print("=" * 55)

# try to load real .mat files, fall back to mock
try:
    cells = load_all_mat_batches(DATA_DIR)
    if not cells:
        raise FileNotFoundError
except Exception as e:
    print(f"\n[!] Could not load real data ({e})")
    print("    Running with MOCK data — put your .mat files in data/raw/ and re-run\n")
    cells = _make_mock_cells(n=40)

summary_df = get_summary_df(cells)
cap_df     = get_capacity_df(cells)

# ─────────────────────────────────────────────────────
#  1. DATA STRUCTURE PRINTOUT
# ─────────────────────────────────────────────────────
print("\n── DATASET OVERVIEW ──────────────────────────")
print(f"  Total cells         : {len(cells)}")
print(f"  Cycle life range    : {summary_df.cycle_life.min()} – {summary_df.cycle_life.max()} cycles")
print(f"  Mean cycle life     : {summary_df.cycle_life.mean():.0f} cycles")
print(f"  Median cycle life   : {summary_df.cycle_life.median():.0f} cycles")

sample_cell = list(cells.values())[0]
print(f"\n── ONE CELL STRUCTURE ────────────────────────")
print(f"  Keys                : {list(sample_cell.keys())}")
print(f"  Summary keys        : {list(sample_cell['summary'].keys())}")
print(f"  Number of cycles    : {len(sample_cell['cycles'])}")
_rep_cyc = "50" if "50" in sample_cell["cycles"] else list(sample_cell["cycles"].keys())[len(sample_cell["cycles"])//2]
cyc1 = sample_cell["cycles"][_rep_cyc]
print(f"  (using cycle {_rep_cyc} for display — cycle 1 is often a near-empty formation cycle)")
print(f"  Cycle 1 keys        : {list(cyc1.keys())}")
print(f"  Cycle 1 V shape     : {cyc1['V'].shape}")
print(f"  Voltage range (c1)  : {cyc1['V'].min():.2f}V – {cyc1['V'].max():.2f}V")
print(f"  Capacity range (c1) : {cyc1['Qd'].min():.4f} – {cyc1['Qd'].max():.4f} Ah")

# ─────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.05)
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Day 1 EDA — LiB Dataset Overview", fontsize=16, fontweight="bold", y=0.98)

def _baseline_capacity(q):
    """
    Get a stable early-cycle capacity baseline for normalization.
    Index 0 (cycle 1) is often a near-empty formation cycle with a
    trivial ~0 reading — skip it and use index 4 (cycle 5) or the
    first positive value instead.
    """
    if len(q) == 0:
        return np.nan
    idx = min(4, len(q) - 1)
    if q[idx] > 0:
        return q[idx]
    positive = q[q > 0]
    return positive[0] if len(positive) > 0 else np.nan


# ── Plot 1: Cycle life distribution ──
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(summary_df.cycle_life, bins=20, color="#534AB7", edgecolor="white", linewidth=0.5)
ax1.axvline(summary_df.cycle_life.mean(),   color="#E86540", linewidth=2, linestyle="--", label=f"Mean: {summary_df.cycle_life.mean():.0f}")
ax1.axvline(summary_df.cycle_life.median(), color="#0F6E56", linewidth=2, linestyle=":",  label=f"Median: {summary_df.cycle_life.median():.0f}")
ax1.set_xlabel("Cycle life (total cycles)")
ax1.set_ylabel("Number of cells")
ax1.set_title("Cycle Life Distribution")
ax1.legend(fontsize=9)

# ── Plot 2: Degradation curves — sample of 15 cells ──
ax2 = fig.add_subplot(3, 3, (2, 3))
# only use cells with valid (non-empty) QDischarge data
_valid_cell_ids = [cid for cid in cells if len(cells[cid]["summary"].get("QDischarge", [])) > 0]
sample_cells = _valid_cell_ids[:15]
cmap = matplotlib.colormaps["viridis"].resampled(len(sample_cells))

for idx, cell_id in enumerate(sample_cells):
    cell = cells[cell_id]
    q    = cell["summary"]["QDischarge"]
    c    = np.arange(1, len(q) + 1)
    ax2.plot(c, q / _baseline_capacity(q) * 100, color=cmap(idx), linewidth=1.0, alpha=0.85)

ax2.axhline(80, color="#E86540", linewidth=1.5, linestyle="--", label="80% EOL threshold")
ax2.axhline(85, color="#BA7517", linewidth=1.0, linestyle=":",  label="85% second-life trigger")
ax2.set_xlabel("Cycle number")
ax2.set_ylabel("SOH % (capacity / initial)")
ax2.set_title("Degradation Curves — 15 Cells")
ax2.legend(fontsize=9)

# ── Plot 3: Voltage curve at different cycle stages ──
ax3 = fig.add_subplot(3, 3, 4)
cell0    = list(cells.values())[0]
n_cyc    = len(cell0["cycles"])
checkpts = [2, max(3, n_cyc // 4), max(4, n_cyc // 2), max(5, n_cyc * 3 // 4), n_cyc]
checkpts = sorted(set(min(c, n_cyc) for c in checkpts))

cmap2 = matplotlib.colormaps["plasma"].resampled(len(checkpts))
for idx, cp in enumerate(checkpts):
    cyc = get_cycle(cell0, cp)
    ax3.plot(cyc["Qd"], cyc["V"], color=cmap2(idx), linewidth=1.5, label=f"Cycle {cp}")
ax3.set_xlabel("Discharge capacity (Ah)")
ax3.set_ylabel("Voltage (V)")
ax3.set_title("Voltage Curves — Same Cell, Different Cycles")
ax3.legend(fontsize=8)

# ── Plot 4: Capacity fade — early vs late cells ──
ax4 = fig.add_subplot(3, 3, 5)
sorted_cells = summary_df.sort_values("cycle_life")
short_ids = sorted_cells.head(5)["cell_id"].tolist()
long_ids  = sorted_cells.tail(5)["cell_id"].tolist()

for cid in short_ids:
    q = cells[cid]["summary"]["QDischarge"]
    if len(q) == 0:
        continue
    base = _baseline_capacity(q)
    if not np.isfinite(base) or base <= 0:
        continue
    ax4.plot(np.arange(1, len(q)+1), q / base * 100,
             color="#E86540", linewidth=1.0, alpha=0.7)
for cid in long_ids:
    q = cells[cid]["summary"]["QDischarge"]
    if len(q) == 0:
        continue
    base = _baseline_capacity(q)
    if not np.isfinite(base) or base <= 0:
        continue
    ax4.plot(np.arange(1, len(q)+1), q / base * 100,
             color="#534AB7", linewidth=1.0, alpha=0.7)

from matplotlib.lines import Line2D
ax4.legend(handles=[
    Line2D([0],[0], color="#E86540", label="Short-life cells (bottom 5)"),
    Line2D([0],[0], color="#534AB7", label="Long-life cells (top 5)"),
], fontsize=8)
ax4.axhline(80, color="gray", linewidth=1, linestyle="--")
ax4.set_xlabel("Cycle number")
ax4.set_ylabel("SOH %")
ax4.set_title("Short-life vs Long-life Cells")

# ── Plot 5: Internal resistance rise ──
ax5 = fig.add_subplot(3, 3, 6)
for idx, cell_id in enumerate(sample_cells[:8]):
    cell = cells[cell_id]
    if "IR" in cell["summary"]:
        ir = cell["summary"]["IR"]
        ax5.plot(np.arange(1, len(ir)+1), ir,
                 color=cmap(idx), linewidth=1.0, alpha=0.8)
ax5.set_xlabel("Cycle number")
ax5.set_ylabel("Internal resistance (Ω)")
ax5.set_title("Internal Resistance Growth Over Cycles")

# ── Plot 6: Temperature profile (cycle 1 of first cell) ──
ax6 = fig.add_subplot(3, 3, 7)
_temp_cyc_num = 50 if "50" in cell0["cycles"] else 2
cyc1_data = get_cycle(cell0, _temp_cyc_num)
ax6.plot(cyc1_data["t"] / 60, cyc1_data["T"], color="#0F6E56", linewidth=1.5)
ax6.set_xlabel("Time (min)")
ax6.set_ylabel("Temperature (°C)")
ax6.set_title(f"Temperature Profile — Cycle {_temp_cyc_num}")

# ── Plot 7: SOH at cycle 100 vs final cycle life ──
ax7 = fig.add_subplot(3, 3, 8)
soh100, life = [], []
for cell_id, cell in cells.items():
    q = cell["summary"]["QDischarge"]
    if len(q) >= 100:
        base = _baseline_capacity(q)
        if np.isfinite(base) and base > 0:
            soh100.append(q[99] / base * 100)
            life.append(cell["cycle_life"])
ax7.scatter(soh100, life, color="#534AB7", alpha=0.6, s=30, edgecolors="white", linewidths=0.3)
ax7.set_xlabel("SOH at cycle 100 (%)")
ax7.set_ylabel("Total cycle life")
ax7.set_title("Early SOH vs Final Cycle Life\n(key insight: early degradation predicts total life)")

# ── Plot 8: Charge time evolution ──
ax8 = fig.add_subplot(3, 3, 9)
for idx, cell_id in enumerate(sample_cells[:8]):
    cell = cells[cell_id]
    if "chargetime" in cell["summary"]:
        ct = cell["summary"]["chargetime"]
        ax8.plot(np.arange(1, len(ct)+1), ct,
                 color=cmap(idx), linewidth=1.0, alpha=0.8)
ax8.set_xlabel("Cycle number")
ax8.set_ylabel("Charge time (s)")
ax8.set_title("Charge Time Increases as Battery Ages")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "day1_eda.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[✓] Plot saved → {out_path}")

# ── BONUS PLOT: dQdV (IC curve) — only if mat file data ──────────
sample_cell = list(cells.values())[0]
cyc1_data   = sample_cell["cycles"].get("1", {})
if "dQdV" in cyc1_data and len(cyc1_data["dQdV"]) > 0:
    fig2, ax = plt.subplots(figsize=(9, 4))
    n_cyc    = len(sample_cell["cycles"])
    checkpts = [2, max(3, n_cyc//5), max(4, n_cyc*2//5),
                max(4, n_cyc*3//5), max(5, n_cyc*4//5)]
    cmap3    = plt.colormaps.get_cmap("plasma")
    colors   = [cmap3(i / len(checkpts)) for i in range(len(checkpts))]

    for idx, cp in enumerate(checkpts):
        cyc = sample_cell["cycles"].get(str(cp), {})
        if "dQdV" in cyc and "Qdlin" in cyc and len(cyc["dQdV"]) > 0:
            # Qdlin is on a linear voltage grid — use as x-axis
            ax.plot(cyc["Qdlin"], cyc["dQdV"],
                    color=colors[idx], linewidth=1.5, label=f"Cycle {cp}")

    ax.set_xlabel("Discharge capacity Qdlin (Ah)")
    ax.set_ylabel("dQ/dV  (IC curve)")
    ax.set_title("IC Curves at Different Cycle Stages\n"
                 "(peaks shrink/shift as battery ages — your key ECE feature)")
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ic_path = os.path.join(OUT_DIR, "day1_ic_curves.png")
    plt.tight_layout()
    plt.savefig(ic_path, dpi=150, bbox_inches="tight")
    print(f"[✓] IC curve plot saved → {ic_path}")
    print("    dQdV is already pre-computed in the dataset — Day 2 feature extraction is easier!")

# ─────────────────────────────────────────────────────
#  SAVE SUMMARY CSV
# ─────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "cell_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"[✓] Summary CSV saved → {csv_path}")

print("\n── TOP 5 LONGEST CELLS ───────────────────────")
print(summary_df.tail(5).to_string(index=False))
print("\n── TOP 5 SHORTEST CELLS ──────────────────────")
print(summary_df.head(5).to_string(index=False))

print("\n[DAY 1 COMPLETE] Next: day2_features.py — extract IC curves, ΔQ(V), dV/dQ")