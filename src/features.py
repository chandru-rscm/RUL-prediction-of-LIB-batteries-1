"""
features.py — Day 2: Feature Engineering
Extracts ECE-domain features from raw battery cycle data:
  - DeltaQ(V)   : discharge curve difference between early & later cycle (Nature paper's feature)
  - IC curve    : dQ/dV computed via numpy gradient (since dataset's dQdV field is empty)
  - Peak stats  : height, position, width of IC curve peaks
  - IR trend    : internal resistance growth
  - Charge time : time to fully charge, per cycle

All features computed AT a checkpoint cycle (default: cycle 100) so we can
directly compare against the Nature paper's 9.1% error benchmark.
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.signal import find_peaks, savgol_filter

# Voltage grid used for Qdlin/Tdlin in the MIT-Stanford dataset
# (matches observed cycle voltage range: ~2.0V to ~3.6V)
V_GRID = np.linspace(3.6, 2.0, 1000)


# ─────────────────────────────────────────────────────
#  CORE FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────

def compute_dqdv(Qdlin: np.ndarray, v_grid: np.ndarray = V_GRID,
                  smooth: bool = True) -> np.ndarray:
    """
    Compute the IC curve (dQ/dV) from linearised discharge capacity.
    Qdlin is already on a fixed voltage grid, so we just need the gradient.

    smooth=True applies Savitzky-Golay filtering — IC curves are noisy
    and need smoothing to get clean peaks (this is where MATLAB's
    smooth()/sgolayfilt() would normally be used; scipy gives an
    equivalent here).
    """
    if Qdlin is None or len(Qdlin) < 5:
        return np.array([])

    dqdv = np.gradient(Qdlin, v_grid)

    if smooth:
        window = min(51, len(dqdv) - (1 - len(dqdv) % 2))  # must be odd, < len
        if window >= 5:
            dqdv = savgol_filter(dqdv, window_length=window, polyorder=3)

    return dqdv


def extract_ic_peak_features(dqdv: np.ndarray, v_grid: np.ndarray = V_GRID,
                              n_peaks: int = 2) -> dict:
    """
    Extract peak height, position, and width from an IC curve.
    IC curves for LFP cells typically show 1-2 dominant peaks.
    """
    feats = {}
    if dqdv is None or len(dqdv) == 0:
        for i in range(1, n_peaks + 1):
            feats[f"ic_peak{i}_height"] = np.nan
            feats[f"ic_peak{i}_voltage"] = np.nan
            feats[f"ic_peak{i}_width"]   = np.nan
        return feats

    # IC curves for discharge are typically negative (capacity decreasing
    # as voltage decreases) — look at the magnitude
    abs_dqdv = np.abs(dqdv)

    peak_idx, properties = find_peaks(abs_dqdv, prominence=np.std(abs_dqdv) * 0.3,
                                       width=1)

    # sort peaks by height, descending
    if len(peak_idx) > 0:
        order = np.argsort(abs_dqdv[peak_idx])[::-1]
        peak_idx = peak_idx[order]

    for i in range(1, n_peaks + 1):
        if i - 1 < len(peak_idx):
            idx = peak_idx[i - 1]
            feats[f"ic_peak{i}_height"]  = float(abs_dqdv[idx])
            feats[f"ic_peak{i}_voltage"] = float(v_grid[idx])
            feats[f"ic_peak{i}_width"]   = float(properties["widths"][np.where(peak_idx == idx)[0][0]]) \
                                            if "widths" in properties else np.nan
        else:
            feats[f"ic_peak{i}_height"]  = 0.0
            feats[f"ic_peak{i}_voltage"] = np.nan
            feats[f"ic_peak{i}_width"]   = np.nan

    return feats


def compute_delta_q_features(qdlin_early: np.ndarray, qdlin_late: np.ndarray) -> dict:
    """
    Nature paper's core feature: Delta-Q(V) = Q_late(V) - Q_early(V)
    Returns summary statistics of this difference curve.
    """
    if qdlin_early is None or qdlin_late is None or \
       len(qdlin_early) == 0 or len(qdlin_late) == 0:
        return {
            "dQ_min": np.nan, "dQ_var": np.nan,
            "dQ_skew": np.nan, "dQ_kurtosis": np.nan,
            "dQ_mean": np.nan,
        }

    diff = qdlin_late - qdlin_early
    diff = diff[np.isfinite(diff)]

    if len(diff) < 5:
        return {
            "dQ_min": np.nan, "dQ_var": np.nan,
            "dQ_skew": np.nan, "dQ_kurtosis": np.nan,
            "dQ_mean": np.nan,
        }

    return {
        "dQ_min":      float(np.min(diff)),
        "dQ_var":      float(np.var(diff)),
        "dQ_skew":     float(scipy_stats.skew(diff)),
        "dQ_kurtosis": float(scipy_stats.kurtosis(diff)),
        "dQ_mean":     float(np.mean(diff)),
    }


def compute_ir_features(ir_array: np.ndarray, checkpoint: int) -> dict:
    """Internal resistance value and growth rate up to the checkpoint cycle."""
    if ir_array is None or len(ir_array) < checkpoint:
        return {"ir_at_checkpoint": np.nan, "ir_slope": np.nan}

    ir_slice = ir_array[:checkpoint]
    ir_slice = ir_slice[np.isfinite(ir_slice)]
    if len(ir_slice) < 5:
        return {"ir_at_checkpoint": np.nan, "ir_slope": np.nan}

    x = np.arange(len(ir_slice))
    slope = float(np.polyfit(x, ir_slice, 1)[0])

    return {
        "ir_at_checkpoint": float(ir_slice[-1]),
        "ir_slope":         slope,
    }


def compute_chargetime_features(ct_array: np.ndarray, checkpoint: int) -> dict:
    """Charge time value and trend up to checkpoint."""
    if ct_array is None or len(ct_array) < checkpoint:
        return {"chargetime_at_checkpoint": np.nan, "chargetime_slope": np.nan}

    ct_slice = ct_array[:checkpoint]
    ct_slice = ct_slice[np.isfinite(ct_slice)]
    if len(ct_slice) < 5:
        return {"chargetime_at_checkpoint": np.nan, "chargetime_slope": np.nan}

    x = np.arange(len(ct_slice))
    slope = float(np.polyfit(x, ct_slice, 1)[0])

    return {
        "chargetime_at_checkpoint": float(ct_slice[-1]),
        "chargetime_slope":         slope,
    }


def compute_capacity_fade_features(qd_summary: np.ndarray, checkpoint: int) -> dict:
    """Capacity fade rate over the last 20 cycles up to checkpoint."""
    if qd_summary is None or len(qd_summary) < checkpoint:
        return {"soh_at_checkpoint": np.nan, "fade_rate_recent": np.nan}

    qd_slice = qd_summary[:checkpoint]
    qd_slice = qd_slice[np.isfinite(qd_slice)]
    if len(qd_slice) < 25:
        return {"soh_at_checkpoint": np.nan, "fade_rate_recent": np.nan}

    # index 0 = cycle 1, often a near-empty formation cycle (value ~0) —
    # use a stable early baseline instead (index 4 ~ cycle 5) to avoid
    # dividing by zero / using a non-representative reading
    baseline_idx = min(4, len(qd_slice) - 1)
    baseline = qd_slice[baseline_idx]
    if baseline <= 0:
        positive = qd_slice[qd_slice > 0]
        baseline = positive[0] if len(positive) > 0 else np.nan

    soh = (qd_slice[-1] / baseline * 100) if (baseline and baseline > 0 and np.isfinite(baseline)) else np.nan

    recent = qd_slice[-20:]
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])

    return {
        "soh_at_checkpoint": float(soh) if np.isfinite(soh) else np.nan,
        "fade_rate_recent":  slope,
    }


# ─────────────────────────────────────────────────────
#  MAIN FEATURE EXTRACTION FOR ONE CELL
# ─────────────────────────────────────────────────────

def extract_cell_features(cell: dict, checkpoint: int = 100, early: int = 10) -> dict:
    """
    Extract the full feature vector for one cell, evaluated at `checkpoint` cycle.
    Returns None if the cell doesn't have enough cycles or data is missing.
    """
    if str(checkpoint) not in cell["cycles"] or str(early) not in cell["cycles"]:
        return None

    early_cyc = cell["cycles"].get(str(early), {})
    late_cyc  = cell["cycles"].get(str(checkpoint), {})

    qdlin_early = early_cyc.get("Qdlin", np.array([]))
    qdlin_late  = late_cyc.get("Qdlin", np.array([]))

    if len(qdlin_early) == 0 or len(qdlin_late) == 0:
        return None

    feats = {}

    # 1 — Delta-Q(V) features (Nature paper baseline)
    feats.update(compute_delta_q_features(qdlin_early, qdlin_late))

    # 2 — IC curve features (computed from Qdlin via gradient — your ECE addition)
    dqdv_late = compute_dqdv(qdlin_late)
    feats.update(extract_ic_peak_features(dqdv_late, n_peaks=2))

    # 3 — Internal resistance trend
    feats.update(compute_ir_features(cell["summary"].get("IR", np.array([])), checkpoint))

    # 4 — Charge time trend
    feats.update(compute_chargetime_features(cell["summary"].get("chargetime", np.array([])), checkpoint))

    # 5 — Capacity fade / SOH
    feats.update(compute_capacity_fade_features(cell["summary"].get("QDischarge", np.array([])), checkpoint))

    # target variable
    feats["cycle_life"] = int(cell["cycle_life"])

    return feats


def build_feature_dataframe(cells: dict, checkpoint: int = 100, early: int = 10):
    """
    Loop through all cells, extract features, return a pandas DataFrame.
    Skips cells with insufficient data.
    """
    import pandas as pd

    rows = []
    skipped = 0

    for cell_id, cell in cells.items():
        try:
            feats = extract_cell_features(cell, checkpoint=checkpoint, early=early)
            if feats is None:
                skipped += 1
                continue
            feats["cell_id"] = cell_id
            rows.append(feats)
        except Exception as e:
            print(f"  [skip] {cell_id} — feature extraction error: {e}")
            skipped += 1

    df = pd.DataFrame(rows)
    if len(df) > 0:
        cols = ["cell_id"] + [c for c in df.columns if c != "cell_id"]
        df = df[cols]

    print(f"\nFeature extraction done: {len(df)} cells with features, {skipped} skipped")
    return df


# ─────────────────────────────────────────────────────
#  SEQUENCE CACHING — for Day 3+ LSTM/CNN models
#  (extracted once now while cells are in memory, so we
#   never have to reload the huge .mat files again)
# ─────────────────────────────────────────────────────

def build_sequence_cache(cells: dict, checkpoint: int = 100,
                          curve_len: int = 100) -> dict:
    """
    For every valid cell, cache:
      - qdlin_curve   : the Qdlin discharge curve (1000 pts) AT the checkpoint
                         cycle — a real signal shape an LSTM/CNN can learn from
      - capacity_curve: QDischarge fade trend for the first `curve_len` cycles
                         (padded/truncated) — an alternative sequence input
      - cycle_life    : target value

    Returns a dict ready to be saved with np.savez(**dict).
    """
    cell_ids   = []
    qdlin_list = []
    cap_list   = []
    life_list  = []
    skipped    = 0

    for cell_id, cell in cells.items():
        cyc = cell["cycles"].get(str(checkpoint), {})
        qdlin = cyc.get("Qdlin", np.array([]))
        qd_summary = cell["summary"].get("QDischarge", np.array([]))

        if len(qdlin) == 0 or len(qd_summary) < curve_len:
            skipped += 1
            continue

        # pad/truncate capacity curve to fixed length
        cap_curve = qd_summary[:curve_len].astype(np.float32)

        cell_ids.append(cell_id)
        qdlin_list.append(qdlin.astype(np.float32))
        cap_list.append(cap_curve)
        life_list.append(int(cell["cycle_life"]))

    print(f"Sequence cache: {len(cell_ids)} cells cached, {skipped} skipped")

    return {
        "cell_ids":        np.array(cell_ids),
        "qdlin_curve":     np.array(qdlin_list, dtype=np.float32),   # (N, 1000)
        "capacity_curve":  np.array(cap_list, dtype=np.float32),     # (N, curve_len)
        "cycle_life":      np.array(life_list, dtype=np.int32),      # (N,)
    }