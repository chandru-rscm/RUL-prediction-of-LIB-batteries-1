"""
data_loader.py — MIT-Stanford + NASA battery dataset loader
Supports: .pkl (Kaggle/MIT), .mat (official MIT), NASA txt/mat format
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────
#  MIT-STANFORD LOADERS
# ─────────────────────────────────────────────

def load_mit_pkl(data_dir: str) -> dict:
    """
    Load MIT-Stanford batches from pickle files (Kaggle format).
    Put batch1.pkl, batch2.pkl, batch3.pkl in data_dir.
    """
    all_cells = {}
    for i in range(1, 4):
        path = os.path.join(data_dir, f"batch{i}.pkl")
        if not os.path.exists(path):
            print(f"  [skip] batch{i}.pkl not found")
            continue
        print(f"Loading batch{i}.pkl ...", end=" ")
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        print(f"{len(batch)} cells")
        all_cells.update(batch)
    print(f"\nTotal MIT cells loaded: {len(all_cells)}")
    return all_cells


def load_mit_mat(mat_path: str) -> dict:
    """
    Load MIT-Stanford from official .mat file using h5py.
    Use this if you downloaded from data.matr.io directly.
    """
    import h5py

    print(f"Loading {os.path.basename(mat_path)} via h5py ...")
    cells = {}

    with h5py.File(mat_path, "r") as f:
        batch_ref = f["batch"]
        n_cells   = batch_ref["summary"].shape[0]

        for i in tqdm(range(n_cells), desc="  Parsing cells"):
            cell = {}

            # — summary (one value per cycle) —
            cell["summary"] = {
                "QDischarge": np.array(f[batch_ref["summary"][i, 0]]["dischar"]).flatten(),
                "QCharge":    np.array(f[batch_ref["summary"][i, 0]]["charg"]).flatten(),
                "IR":         np.array(f[batch_ref["summary"][i, 0]]["IR"]).flatten(),
                "Tmax":       np.array(f[batch_ref["summary"][i, 0]]["Tmax"]).flatten(),
                "chargetime": np.array(f[batch_ref["summary"][i, 0]]["chargetime"]).flatten(),
            }

            # — cycle life (scalar) —
            cell["cycle_life"] = int(
                np.array(f[batch_ref["cycle_life"][i, 0]]).flatten()[0]
            )

            # — per-cycle time-series —
            cycles_ref = batch_ref["cycles"][i, 0]
            n_cycles   = f[cycles_ref]["V"].shape[0]
            cell["cycles"] = {}
            for c in range(n_cycles):
                cell["cycles"][str(c + 1)] = {
                    "V":  np.array(f[f[cycles_ref]["V"][c, 0]]).flatten(),
                    "Qd": np.array(f[f[cycles_ref]["Qd"][c, 0]]).flatten(),
                    "I":  np.array(f[f[cycles_ref]["I"][c, 0]]).flatten(),
                    "T":  np.array(f[f[cycles_ref]["T"][c, 0]]).flatten(),
                    "t":  np.array(f[f[cycles_ref]["t"][c, 0]]).flatten(),
                }

            cells[f"b{mat_path[-5]}c{i}"] = cell

    print(f"Total MIT cells loaded: {len(cells)}")
    return cells


# ─────────────────────────────────────────────
#  NASA LOADER
# ─────────────────────────────────────────────

def load_nasa_mat(nasa_dir: str) -> dict:
    """
    Load NASA battery dataset (.mat files: B0005.mat, B0006.mat, etc.)
    Download from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
    """
    import scipy.io

    cells = {}
    nasa_files = ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"]

    for fname in nasa_files:
        fpath = os.path.join(nasa_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] {fname} not found")
            continue

        print(f"Loading {fname} ...", end=" ")
        mat      = scipy.io.loadmat(fpath, simplify_cells=True)
        key      = fname.replace(".mat", "")   # e.g. "B0005"
        raw      = mat[key]["cycle"]
        cell_id  = key.lower()

        discharge_cycles = {}
        qdischarge       = []
        cyc_count        = 0

        for cyc in raw:
            if cyc["type"] != "discharge":
                continue
            cyc_count += 1
            data = cyc["data"]
            discharge_cycles[str(cyc_count)] = {
                "V":  np.array(data["Voltage_measured"]).flatten(),
                "I":  np.array(data["Current_measured"]).flatten(),
                "T":  np.array(data["Temperature_measured"]).flatten(),
                "t":  np.array(data["Time"]).flatten(),
                "Qd": np.array(data["Capacity"]).flatten(),
            }
            qdischarge.append(float(np.array(data["Capacity"]).flatten()[-1]))

        # EOL = first cycle capacity drops below 1.4 Ah (80% of 1.8 Ah nominal)
        eol = next(
            (i + 1 for i, q in enumerate(qdischarge) if q < 1.4),
            len(qdischarge)
        )

        cells[cell_id] = {
            "cycles":     discharge_cycles,
            "cycle_life": eol,
            "summary": {
                "QDischarge": np.array(qdischarge),
            },
        }
        print(f"{cyc_count} discharge cycles, EOL @ {eol}")

    print(f"\nTotal NASA cells loaded: {len(cells)}")
    return cells


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_summary_df(cells: dict) -> pd.DataFrame:
    """One-row-per-cell summary DataFrame."""
    rows = []
    for cell_id, cell in cells.items():
        rows.append({
            "cell_id":    cell_id,
            "cycle_life": int(cell["cycle_life"]),
            "n_cycles":   len(cell["cycles"]),
        })
    return pd.DataFrame(rows).sort_values("cycle_life").reset_index(drop=True)


def get_cycle(cell: dict, cycle_num: int) -> dict:
    """Return time-series dict for a specific cycle number (1-indexed)."""
    key = str(cycle_num)
    if key not in cell["cycles"]:
        raise KeyError(f"Cycle {cycle_num} not found (cell has {len(cell['cycles'])} cycles)")
    return cell["cycles"][key]


def get_capacity_df(cells: dict) -> pd.DataFrame:
    """Discharge capacity per cycle for every cell — for plotting degradation curves."""
    rows = []
    for cell_id, cell in cells.items():
        caps = cell["summary"]["QDischarge"]
        for cyc_idx, cap in enumerate(caps):
            rows.append({
                "cell_id":    cell_id,
                "cycle":      cyc_idx + 1,
                "capacity":   float(cap),
                "cycle_life": int(cell["cycle_life"]),
            })
    return pd.DataFrame(rows)