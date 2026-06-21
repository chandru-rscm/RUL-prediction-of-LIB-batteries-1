"""
mat_loader.py — Load official MIT-Stanford .mat files from data.matr.io
These are MATLAB v7.3 HDF5 format — use h5py, NOT scipy.io

Usage:
    from src.mat_loader import load_all_mat_batches
    cells = load_all_mat_batches("data/raw")

File naming — put files in data/raw/ with these exact names:
    batch1.mat  ← rename from 2017-05-12_batchdata_updated_struct_errorcorrect.mat
    batch2.mat  ← rename from 2017-06-30_batchdata_updated_struct_errorcorrect.mat
    batch3.mat  ← rename from 2018-04-12_batchdata_updated_struct_errorcorrect.mat

OR keep original names — the loader accepts both.
"""

import os
import h5py
import numpy as np
from tqdm import tqdm


# map original filenames → batch number
_ORIGINAL_NAMES = {
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat": 1,
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat": 2,
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat": 3,
}


def _arr(f, ref):
    """Dereference an HDF5 object reference and return flat numpy array."""
    return np.array(f[ref]).flatten()


def _load_single_mat(mat_path: str, batch_num: int) -> dict:
    """
    Load one MIT-Stanford .mat batch file.

    Returns dict keyed by cell_id (e.g. 'b1c0', 'b1c1', ...)
    Each cell contains:
        cycle_life    : int — total cycles until 80% EOL
        charge_policy : str — charging policy string
        summary       : dict of 1D arrays (one value per cycle)
            QDischarge, QCharge, IR, Tmax, Tmin, Tavg, chargetime, cycle
        cycles        : dict keyed by cycle number string ('1', '2', ...)
            Each cycle:
                V     — voltage array
                I     — current array
                Qc    — charge capacity array
                Qd    — discharge capacity array
                Qdlin — discharge capacity on LINEAR voltage grid (1000 pts)
                T     — temperature array
                Tdlin — temperature on linear voltage grid
                dQdV  — IC curve (already computed!) on linear voltage grid
                t     — time array
    """
    cells = {}
    print(f"\nLoading batch {batch_num}: {os.path.basename(mat_path)}")

    with h5py.File(mat_path, "r") as f:
        batch     = f["batch"]
        n_cells   = batch["summary"].shape[0]
        print(f"  Found {n_cells} cells")

        for i in tqdm(range(n_cells), desc=f"  Batch {batch_num}"):
            cell_id = f"b{batch_num}c{i}"

            try:
                # ── cycle life ──────────────────────────────
                cycle_life_raw = _arr(f, batch["cycle_life"][i, 0])[0]
                if np.isnan(cycle_life_raw):
                    print(f"  [skip] {cell_id} — cycle_life is NaN (known bad cell)")
                    continue
                cycle_life = int(cycle_life_raw)

                # ── charge policy ───────────────────────────
                try:
                    policy_raw = f[batch["policy_readable"][i, 0]][()]
                    policy     = "".join(chr(int(c)) for c in policy_raw.flatten())
                except Exception:
                    policy = "unknown"

                # ── summary (one value per cycle) ───────────
                s_ref  = batch["summary"][i, 0]
                s_node = f[s_ref]

                def _s(key):
                    try:
                        return np.array(s_node[key]).flatten()
                    except Exception:
                        return np.array([])

                summary = {
                    "QDischarge": _s("QDischarge"),
                    "QCharge":    _s("QCharge"),
                    "IR":         _s("IR"),
                    "Tmax":       _s("Tmax"),
                    "Tmin":       _s("Tmin"),
                    "Tavg":       _s("Tavg"),
                    "chargetime": _s("chargetime"),
                    "cycle":      _s("cycle"),
                }

                # ── per-cycle time series ────────────────────
                c_ref   = batch["cycles"][i, 0]
                c_node  = f[c_ref]
                n_cycles = c_node["V"].shape[0]
                cycles  = {}

                for j in range(n_cycles):
                    def _c(key):
                        try:
                            return _arr(f, c_node[key][j, 0])
                        except Exception:
                            return np.array([])

                    cycles[str(j + 1)] = {
                        "V":     _c("V"),
                        "I":     _c("I"),
                        "Qc":    _c("Qc"),
                        "Qd":    _c("Qd"),
                        "Qdlin": _c("Qdlin"),   # ← linearised capacity (1000 pts)
                        "T":     _c("T"),
                        "Tdlin": _c("Tdlin"),
                        "dQdV":  _c("dQdV"),    # ← IC curve ALREADY computed!
                        "t":     _c("t"),
                    }

                cells[cell_id] = {
                    "cycle_life":    cycle_life,
                    "charge_policy": policy,
                    "summary":       summary,
                    "cycles":        cycles,
                }

            except Exception as e:
                print(f"  [skip] {cell_id} — error: {e}")
                continue

    print(f"  ✓ Loaded {len(cells)} cells from batch {batch_num}")
    return cells


def load_all_mat_batches(data_dir: str) -> dict:
    """
    Auto-detect and load all 3 MIT mat batches from data_dir.
    Accepts both original long names and short names (batch1.mat etc.)
    """
    all_cells = {}
    files_found = []

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not fname.endswith(".mat"):
            continue

        # determine batch number
        if fname in _ORIGINAL_NAMES:
            batch_num = _ORIGINAL_NAMES[fname]
        elif fname.startswith("batch") and fname[5].isdigit():
            batch_num = int(fname[5])
        else:
            print(f"  [skip] unrecognised file: {fname}")
            continue

        files_found.append((batch_num, fpath))

    if not files_found:
        raise FileNotFoundError(
            f"No .mat files found in '{data_dir}'.\n"
            "Expected files like:\n"
            "  2017-05-12_batchdata_updated_struct_errorcorrect.mat\n"
            "  2017-06-30_batchdata_updated_struct_errorcorrect.mat\n"
            "  2018-04-12_batchdata_updated_struct_errorcorrect.mat\n"
            "OR renamed as batch1.mat / batch2.mat / batch3.mat"
        )

    for batch_num, fpath in sorted(files_found):
        batch_cells = _load_single_mat(fpath, batch_num)
        all_cells.update(batch_cells)

    print(f"\n{'='*45}")
    print(f"  TOTAL CELLS LOADED : {len(all_cells)}")
    print(f"{'='*45}")
    return all_cells


# ── quick verify script ──────────────────────────────
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    print("Verifying .mat files in:", data_dir)
    print("Files found:")
    for f in os.listdir(data_dir):
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / 1e6
        print(f"  {f}  ({size_mb:.1f} MB)")

    cells = load_all_mat_batches(data_dir)

    # print one cell's structure
    sample_id   = list(cells.keys())[0]
    sample_cell = cells[sample_id]

    print(f"\n── SAMPLE CELL: {sample_id} ──")
    print(f"  Cycle life        : {sample_cell['cycle_life']}")
    print(f"  Charge policy     : {sample_cell['charge_policy']}")
    print(f"  Summary keys      : {list(sample_cell['summary'].keys())}")
    print(f"  QDischarge length : {len(sample_cell['summary']['QDischarge'])}")
    print(f"  Total cycles      : {len(sample_cell['cycles'])}")

    cyc1 = sample_cell["cycles"]["1"]
    print(f"\n── CYCLE 1 OF {sample_id} ──")
    for k, v in cyc1.items():
        print(f"  {k:8s} : shape {v.shape}  range [{v.min():.4f}, {v.max():.4f}]")

    print("\n✓ All good — data/raw is ready for day1_eda.py")