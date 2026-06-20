"""
load_and_run.py
One script to run after placing your .mat files in data/raw/
Verifies loading then launches EDA.
"""

import os, sys
sys.path.insert(0, "src")
from mat_loader import load_all_mat_batches
from data_loader import get_summary_df, get_capacity_df

DATA_DIR = "data/raw"

# ── step 1: show what files are present ─────────────
print("Files in data/raw:")
for f in sorted(os.listdir(DATA_DIR)):
    mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6
    print(f"  {f}  ({mb:.1f} MB)")

# ── step 2: load ────────────────────────────────────
cells = load_all_mat_batches(DATA_DIR)

# ── step 3: print structure ─────────────────────────
sample_id   = list(cells.keys())[0]
sample_cell = cells[sample_id]
sample_cycle_num = "50" if "50" in sample_cell["cycles"] else list(sample_cell["cycles"].keys())[len(sample_cell["cycles"])//2]
cyc1        = sample_cell["cycles"][sample_cycle_num]

print(f"\n── SAMPLE CELL: {sample_id} ──────────────────")
print(f"  cycle_life    : {sample_cell['cycle_life']} cycles")
print(f"  charge_policy : {sample_cell['charge_policy']}")
print(f"  summary keys  : {list(sample_cell['summary'].keys())}")
print(f"  total cycles  : {len(sample_cell['cycles'])}")

print(f"\n── CYCLE {sample_cycle_num} ARRAYS ──────────────────────────")
for k, v in cyc1.items():
    if v.size == 0:
        print(f"  {k:8s}  shape={str(v.shape):12s}  (empty array)")
    else:
        print(f"  {k:8s}  shape={str(v.shape):12s}  "
              f"min={v.min():8.4f}  max={v.max():8.4f}")

# ── step 4: summary stats ───────────────────────────
df = get_summary_df(cells)
print(f"\n── DATASET STATS ───────────────────────────")
print(f"  Total cells      : {len(cells)}")
print(f"  Cycle life min   : {df.cycle_life.min()}")
print(f"  Cycle life max   : {df.cycle_life.max()}")
print(f"  Cycle life mean  : {df.cycle_life.mean():.0f}")

# ── step 5: check dQdV is present ───────────────────
has_dqdv = "dQdV" in cyc1 and len(cyc1["dQdV"]) > 0
has_qdlin = "Qdlin" in cyc1 and len(cyc1["Qdlin"]) > 0
print(f"\n── KEY FEATURES PRE-COMPUTED IN DATASET ────")
print(f"  dQdV  (IC curve)  : {'✓ present' if has_dqdv else '✗ missing'}"
      f"  shape={cyc1.get('dQdV', []).shape if has_dqdv else 'N/A'}")
print(f"  Qdlin (linear Qd) : {'✓ present' if has_qdlin else '✗ missing'}"
      f"  shape={cyc1.get('Qdlin', []).shape if has_qdlin else 'N/A'}")

print("\n✓ Load successful! Now run:  python day1_eda.py")