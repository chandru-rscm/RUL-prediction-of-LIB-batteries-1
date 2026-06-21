"""
diagnose_summary.py — One-off diagnostic to inspect how 'summary' fields
are actually stored in the .mat file, so we can fix mat_loader.py correctly.

Run: python diagnose_summary.py
"""

import h5py
import numpy as np
import os

DATA_DIR = "data/raw"

# find the first batch file
fname = None
for f in os.listdir(DATA_DIR):
    if f.endswith(".mat"):
        fname = f
        break

fpath = os.path.join(DATA_DIR, fname)
print(f"Inspecting: {fname}\n")

with h5py.File(fpath, "r") as f:
    batch = f["batch"]
    s_ref  = batch["summary"][0, 0]   # first cell's summary
    s_node = f[s_ref]

    print(f"Summary node keys: {list(s_node.keys())}\n")

    for key in s_node.keys():
        node = s_node[key]
        print(f"── {key} ──")
        print(f"  type      : {type(node)}")
        print(f"  shape     : {node.shape}")
        print(f"  dtype     : {node.dtype}")

        try:
            raw = np.array(node)
            print(f"  raw shape : {raw.shape}, raw dtype: {raw.dtype}")
            print(f"  sample    : {raw.flatten()[:3]}")

            # check if it's a reference array
            if raw.dtype == object or h5py.check_dtype(ref=raw.dtype) is not None:
                print(f"  -> looks like a REFERENCE array, needs dereferencing")
                first_ref = raw.flatten()[0]
                deref = np.array(f[first_ref]).flatten()
                print(f"  -> dereferenced[0] shape: {deref.shape}, sample: {deref[:3]}")
            else:
                print(f"  -> looks like DIRECT numeric data, shape[1]={node.shape[1] if len(node.shape)>1 else 'N/A'}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()