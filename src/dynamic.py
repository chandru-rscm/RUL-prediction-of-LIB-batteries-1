"""
src/dynamic.py — Day 5 dynamic re-prediction logic.

The whole point (recap from way earlier in the project): a static model predicts
once at cycle 100 and freezes. A dynamic model re-predicts at later checkpoints
using more real aging data each time, and should get MORE accurate as the
battery ages further — exactly like GPS rerouting as new traffic data comes in.

This reuses Day 2/4a's existing per-checkpoint feature extraction
(extract_cell_features) rather than inventing a new pipeline — same features,
just evaluated at extra checkpoints beyond the original 5 (10/30/50/70/100).

IMPORTANT — cell survivorship at later checkpoints:
Not every cell lives long enough to reach every checkpoint (e.g. a cell with
cycle_life=300 has no real cycle-600 data). We only evaluate a cell at a
checkpoint if it actually survived past that checkpoint. This means the test
set naturally shrinks at later checkpoints — that's expected and we report it
explicitly rather than hiding it.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

# ElasticNetCV can print convergence warnings on small/scaled datasets like ours
# (138 cells, 17 features) — harmless here since results are stable across runs,
# but floods the terminal. Suppressed at the module level so day5_explain.py
# doesn't need a -W ignore flag.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def build_dynamic_checkpoint_features(cells: dict, checkpoints: list,
                                       extract_cell_features_fn,
                                       min_survival_margin: int = 5,
                                       early: int = 10,
                                       verbose: bool = True):
    """
    For each checkpoint in `checkpoints`, extract features for every cell that
    survived at least `min_survival_margin` cycles past that checkpoint
    (so we're not extracting features right at end-of-life, which would leak
    information about imminent death into the "early" prediction).

    extract_cell_features_fn is called as:
        extract_cell_features_fn(cell, checkpoint=cp, early=early)
    matching src/features.py's real signature, and is expected to return
    a DICT of {feature_name: value, ..., "cycle_life": int} or None — NOT
    a (features, names) tuple. We strip "cycle_life" and "cell_id" out of
    the dict ourselves and use the cell's own dict entry as ground truth.

    DEAD COLUMNS: this dataset's chemistry (LFP/graphite) shows only ONE
    dominant IC curve peak, so extract_ic_peak_features always returns NaN
    for ic_peak2_voltage and ic_peak2_width (no second peak exists to
    measure). Day 2/explain.py already drop these before modelling — we do
    the same here, otherwise every single cell would get rejected by the
    NaN check below purely because of these two permanently-empty columns
    (this was the actual cause of the earlier "0 cells survive" bug).

    verbose=True prints the first few real errors/None-returns per checkpoint
    instead of silently skipping everything.

    Returns: dict {checkpoint: (feature_matrix, cycle_life_array, cell_ids, feature_names)}
    """
    DEAD_COLS = {"ic_peak2_height", "ic_peak2_voltage", "ic_peak2_width"}

    out = {}
    for cp in checkpoints:
        rows, targets, ids = [], [], []
        feature_names = None
        n_too_short = 0
        n_none_returned = 0
        n_exceptions = 0
        first_errors = []

        for cell_id, cell in cells.items():
            if cell["cycle_life"] < cp + min_survival_margin:
                n_too_short += 1
                continue
            try:
                feats_dict = extract_cell_features_fn(cell, checkpoint=cp, early=early)
            except Exception as e:
                n_exceptions += 1
                if len(first_errors) < 3:
                    first_errors.append(f"{cell_id}: {type(e).__name__}: {e}")
                continue

            if feats_dict is None:
                n_none_returned += 1
                continue

            # strip non-feature keys AND known dead columns, keep a stable,
            # consistent column order
            names = sorted(
                k for k in feats_dict.keys()
                if k not in ("cycle_life", "cell_id") and k not in DEAD_COLS
            )
            vec = np.array([feats_dict[k] for k in names], dtype=np.float64)

            if np.any(np.isnan(vec)):
                n_none_returned += 1  # treat NaN features same as "no usable data"
                continue

            rows.append(vec)
            targets.append(cell["cycle_life"])
            ids.append(cell_id)
            feature_names = names

        if verbose:
            print(f"  checkpoint {cp}: {len(ids)} usable | "
                  f"{n_too_short} too-short, {n_none_returned} no-data/NaN, "
                  f"{n_exceptions} errors")
            if first_errors:
                print(f"    sample errors: {first_errors}")

        if len(rows) == 0:
            out[cp] = (np.zeros((0, 0)), np.zeros(0), [], [])
            continue

        out[cp] = (np.array(rows), np.array(targets), ids, feature_names)
    return out


def get_fixed_dynamic_cohort(checkpoint_data: dict, candidate_test_ids: set):
    """
    Returns the subset of candidate_test_ids (and, separately, of all OTHER
    cells) that survive every single checkpoint in checkpoint_data — i.e. a
    cohort we can legitimately track across the whole dynamic re-prediction
    series without the group composition silently changing.

    Without this, evaluate_dynamic_predictions's test set shrinks at later
    checkpoints (e.g. 28 -> 17 -> 13) because shorter-lived batteries drop
    out — so "error at checkpoint 800" was being computed on a different,
    survivorship-biased group than "error at checkpoint 100". That makes
    the checkpoints incomparable: it's not the same batteries getting a
    sharper prediction, it's an increasingly easy (long-lived-only) subset.

    This pins down ONE fixed test cohort (cells alive at the LAST/longest
    checkpoint) so every checkpoint's accuracy number describes the exact
    same group of batteries, predicted with more or less data.
    """
    ids_per_checkpoint = [set(ids) for (_, _, ids, _) in checkpoint_data.values()]
    cells_alive_everywhere = set.intersection(*ids_per_checkpoint) if ids_per_checkpoint else set()

    fixed_test_ids = candidate_test_ids & cells_alive_everywhere
    fixed_train_pool = cells_alive_everywhere - fixed_test_ids
    return fixed_test_ids, fixed_train_pool, cells_alive_everywhere


def evaluate_dynamic_predictions(checkpoint_data: dict, test_cell_ids: set,
                                  random_state: int = 42, fixed_cohort: bool = True):
    """
    For each checkpoint, train an Elastic Net and predict on test cells.
    Reports accuracy at each checkpoint so we can show the "prediction gets
    sharper over time" effect.

    fixed_cohort=True (recommended, default): test_cell_ids is first
    intersected with the set of cells that survive EVERY checkpoint, and
    that exact same group is used as the test set at every checkpoint —
    so n_test stays constant across the whole table and the comparison is
    honest (same batteries, more data each time). The training pool is
    similarly fixed to cells alive everywhere, so n_train is also constant;
    only the FEATURES change per checkpoint, isolating "does more data
    improve the prediction" from "did the test/train group change".

    fixed_cohort=False: reproduces the old (less rigorous) behaviour where
    each checkpoint uses whichever cells happen to survive it — test set
    size can shrink at later checkpoints. Kept available for comparison/
    discussion in the report (e.g. "naive vs cohort-corrected" finding).

    test_cell_ids: the fixed set of cell IDs originally held out as test
    cells (from Day 3's split) — used as the starting candidate pool either way.
    """
    rows = []
    per_cell_predictions = {}  # cell_id -> {checkpoint: (true, pred)}

    if fixed_cohort:
        fixed_test_ids, fixed_train_pool, alive_everywhere = get_fixed_dynamic_cohort(
            checkpoint_data, test_cell_ids
        )
        print(f"  Fixed cohort: {len(alive_everywhere)} cells survive ALL checkpoints "
              f"({len(fixed_test_ids)} test, {len(fixed_train_pool)} train)")
        if len(fixed_test_ids) < 3 or len(fixed_train_pool) < 10:
            print("  [WARNING] Fixed cohort too small for the longest checkpoints — "
                  "consider shortening DYNAMIC_CHECKPOINTS in day5_explain.py.")

    for cp, (X, y, ids, feature_names) in checkpoint_data.items():
        if len(ids) == 0:
            continue
        ids_arr = np.array(ids)

        if fixed_cohort:
            is_test = np.array([cid in fixed_test_ids for cid in ids_arr])
            is_train = np.array([cid in fixed_train_pool for cid in ids_arr])
        else:
            is_test = np.array([cid in test_cell_ids for cid in ids_arr])
            is_train = ~is_test

        if is_train.sum() < 10 or is_test.sum() < 3:
            # not enough data on either side to train/evaluate meaningfully
            continue

        X_train, y_train = X[is_train], y[is_train]
        X_test, y_test = X[is_test], y[is_test]
        test_ids_here = ids_arr[is_test]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-2, 2, 50),
            cv=min(5, is_train.sum()),
            max_iter=50000,
            tol=1e-3,
            random_state=random_state,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_pct = rmse / np.mean(y_test) * 100
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan

        rows.append({
            "checkpoint": cp,
            "n_train": int(is_train.sum()),
            "n_test": int(is_test.sum()),
            "mae": mae, "rmse": rmse, "rmse_pct": rmse_pct, "r2": r2,
        })

        for cid, yt, yp in zip(test_ids_here, y_test, y_pred):
            per_cell_predictions.setdefault(cid, {})[cp] = (float(yt), float(yp))

    metrics_df = pd.DataFrame(rows)
    if len(metrics_df) > 0:
        metrics_df = metrics_df.sort_values("checkpoint").reset_index(drop=True)
    return metrics_df, per_cell_predictions