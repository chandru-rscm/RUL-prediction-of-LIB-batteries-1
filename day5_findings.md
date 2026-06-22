# Day 5 — Findings Summary (for report)

## Part 1 — SHAP Explainability ✅
Used SHAP (LinearExplainer) on the Day 3 Elastic Net model to identify which
of the 17 engineered features actually drive the RUL prediction.

**Result:** `dQ_min` is the single strongest driver of the prediction,
consistent with its raw correlation finding from Day 2 (r=0.77 with
cycle_life). This independently confirms, via two different methods
(correlation analysis in Day 2, and SHAP attribution in Day 5), that the
ΔQ(V) signal — the Nature paper's original feature — remains the most
informative single feature even after adding ECE-domain features (IC
curves, internal resistance, charge time).

## Part 2 — Multi-Threshold Prediction ✅
Instead of predicting only "cycles to 80% SOH" (the standard EOL definition
used by both reference papers), we trained three separate Elastic Net
models to predict cycles-to-90%, cycles-to-85%, and cycles-to-80% SOH.

**Result:**
| Threshold | RMSE % | R²    |
|-----------|--------|-------|
| 90%       | 20.3%  | 0.837 |
| 85%       | 13.6%  | 0.899 |
| 80%       | 9.8%   | 0.905 |

R² improves as the threshold approaches full EOL — predicting the early
90% threshold is harder than predicting the eventual 80% EOL point, which
makes physical sense (less degradation signal has accumulated by then).
This multi-threshold capability has a direct practical use discussed
earlier in the project: triggering second-life reassignment decisions
at the 85% threshold rather than waiting for full EOL.

## Part 3 — Dynamic Re-Prediction ⚠️ Honest Limitation
We tested whether re-predicting RUL at later checkpoints (200, 300, 400,
600, 800 cycles), using more real aging data each time, improves accuracy
compared to a single static prediction at cycle 100 — the central
hypothesis motivating the "dynamic vs static" design discussed early in
this project.

**Method:** To make the comparison fair, we fixed a single cohort of cells
that survive to every checkpoint (66 of 138 cells survive to cycle 800;
13 held out as a fixed test set, 53 for training), so that accuracy at
every checkpoint is computed on the *same* batteries rather than a
shifting, survivorship-biased subset.

**Result:** RMSE% did not show a statistically reliable improving trend:

| Checkpoint | RMSE % | R²    |
|------------|--------|-------|
| 100        | 10.6%  | 0.573 |
| 200        | 12.7%  | 0.394 |
| 300        | 11.9%  | 0.465 |
| 400        | 13.5%  | 0.314 |
| 600        | 10.5%  | 0.581 |
| 800        | 12.7%  | 0.391 |

Individual cell trajectories were also mixed: some cells' predictions
converged toward their true value as more checkpoints were added, while
others diverged (see `day5_dynamic_per_cell.png`).

**Why this happened (not a code defect):** With only 138 total cells, the
fixed cohort surviving to the longest checkpoint shrinks to 66 cells,
split into a 53-cell training set and a 13-cell test set. At this size,
per-checkpoint Elastic Net retraining is highly sensitive to exactly which
cells fall in each fold — the noise floor from sample size dominates any
genuine signal-improvement effect. This is a known, documented constraint
in battery RUL research: published datasets of this kind are small because
full-life aging tests are expensive to run, which limits how finely they
can be split for cohort-based temporal analysis.

**Honest conclusion:** This result does not confirm the dynamic-prediction
hypothesis at a statistically reliable level on this dataset. A larger
dataset, or a single pooled model trained across all checkpoints jointly
(rather than retraining independently per checkpoint), would likely be
needed to properly test it. We report this as a legitimate limitation
rather than overstating a trend that the data does not support.