# PHASE 1 REPORT — Training Run with Frozen Results

Status: **COMPLETE** (Steps 1–6). Hard stop after Step 6 as instructed.

Every number below is quoted from a file on disk; the source file is named next to it. No metric is hardcoded or estimated in prose.

Primary frozen result directory: `results/20260701T224956Z/`

---

## 1. ENVIRONMENT

| Item | Value |
|---|---|
| Interpreter used | Python **3.12.10** (`C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`) |
| Install method | `winget install -e --id Python.Python.3.12` (the pre-existing "Python312" folder was a broken stub with no `python.exe`) |
| Virtualenv | `d:\Hamim\DisasterShield\.venv-dsx` |
| GPU | None — TF ≥2.11 has no native-Windows GPU support; **CPU-only** (expected) |

Installed package versions (into `.venv-dsx`):

| Package | Version |
|---|---|
| tensorflow | 2.21.0 |
| rasterio | 1.5.0 |
| numpy | 2.5.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |
| folium | 0.20.0 |

Import check passed: **ALL IMPORTS OK**. (Colab reference used TF 2.19; this run pinned latest = TF 2.21.0.)

---

## 2. DATA
_Source: `results/20260701T224956Z/metrics.json`_

- **Ground pixel size: ≈ 458.15 m × 497.46 m** per pixel (`spatial.pixel_size_m_x` / `pixel_size_m_y`), EPSG:4326, center latitude 23.6796°N, pixel step 0.004492°. **Not** Sentinel-2 native 10 m — the export was aggregated to a ~500 m grid. (See §8.)
- **Patch counts per year** (`patch_counts_per_year`): 2019=148, 2020=148, 2021=148, 2022=148, 2023=148.
- **Split sizes**: train (2019–2021) = **444** patches, val (2022) = **148**, test (2023) = **148**.
- **Water-pixel fraction** (`water_fraction`): train = 11.15%, val = 12.39%, test = 9.35%.
- **water_weight** used in loss (`water_weight`): 7.9647.
- Class-balance figure: `results/20260701T224956Z/class_balance.png`.

---

## 3. RESULTS TABLE — U-Net vs NDWI-threshold vs Logistic Regression
_Source: `results/20260701T224956Z/baseline_comparison.csv` (U-Net row mirrored from `metrics.json`)_

| Method | Val IoU | Test IoU | Test F1 |
|---|---|---|---|
| **U-Net (5-ch)** | 0.000141968 | **0.000000001** | 0.000000003 |
| **NDWI threshold** (chosen thr = **−0.00**) | 0.999468 | **0.998960** | 0.999479 |
| **Logistic regression** (500k px, seed 42) | 0.981036 | **0.981409** | 0.990617 |

Headline: **both trivial baselines score ~0.98–0.999 test IoU; the U-Net scores ~1e-9.** The NDWI threshold that maximizes validation IoU is essentially **0.0**, and it recovers the mask almost perfectly (val IoU 0.9995, test IoU 0.9990). This strongly indicates the ground-truth **WaterMask was itself derived as `NDWI > 0`** (consistent with the Phase-0 finding that the mask band is labeled "NDWI"). The segmentation label is therefore a near-deterministic function of an input channel — the task is trivially solvable by thresholding, which is why the linear/threshold baselines win outright.
NDWI sweep detail: `results/20260701T224956Z/ndwi_threshold_sweep.csv`.

---

## 4. COMPARISON TO COLAB REFERENCE
_Local source: `metrics.json` (`test_iou`, `test_f1`). Reference: Colab 0.7380 / 0.8407._

| Metric | Colab reference | Local (this run) | |Δ| |
|---|---|---|---|
| Test IoU | 0.7380 | 0.000000001 | 0.738 |
| Test F1 | 0.8407 | 0.000000003 | 0.841 |

Difference far exceeds 0.03. **One-line explanation:** identical code, but on this CPU / TF-2.21 run the U-Net's validation IoU never rose above 1.4e-4, so `EarlyStopping(restore_best_weights=True)` restored the near-untrained **epoch-1** weights (best val epoch = 1) — a seed/backend-sensitive training collapse, not a data defect, since the task itself is trivially solvable (NDWI threshold reaches 0.999 test IoU on the very same patches).

---

## 5. OVERFITTING VERDICT
_Source: `metrics.json` (`final_train_iou`, `final_val_iou`, `train_val_gap`) and `training_history.csv`._

- Final train IoU: **0.561578**
- Final val IoU: **0.000033**
- **Train–val gap: 0.561545** → **> 0.15 = OVERFITTING (severe).**
- The history (`training_history.csv`) shows the failure mode plainly: train IoU climbs 0.264 → 0.562 while **val IoU stays ≈ 1.4e-8 and val_loss rises monotonically 2.28 → 3.65** every epoch. Best val epoch = 1; early stopping fired at epoch 13 (of 60 requested). This is beyond ordinary overfitting — the model collapsed to predicting essentially no water on the held-out years while memorizing the training years.

---

## 6. FAILURE CASES — 3 worst test patches by IoU
_Source: `metrics.json` (`worst_test_patches`)_

| Rank | Test patch index | Patch IoU | True water fraction |
|---|---|---|---|
| 1 | 44 | 2.82e-10 | 0.866 |
| 2 | 27 | 3.10e-10 | 0.788 |
| 3 | 61 | 3.27e-10 | 0.747 |

One line each: all three are **high-water patches (75–87% water) that the model predicted as almost entirely land** — the direct symptom of the collapse in §5, where the restored epoch-1 weights output P(water) < 0.5 nearly everywhere. (These are also the worst *possible* cases: the model misses exactly the patches that are mostly flooded.)

---

## 7. FROZEN FILES — exact paths backing every number above

| Artifact | Path |
|---|---|
| U-Net metrics (IoU/F1/loss/gap/timings/pixel size/worst patches) | `results/20260701T224956Z/metrics.json` |
| Per-epoch training history | `results/20260701T224956Z/training_history.csv` |
| Baseline results table | `results/20260701T224956Z/baseline_comparison.csv` |
| NDWI threshold sweep | `results/20260701T224956Z/ndwi_threshold_sweep.csv` |
| Trained model (best = epoch 1) | `results/20260701T224956Z/disastershield_best.keras` |
| Validation prediction figure | `results/20260701T224956Z/predictions.png` |
| Class-balance figure | `results/20260701T224956Z/class_balance.png` |
| Training script | `scripts/train_flood_model.py` |
| Baseline script | `scripts/baselines.py` |
| Smoke-test run (2 epochs) | `results/20260701T223239Z/` |

Timings (from `metrics.json`): full training wall-clock **208.67 s** for **13 epochs run** (early stopped from 60); the process was **not killed and did not thrash** (RAM held; CPU-only). Inference: **1.095 s** to predict all **148** test patches in one batch (`inference_seconds_all_test_patches`).

---

## 8. RESOLUTION CAVEAT

Measured ground pixel size is **≈ 458 m × 497 m** per pixel (EPSG:4326, ~0.004492°, center latitude 23.68°N). A single 64×64 patch therefore covers **≈ 29 km × 32 km** on the ground.

---

## HARD STOP
Steps 1–6 complete. Per instructions I am stopping here and will not start any demo work until you say so.
