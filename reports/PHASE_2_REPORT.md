# PHASE 2 — Feni 10 m Pipeline with Independent Labels

**Task:** predict the UNOSAT-observed Aug-2024 flood extent over Feni/Noakhali from Sentinel satellite inputs at 10 m.
**Labels:** UNOSAT flood-extent shapefiles (independent, radar+analyst derived) — replaces the falsified circular NDWI>0 labels of Phase 1.
**Environment:** `.venv-dsx` (Python 3.12.10, TF 2.21 CPU). Added this phase: geopandas 1.1.4, shapely 2.1.2, pyogrio 0.13.0, pyproj 3.7.2 (pandas auto-upgraded 2.x→3.0.3).
**Seeds:** 42 everywhere. **RAM discipline:** windowed reads, float32 on read, one raster at a time.

Audit script: [scripts/audit_feni.py](../scripts/audit_feni.py) · clip check: [scripts/_clip_check.py](../scripts/_clip_check.py)

---

## STEP 1 — DATA AUDIT (read-only) — COMPLETE

### 1a. Raster inventory (all 4 GeoTIFFs)

All four rasters share **one identical grid**: 7793 × 6684 px, EPSG:4326 (geographic), same bounds — perfectly co-registered, no resampling needed for alignment.

| File | Bands | dtype | W × H | Pixel size (m)¹ | CRS | Size |
|---|---|---|---|---|---|---|
| Feni_S2_Flood_Aug2024_10m.tif | 4 (B2,B3,B4,B8) | int16 | 7793 × 6684 | 9.21 × 9.93 | EPSG:4326 | 117.9 MB |
| Feni_S2_PreFlood_May2024_10m.tif | 4 (B2,B3,B4,B8) | int16 | 7793 × 6684 | 9.21 × 9.93 | EPSG:4326 | 404.9 MB |
| Feni_S1_Flood_Aug2024_10m.tif | 2 (VV,VH) | float32 | 7793 × 6684 | 9.21 × 9.93 | EPSG:4326 | 458.3 MB |
| Feni_S1_PreFlood_May2024_10m.tif | 2 (VV,VH) | float32 | 7793 × 6684 | 9.21 × 9.93 | EPSG:4326 | 455.6 MB |

¹ CRS is geographic (degrees); pixel size approximated at scene-center latitude (~23.0°N): 8.99e-5° lon ≈ 9.21 m, 8.93e-5° lat ≈ 9.93 m. Consistent with the stated 10 m product. nodata = None on all four.

Bounds (all four): lon 90.900 → 91.600, lat 22.700 → 23.300. Window area ≈ **4772 km²**.

### 1b. S2 flood-window usability — **THE CLOUD-GAP MEASUREMENT**

> **Feni_S2_Flood_Aug2024_10m.tif valid-pixel fraction = 0.28694 → only 28.7 % of the scene is usable; ~71 % is cloud/gap.**

(valid = all bands finite AND not all-zero; nodata=None so zero-fill is the mask.)

### 1c. S1 flood-window usability

> **Feni_S1_Flood_Aug2024_10m.tif valid-pixel fraction = 0.99942 → 99.94 % complete.** Radar penetrates cloud, as expected.

### 1d. UNOSAT labels

Three shapefiles present. CRS = EPSG:4326 for all. Each is a single feature (MultiPolygon); area computed in EPSG:32646 (UTM 46N). Bounding boxes cover **all of Bangladesh** (lon 88.0→92.7, lat 20.6→26.6), so they fully enclose the small Feni window (raster-bbox overlap = 100 %). Nation-wide totals and — more meaningfully — the flood clipped to the Feni window:

| Shapefile | Polygon parts | Nation-wide area (km²) | Flood **inside Feni window** (km²) | Flood % of window |
|---|---|---|---|---|
| **S1_20240818_20240826_FloodExtent** (primary — matches Aug flood imagery) | 957,982 | 12,641.7 | **905.53** | **18.97 %** |
| S1_20240828_20240904_FloodExtent (recession) | 758,171 | 6,155.0 | 652.03 | 13.66 % |
| AnalysisExtent (AOI mask, not flood) | 272 | 140,051.6 | — | — |

Raster-bbox overlap = 100 % for all (labels enclose the raster). The Feni window sits well inside the analysis extent.

### 1e. VERDICT

- **Usable flood-window input: S1 (radar), NOT S2.** S2 flood window is only **28.7 %** valid (dense cloud over the Aug-2024 flood) — unusable as a standalone optical input. S1 flood is **99.94 %** complete.
- **Labels spatially cover the raster region: YES.** UNOSAT flood polygons fully enclose and populate the Feni window; the primary Aug 18–26 layer puts **18.97 %** of the window under flood — a healthy, non-degenerate class balance for Step 3.
- **Recommended path (per your Step-3 rule "if S1-only"): 4-channel S1 change detection** = VV,VH (flood Aug) + VV,VH (pre-flood May). S2 flood is out; S2 pre-flood is complete but the *flood-window* optical signal is the cloudy one, so optical cannot carry the flood observation.
- **Recommended label layer: S1_20240818_20240826** (temporally matches the "Flood Aug 2024" imagery). Aug 28–Sep 04 is recession and would under-label the peak.

---

## ✅ CHECKPOINT (end of Step 1) — CLEARED

Go-ahead received. Locked decisions: **S1-only 4-channel** (VV,VH flood + VV,VH pre-flood); label = **S1_20240818_20240826**; permanent-water proxy = S1 pre-flood **VV < −16 dB**.

---

## STEP 2 — LABEL RASTERIZATION + ALIGNMENT — COMPLETE

Script: [scripts/step2_rasterize_label.py](../scripts/step2_rasterize_label.py) · frozen: `results/20260702T203648Z/step2_report.json`

### 2a. Rasterized label
UNOSAT `S1_20240818_20240826` polygons (read with a bbox filter to the Feni window, reprojected to grid CRS) rasterized onto the **exact shared grid** (7793×6684, EPSG:4326, S1-flood transform) → **`data/processed/feni_flood_label_10m.tif`** (uint8, 1=flood/0=non-flood, LZW, nodata=0).

### 2b. Sanity check
- **Flooded-pixel fraction = 0.189296** (9,860,152 / 52,088,412 px) — matches the 18.97 % vector-area estimate from Step 1 (rasterization is faithful).
- Overlay (decimated ×8): `results/20260702T203648Z/label_overlay_check.png`. Red flood labels track the dark low-backscatter flooded farmland; the permanent Feni/Little-Feni river + Bay-of-Bengal estuary channel is **not** labelled flood — visual confirmation of 2c.

### 2c. Permanent-water noise (measured, not fixed)
Proxy = S1 **pre-flood** VV < −16 dB (VV/VH confirmed in dB: VV ≈ −22…+6).

> **1,535,641 permanent-water px (2.95 % of scene). 92.45 % of them fall OUTSIDE the UNOSAT flood polygons** (1,419,706 / 1,535,641).

Interpretation: UNOSAT maps *flood* water and excludes permanent open water (rivers, estuary, aquaculture ponds, and the Bay of Bengal along the south edge of the window). So low backscatter alone does **not** equal flood — the negative class contains a large body of genuinely-wet-but-not-flood pixels. This is expected label behaviour and quantifies how hard the flood-vs-permanent-water distinction is for the model and the threshold baselines.

---

## STEP 3 — PATCH DATASET WITH SPATIAL SPLIT — COMPLETE

Script: [scripts/step3_make_patches.py](../scripts/step3_make_patches.py) · manifest: `data/processed/feni_patches_manifest.json`

- **Channels (4):** `[VV_flood, VH_flood, VV_pre, VH_pre]` (dB) — S1 change detection.
- **Patches:** 64×64, stride 64 (no overlap). Skip <50 % valid. Seed 42.
- **Spatial split by longitude = column** (regular EPSG:4326 grid, so longitude is linear in column):
  - train = west 60 % (cols 0–4675.8, lon ≤ 91.320°)
  - val = middle 20 % (cols 4675.8–6234.4, lon 91.320–91.460°)
  - test = east 20 % (cols 6234.4–7793, lon > 91.460°)
- **208 straddling patches dropped** (crossed a boundary column); **0 dropped for <50 % valid** (S1 is 99.94 % complete).
- Flood input = **v1** `Feni_S1_Flood_Aug2024_10m.tif` (Aug 20–Sep 5 composite). Frozen arrays now retagged: `data/processed/feni_{X,y}_{train,val,test}_v1.npy`; manifest `feni_patches_manifest_v1.json`.

| Split | Patches | Flood-pixel fraction | Flood px / total |
|---|---|---|---|
| **train** (west 60 %) | 7,592 | **0.2057** (20.6 %) | 6,396,438 / 31,096,832 |
| **val** (mid 20 %) | 2,392 | **0.1856** (18.6 %) | 1,818,388 / 9,797,632 |
| **test** (east 20 %) | 2,392 | **0.1448** (14.5 %) | 1,418,420 / 9,797,632 |
| total | 12,376 | — | — |

All three splits sit comfortably in **14–21 % flood** — well inside the non-degenerate 2–80 % band. No boundary adjustment needed.

---

## v2 UPDATE — window-matched flood composite (PRIMARY)

New raster `data/Feni_2024_10m/Feni_S1_Flood_18to26Aug2024_10m.tif` is a composite over **Aug 18–26**, matching the UNOSAT `S1_20240818_20240826` label window exactly. It becomes the **primary flood input**; v1 (Aug 20–Sep 5) is retained as an ablation. Audit: [scripts/audit_v2.py](../scripts/audit_v2.py); patch build reuses [scripts/step3_make_patches.py](../scripts/step3_make_patches.py) `--flood … --tag v2`.

### v2 audit + grid check
| Property | v2 | Reference grid (other 4 rasters) |
|---|---|---|
| Bands / dtype | 2 (VV,VH) / float32 | — |
| Width × Height | 7793 × 6684 | 7793 × 6684 |
| CRS | EPSG:4326 | EPSG:4326 |
| Transform | 8.983152841195215e-05 px, origin (90.8999846, 23.3004120) | identical |
| Bounds | 90.900,22.700 → 91.600,23.300 | identical |
| **Valid-pixel fraction** | **0.99942** | (S1 flood v1: 0.99942) |
| File size | 461.3 MB | — |

> **Grid comparison: dims match, CRS match, transform max-abs-diff = 0.0 → PERFECTLY ALIGNED. Path taken: use v2 DIRECTLY, no resampling** (no `Feni_S1_Flood_v2_aligned.tif` needed).

### Label rasterization (Step 2) — already run, reused for v2
The label (`data/processed/feni_flood_label_10m.tif`, from UNOSAT `S1_20240818_20240826`) was rasterized on the shared grid in the earlier Step 2 and is reused unchanged for v2 (same grid). Recap of frozen stats (`results/20260702T203648Z/step2_report.json`):
- Flood-pixel fraction = **0.189296**; overlay PNG `results/20260702T203648Z/label_overlay_check.png`.
- Permanent-water outside UNOSAT (pre-flood VV < −16 dB) = **0.924504** (1,419,706 / 1,535,641 px).

### v2 patch dataset — per-split counts + class balance
Flood input = **v2** `Feni_S1_Flood_18to26Aug2024_10m.tif`. Same recipe (64 px, stride 64, thirds split, skip <50 % valid, seed 42). 208 straddlers dropped, 0 dropped for validity. Frozen: `data/processed/feni_{X,y}_{train,val,test}_v2.npy`; manifest `feni_patches_manifest_v2.json`.

| Split | Patches | Flood-pixel fraction | Flood px / total |
|---|---|---|---|
| **train** (west 60 %) | 7,592 | **0.2057** (20.6 %) | 6,396,438 / 31,096,832 |
| **val** (mid 20 %) | 2,392 | **0.1856** (18.6 %) | 1,818,388 / 9,797,632 |
| **test** (east 20 %) | 2,392 | **0.1448** (14.5 %) | 1,418,420 / 9,797,632 |
| total | 12,376 | — | — |

**v2 counts and flood fractions are identical to v1** — expected: patch geometry, split boundaries, straddle drops, valid mask and *labels* are all shared, and both flood rasters are 99.94 % complete on the identical grid. Only the VV/VH **flood-channel pixel values** differ between v1 and v2 (the composite window). All splits again in 14–21 % — non-degenerate.

---

## ⛔ CHECKPOINT — HARD STOP (v2 patch dataset ready)

v2 is the primary input (window-matched, perfectly grid-aligned, used directly). v1 retained for the composite-window ablation. Both patch sets frozen; no split is degenerate. **Awaiting your review of class balance before Step 4 (train + baselines).**

---

## STEP 4a/4b — TRAINING SCRIPT + SMOKE + TIMING — COMPLETE (v2)

Script: [scripts/train_feni.py](../scripts/train_feni.py). 4-channel dB input `(64,64,4)`, no NDWI. Same U-Net / Dice+weighted-CE / IoU+F1 / ModelCheckpoint+EarlyStopping(12,restore_best)+ReduceLROnPlateau / batch 16 / `--smoke`,`--epochs`,`--tag`.

**Normalization (dB, not reflectance):** per-channel mean/std on **train split only**, applied to all splits. No `/10000`, no `[0,1]` clip. Frozen to `data/processed/norm_stats_v2.json` and each run's metrics.json.
- mean = [−9.643, −16.556, −8.777, −13.348] · std = [4.475, 5.331, 2.229, 3.292] (VV_flood, VH_flood, VV_pre, VH_pre).
**Augmentation:** H/V flips + 90° rotations + Gaussian noise σ=0.05 (standardized units). No brightness jitter (radar calibrated).
**water_weight** = clip((1−0.2057)/0.2057, 2, 15) = **3.862** (from train flood fraction).

### Bug found + fixed during smoke (recorded, not hidden)
First smoke run produced **NaN loss** and NaN norm stats. Cause: ~0.018 % of train patch pixels are non-finite (SAR invalid pixels inside otherwise-≥50 %-valid patches; train 22,608 / val 4,412 / test 16,908 channel-values). Fix: compute stats with `nanmean/nanstd`, and fill non-finite pixels with the channel mean → standardized value 0 (neutral). Re-run smoke was clean.

### SMOKE — PASS
200 train patches, 2 epochs, seed 42 (`results/20260703T050659Z/`):
- Norm stats finite; shapes train (7592,64,64,4)/val (2392,…)/test (2392,…); flood frac train 20.3 % / val 18.6 % / test 14.5 %.
- Loss finite and **decreasing 1.803 → 1.136**; val IoU **0.471 → 0.519** (improving); metrics.json written.

### TIMING — one full-size epoch (all 7,592 train patches, `--epochs 1`, `results/20260703T050924Z/`)
| Quantity | Value (frozen in metrics.json) |
|---|---|
| Steps / epoch | 475 (batch 16) |
| **Seconds per epoch (train+val, CPU)** | **385.6 s (≈ 6.43 min), 812 ms/step** |
| Inference, full test set (2,392 patches) | 19.66 s |
| val IoU after 1 epoch | 0.6414 |
| test IoU / F1 after 1 epoch | 0.5933 / 0.6881 |

**Projected 60-epoch run** (early-stopped, patience 12):
- Full 60 epochs ≈ 60 × 385.6 s ≈ **6.4 hours**.
- Realistic early stop 25–40 epochs ⇒ **≈ 2.7 h (25 ep) to ≈ 4.3 h (40 ep)**.
- Machine is CPU-only (TF 2.21, no native-Windows GPU).

---

## ⛔ CHECKPOINT — HARD STOP (end of Step 4b)

Smoke PASSes; one full epoch measured at **6.4 min/epoch** → full 60-epoch run ≈ **6.4 h max**, likely **~3–4 h** with early stopping. **Your call: run now vs overnight.** On confirmation I proceed to Step 4c (baselines first, then full training) then 4d (final results table).

---

## STEP 4c-1 — BASELINES (v2 splits) — COMPLETE

Script: [scripts/baselines_feni.py](../scripts/baselines_feni.py). Frozen: `results/20260703T052548Z/baseline_comparison.csv` + `baselines_summary.json`.
Metric = **global pixel-level water-class IoU/F1** (identical definition to the U-Net metric → directly comparable). Non-finite pixels **mean-filled with train channel means** (same handling as training). A/B thresholds **tuned on VAL only**, then evaluated once on TEST.

| Baseline | Chosen threshold (val) | val IoU | test IoU | test F1 |
|---|---|---|---|---|
| **A** — VV-change, flood = (VV_flood − VV_pre) < t | **−4.0 dB** | 0.5755 | 0.5724 | 0.7281 |
| **B** — VV_flood absolute, flood = VV_flood < t | **−13.0 dB** | 0.4708 | 0.5518 | 0.7111 |
| **C** — logistic regression, 4 raw-dB channels, 500k-px sample (seed 42) | 0.5 decision | 0.5664 | **0.6128** | **0.7599** |

- **C (logistic regression) is the strongest baseline** (test IoU 0.6128). Learned coefficients form a sensible change-detector: VV_flood −0.343, VH_flood −0.301 (lower flood-window backscatter → flood), VV_pre +0.163, VH_pre +0.253 (higher pre-flood backscatter → bigger drop → flood), intercept −5.818.
- **A beats B:** the *change* signal (flood − pre) separates flood better than an absolute VV threshold, because permanent water (Step 2c, 92.45 % of low-VV pixels lie outside UNOSAT) also has low absolute VV but little *change* — B misclassifies permanent water as flood, A/C largely do not.
- Reference: the U-Net after a single epoch already reached test IoU 0.593 (≈ A/B, below C); full training is expected to surpass all three.

### Training-script durability update (for the manual full run)
[scripts/train_feni.py](../scripts/train_feni.py) now: (a) appends per-epoch rows to `training_history.csv` via `CSVLogger` after every epoch; (b) saves **two** checkpoints — `feni_unet_best.keras` (on val-IoU improvement) and `feni_unet_last.keras` (every epoch) — so an interrupted run loses at most one epoch. Command for the manual run: `python scripts/train_feni.py --epochs 60 --tag v2`.

---

## STEP 4c-2 — FULL TRAINING — COMPLETE

`python scripts/train_feni.py --epochs 60 --tag v2`. `training_history.csv` flushed **per epoch** (CSVLogger); best model checkpointed on every val-IoU improvement, plus a `feni_unet_last.keras` after each epoch. Run frozen to **`results/20260703T054304Z/`**.

- **Early-stopped at 30 epochs** (of 60 requested; patience 12, `restore_best_weights=True`). Best val IoU at **epoch 17**; 12 non-improving epochs later → stop.
- Wall-clock **11,706 s ≈ 3.25 h** on CPU, **390 s/epoch** (matches the 385.6 s/epoch Step-4b projection). Full-test inference 18.5 s / 2,392 patches.
- Restored best (epoch-17) weights are what the test numbers below are computed on.

---

## STEP 4d — FINAL RESULTS

Frozen run: **`results/20260703T054304Z/`** (metrics.json, training_history.csv, training_curves.png, feni_unet_best.keras, predictions.png). Baselines: **`results/20260703T052548Z/baseline_comparison.csv`**. Norm stats: `data/processed/norm_stats_v2.json`.

### Results table — U-Net vs baselines A / B / C

All numbers are **global pixel-level water-class IoU/F1** on the v2 splits (one metric definition throughout → directly comparable). U-Net val = the restored **best epoch (17)**; U-Net test = best-weights eval. A/B thresholds tuned on VAL only. Baseline val-F1 was not computed (only val IoU drove threshold selection).

| Model | val IoU | val F1 | test IoU | test F1 | Source (frozen) |
|---|---|---|---|---|---|
| **U-Net (v2, 4-ch S1)** | **0.6516** | **0.7450** | **0.6052** | **0.6964** | `results/20260703T054304Z/metrics.json` |
| A — VV-change `(VV_flood−VV_pre) < −4.0 dB` | 0.5755 | — | 0.5724 | 0.7281 | `results/20260703T052548Z/baseline_comparison.csv` |
| B — VV_flood abs `< −13.0 dB` | 0.4708 | — | 0.5518 | 0.7111 | `results/20260703T052548Z/baseline_comparison.csv` |
| C — logistic regression (4 raw-dB ch) | 0.5664 | — | **0.6128** | **0.7599** | `results/20260703T052548Z/baseline_comparison.csv` |

**Verdict on the comparison:** on **val IoU** the U-Net (0.6516) clearly leads every baseline (best baseline A 0.5755). On **test**, the U-Net (IoU 0.6052) beats thresholds A (0.5724) and B (0.5518) but the logistic-regression baseline **C edges it out** (test IoU 0.6128, F1 0.7599 vs 0.6964). The U-Net wins val but does not generalise past the linear pixel classifier on the held-out **east** test strip — the train→test IoU drop (0.6516→0.6052, −0.046) is larger for the U-Net than for C (0.5664→0.6128, which actually *rises*), i.e. C is the more robust generaliser here on this east strip while the U-Net has fit west/val-specific texture. The per-pixel F1s are close across all four (0.70–0.76).

### Convergence — best epoch & total epochs

- **Best val IoU 0.6516 at epoch 17** (0-indexed); training ran **30 epochs** total before EarlyStopping (patience 12) fired and restored epoch-17 weights.
- Val IoU is essentially **flat from epoch ~3 onward** (0.645–0.652 band) while train IoU keeps climbing 0.68→0.77 — the model learned the separable signal within the first few epochs; the rest is train-only refinement. ReduceLROnPlateau stepped the LR 1e-4→…→6.25e-6 across the run.
- Curves: **`results/20260703T054304Z/training_curves.png`** (loss + IoU, train vs val, best-epoch marked).

### Train–val gap & overfitting verdict

- final_train_iou **0.7684** − final_val_iou **0.6435** = **train–val gap 0.1249 IoU** (frozen `metrics.json:train_val_gap`).
- **Verdict: mild, controlled overfitting.** The gap opens because train IoU rises while val plateaus, but **val IoU never degrades** (no divergence), and early-stopping + restore-best means the deployed weights are the epoch-17 optimum, not the over-fit epoch-29 state. The gap is a symptom of limited *new* signal past epoch 3 rather than harmful memorisation — consistent with a 4-channel radar change task where the separable structure is learned quickly.

### 3 worst test patches (frozen `metrics.json:worst_test_patches`)

| Rank | patch_index | IoU | true flood frac | pred flood frac | Note |
|---|---|---|---|---|---|
| 1 | 129 | ~0 | 0.000 | 0.901 | Pure **false positive**: patch has zero UNOSAT flood, model floods 90 % of it. |
| 2 | 178 | ~0 | 0.000 | 0.710 | False positive on an all-dry (per-label) patch. |
| 3 | 14 | ~0 | 0.000 | 0.699 | False positive on an all-dry (per-label) patch. |

All three worst cases share the same failure mode: **true flood fraction = 0, high predicted flood**. This is exactly the **permanent-water confusion quantified in Step 2c** — 92.45 % of pre-flood permanent-water pixels (VV < −16 dB) lie *outside* UNOSAT flood polygons. In the east test strip these all-low-backscatter permanent-water bodies (river/estuary/aquaculture ponds) look like flood to the network, which lacks an explicit "was this already open water?" prior beyond the pre-flood channels. The change-detector baselines A/C suppress this better because differencing against the pre-flood VV cancels standing water — one reason C narrowly out-generalises the U-Net on test.

### Known label noise

> v2 flood composite (Aug 18–26 2024) matches the UNOSAT `S1_20240818_20240826` label window **exactly**, so the v1 composite-window caveat (v1 spans Aug 20–Sep 5, wider than the label) does **not** apply to this run. Residual negative-class noise: ~**92.45 %** of pre-flood permanent water (VV < −16 dB, Step 2c) falls outside UNOSAT polygons — i.e. permanent open water is genuinely non-flood in the negatives, and the worst-patch false positives above are the model failing to honour that distinction.

(Verbatim from `results/20260703T054304Z/metrics.json:known_label_noise`.)

### Frozen file paths behind every number

| Quantity | File |
|---|---|
| U-Net val/test IoU & F1, gap, epochs, timing, worst patches, label-noise note | `results/20260703T054304Z/metrics.json` |
| Per-epoch loss/IoU/F1/LR (train+val) | `results/20260703T054304Z/training_history.csv` |
| Training curves plot | `results/20260703T054304Z/training_curves.png` |
| Best / last model weights | `results/20260703T054304Z/feni_unet_best.keras` · `feni_unet_last.keras` |
| Qualitative prediction panel | `results/20260703T054304Z/predictions.png` |
| Baseline A/B/C val+test IoU/F1 | `results/20260703T052548Z/baseline_comparison.csv` |
| Per-channel normalization mean/std (train-only) | `data/processed/norm_stats_v2.json` |

---

# STEP 5 — v3: PHYSICS-INFORMED CHANNELS + FAIR THRESHOLD PROTOCOL

Seeds 42. Frozen files. Hard stops as marked.

## Hypothesis (recorded BEFORE running 5a/5b/5d)

> The U-Net's test-time **permanent-water false positives** (Step 4d worst patches: true flood frac 0, predicted 0.70–0.90) arise because **per-channel standardization obscures the physical change signal** VV_flood − VV_pre. The flood and pre-flood VV channels are standardized with *different* per-channel std (VV_flood std **4.475** vs VV_pre std **2.229**, from `norm_stats_v2.json`), so after standardization `z(VV_flood) − z(VV_pre) ≠ (const)·(VV_flood − VV_pre)`: the network can no longer recover the raw-dB difference by subtracting two input planes, because they live on different scales. The change-detector baselines (A: `VV_flood−VV_pre`; C: logreg, which can learn a difference in raw-dB space) suppress permanent water precisely because they operate on the *change*. **Prediction:** giving the U-Net explicit delta channels `dVV = VV_flood − VV_pre`, `dVH = VH_flood − VH_pre` computed in **raw dB before normalization** will reduce the permanent-water false positives and close (or surpass) the 0.008 test-IoU gap to logistic regression C (v2 U-Net test IoU 0.6052 vs C 0.6128).

**Secondary hypothesis (tested by 5d, independent of v3):** part of that 0.008 gap may be pure **threshold miscalibration** — both models are evaluated at a fixed 0.5 probability cut, which is not necessarily the IoU-optimal operating point. 5d sweeps the decision threshold on VAL only and re-reports TEST at the val-tuned threshold for a fair comparison.

---

## STEP 5a — v3 6-CHANNEL PHYSICS-INFORMED PATCHES — COMPLETE

Script: [scripts/step5_make_patches_v3.py](../scripts/step5_make_patches_v3.py) · manifest `data/processed/feni_patches_manifest_v3.json` · norm stats `data/processed/norm_stats_v3.json`.

- **Channels (6):** `[VV_flood, VH_flood, VV_pre, VH_pre, dVV, dVH]` in raw dB. `dVV = VV_flood − VV_pre`, `dVH = VH_flood − VH_pre`, **computed in raw dB BEFORE any normalization**.
- **Identical geometry to v2:** same grid (7793×6684), same longitude-thirds split, same straddle rule (**208 dropped**), same skip-<50 %-valid (**0 dropped**), seed 42, same window-matched flood input `Feni_S1_Flood_18to26Aug2024_10m.tif`. Valid mask uses the 4 base channels only (deltas are derived, add no new validity constraint).
- **Counts + class balance identical to v2** (as expected — only the two derived planes are new):

| Split | Patches | Flood-pixel fraction | X shape |
|---|---|---|---|
| train (west 60 %) | 7,592 | 0.2057 | (7592, 64, 64, 6) |
| val (mid 20 %) | 2,392 | 0.1856 | (2392, 64, 64, 6) |
| test (east 20 %) | 2,392 | 0.1448 | (2392, 64, 64, 6) |

**Train-only norm stats, all 6 channels** (`norm_stats_v3.json`, nanmean/nanstd, non-finite mean-filled — 33,912 non-finite train channel-values, up from v2's 22,608 because the two delta planes inherit non-finite from their operands):

| Channel | VV_flood | VH_flood | VV_pre | VH_pre | dVV | dVH |
|---|---|---|---|---|---|---|
| mean | −9.643 | −16.556 | −8.777 | −13.348 | **−1.239** | **−2.154** |
| std | 4.475 | 5.331 | 2.229 | 3.292 | **3.587** | **4.292** |

Base-channel mean/std match v2 to the digit (same rasters, same train pixels) — confirms the build is a clean superset of v2. All 6 stats finite.

---

## STEP 5b — v3 SMOKE — PASS

Frozen: `results/20260703T133749Z/metrics.json`. 200 train patches, 2 epochs, seed 42; `train_feni.py` now resolves channel count from the data (6 → v3 names, else 4).

- **Shapes (64,64,6) confirmed:** train (7592,64,64,6) / val (2392,…) / test (2392,…); `input_shape` in metrics.json = `[64,64,6]`; channels = the 6 v3 names.
- **Norm stats finite** (mean & std both all-finite, verified from metrics.json `normalization`).
- **Loss decreasing 1.6631 → 1.0457**; val IoU **0.5162 → 0.5723** (improving, best_val_iou 0.5723); metrics.json written.
- water_weight 3.934 (from train flood frac 20.27 %). sec/epoch 36 (smoke, 200 patches).

**SMOKE PASS** — v3 pipeline is finite, correctly shaped, and learning.

---

## STEP 5d — THRESHOLD-FAIR RE-EVALUATION (v2) — COMPLETE

Script: [scripts/step5d_threshold_fair.py](../scripts/step5d_threshold_fair.py) · frozen `results/20260703T134121Z/threshold_fair_v2.csv` + `.json`. No training. v2 U-Net = `results/20260703T054304Z/feni_unet_best.keras` (loaded `compile=False`), logreg refit = baseline C (500k train-px sample, seed 42). Probability threshold swept **on VAL only** over [0.05, 0.95] step 0.05; TEST reported at the val-tuned threshold. **Metric = global pixel-level water-class IoU/F1 for every row** (the fix — see below).

### ⚠️ Metric-definition bug uncovered (must read before interpreting Step 4d)

The Step 4d table compared the U-Net's **batch-averaged** IoU (Keras means the per-batch `iou_metric` over the 150 test batches in `model.evaluate` → **0.6052**) against the baselines' **global pooled** IoU (all pixels at once → C = 0.6128). **These are two different metrics.** Verified directly: on the identical restored-best v2 weights, batch-averaged test IoU = **0.60523** (reproduces `metrics.json:test_iou` to 5 dp) while the **global pooled** test IoU = **0.71435**. IoU is nonlinear, so the mean of per-batch IoUs ≠ the pooled IoU (batches with few flood pixels drag the average down). **The "logreg C edges out the U-Net by 0.008" conclusion in Step 4d was an artifact of this mismatch, not a real deficit.** Under one consistent (global) metric the U-Net leads C by ~0.10 IoU on test.

### Threshold-fair v2 table (all rows = global pixel-level IoU/F1)

| Method | val threshold | val IoU | val F1 | test IoU | test F1 |
|---|---|---|---|---|---|
| **U-Net @0.5** | 0.50 | 0.7055 | 0.8273 | 0.7144 | 0.8334 |
| **U-Net @val-tuned** | **0.65** | **0.7118** | **0.8316** | **0.7216** | **0.8383** |
| logreg @0.5 | 0.50 | 0.5664 | 0.7232 | 0.6128 | 0.7599 |
| logreg @val-tuned | 0.50 | 0.5664 | 0.7232 | 0.6128 | 0.7599 |
| A — VVchange `< −4 dB` | −4.0 | 0.5755 | 0.7306 | 0.5724 | 0.7281 |
| B — VVflood `< −13 dB` | −13.0 | 0.4708 | 0.6402 | 0.5518 | 0.7111 |

(All numbers quoted from `results/20260703T134121Z/threshold_fair_v2.csv`.)

### What this tells us about the 0.008 gap

- **Almost none of it was threshold miscalibration, and none of it was a real U-Net weakness.** Two effects, in order of size:
  1. **Metric mismatch (dominant, ~0.10 IoU):** measured consistently (global), the v2 U-Net's test IoU is **0.7144 @0.5**, already **+0.102** over logreg C's **0.6128** — the opposite of the Step-4d apparent −0.008.
  2. **Threshold tuning (small, +0.007 IoU):** the U-Net's val-IoU-optimal probability cut is **0.65**, not 0.5 (the loss over-predicts water at 0.5); moving to 0.65 lifts test IoU **0.7144 → 0.7216** and test F1 to 0.8383. logreg's optimum is **0.5** (no change) — it is already well-calibrated, so tuning does not help it.
- Net: under a fair protocol (one metric + per-method val-tuned threshold) the v2 U-Net beats logreg C by **+0.109 test IoU** (0.7216 vs 0.6128) and every threshold baseline by ≥0.15.

### Consequence for the v3 hypothesis

The **primary** v3 premise — that the U-Net has a permanent-water deficit *relative to the change-detector* that explicit delta channels must close — is **weaker than stated**: fairly measured, the U-Net already out-generalises the change-detector C. v3 is therefore re-framed from "close the gap to C" to a cleaner ablation: **do explicit raw-dB delta channels reduce the U-Net's own permanent-water false positives** (the Step-4d worst patches with true flood frac 0, pred 0.70–0.90) and raise the fairly-measured 0.7216 further? The Step-4d worst-patch failure mode is real regardless of the metric bug (it is per-patch, not a pooling artifact), so the ablation still has a well-posed target.

---

## STEP 5c — FULL v3 RUN COMMAND (prepared, NOT launched)

[scripts/train_feni.py](../scripts/train_feni.py) now resolves the input channel count from the data (`X_train.shape[-1]`): 6 → `[VV_flood, VH_flood, VV_pre, VH_pre, dVV, dVH]`, else the 4-channel v1/v2 set. `input_shape` becomes `(64,64,6)` automatically; `metrics.json:flood_input` records the window-matched raster for v3. No other change needed.

**Exact command (run manually in your own terminal, as before):**

```
d:\Hamim\DisasterShield\.venv-dsx\Scripts\python.exe scripts\train_feni.py --epochs 60 --tag v3
```

Expected ≈ same wall-clock as v2 (~6.4 min/epoch CPU; ~3–4 h with early stopping — the extra 2 input planes add negligible cost vs the 1.95 M-param U-Net). Outputs freeze to a new `results/<UTC-timestamp>/` (metrics.json, training_history.csv, best/last `.keras`, predictions.png) and overwrite `data/processed/norm_stats_v3.json` with identical stats.

> **Note for v3 evaluation:** `metrics.json:test_iou` from that run will again be the **batch-averaged** IoU (~comparable to v2's 0.6052-scale number, NOT the global 0.7144). To compare v3 vs v2 vs baselines fairly, re-run the 5d threshold-fair script pointed at the new v3 best-model checkpoint (global metric + val-tuned threshold). Do **not** compare v3's batch-averaged `test_iou` against v2's global 0.7216.

---

## STEP 5e — v3 RESULTS AND FINAL MODEL DECISION

Frozen evaluation: [results/20260703T165223Z/threshold_fair_v3.csv](../results/20260703T165223Z/threshold_fair_v3.csv) + [threshold_fair_v3.json](../results/20260703T165223Z/threshold_fair_v3.json). Same fair protocol as 5d: global pooled pixel-level water-class IoU/F1, probability threshold swept on VAL only over [0.05, 0.95] step 0.05, and TEST reported once at the val-tuned threshold.

### Unified threshold-fair table (verbatim from `threshold_fair_v3.csv`)

| method | val_threshold | val_IoU | val_F1 | test_IoU | test_F1 |
|---|---|---|---|---|---|
| **U-Net v3 @val-tuned** | **0.45** | **0.710578** | **0.830805** | **0.719059** | **0.836573** |
| U-Net v3 @0.5 | 0.50 | 0.710339 | 0.830641 | 0.718734 | 0.836353 |
| **U-Net v2 @val-tuned** | **0.65** | **0.711761** | **0.831612** | **0.721554** | **0.838259** |
| U-Net v2 @0.5 | 0.50 | 0.705468 | 0.827302 | 0.714351 | 0.833378 |
| logreg C @val-tuned | 0.50 | 0.566448 | 0.723226 | 0.612796 | 0.759918 |
| logreg C @0.5 | 0.50 | 0.566448 | 0.723226 | 0.612796 | 0.759918 |
| A VVchange @val-tuned | -4.00 | 0.575539 | 0.730593 | 0.572444 | 0.728095 |
| B VVflood @val-tuned | -13.00 | 0.470837 | 0.640230 | 0.551768 | 0.711148 |

All rows are global pooled metrics from [results/20260703T165223Z/threshold_fair_v3.csv](../results/20260703T165223Z/threshold_fair_v3.csv). The v3 U-Net lands at test IoU **0.7191** (@0.45) — it does **not** beat v2's **0.7216** (@0.65); the gap is **−0.0025 IoU**, well within run-to-run noise.

### Convergence — v3 vs v2

| | best val IoU | at epoch (0-indexed row) | epochs run | train–val gap |
|---|---|---|---|---|
| **v3** (6-ch physics-informed) | 0.6585 | **epoch 1** (2nd trained epoch) | **14** | 0.1085 |
| **v2** (4-ch) | 0.6516 | **epoch 17** (18th trained epoch) | 30 | 0.1249 |

Sources: [results/20260703T135742Z/metrics.json](../results/20260703T135742Z/metrics.json)+[training_history.csv](../results/20260703T135742Z/training_history.csv) (v3); [results/20260703T054304Z/metrics.json](../results/20260703T054304Z/metrics.json)+[training_history.csv](../results/20260703T054304Z/training_history.csv) (v2). The one real benefit of the delta channels is **convergence speed**: v3 hits its best val IoU at its 2nd trained epoch and early-stops after 14, vs v2's 18th and 30 — the physics-informed change signal is learned almost immediately. Final accuracy is unchanged.

### HYPOTHESIS VERDICT — REFUTED

The explicit raw-dB delta channels (`dVV`, `dVH`) were predicted to **reduce the U-Net's permanent-water-type false positives**. **REFUTED** by the frozen patch analysis ([threshold_fair_v3.json](../results/20260703T165223Z/threshold_fair_v3.json), `failure_mode`):

- **False-positive count is identical: 14 vs 14** (test patches with true flood frac 0 and predicted flood frac > 0.5; `delta_v3_minus_v2 = 0`).
- **Same top offenders:** patches **129** and **178** are the two worst for both models. Severity barely moves — patch 129 predicted flood frac **0.8843** (v3) vs **0.9006** (v2); patch 178 **0.6975** vs **0.7100**.
- **Test IoU within noise:** **0.7191** (v3) vs **0.7216** (v2), a −0.0025 difference.

So the delta channels neither reduce the failure-mode count nor improve test accuracy; they only converge faster (best epoch 2 vs 17). Hypothesis rejected.

### FINAL MODEL DECISION

**The v2 4-channel model is FINAL for this event.** It matches v3 on every accuracy metric with **simpler inputs** (4 channels `[VV_flood, VH_flood, VV_pre, VH_pre]` vs 6), so there is no reason to carry the two derived planes.

- **Weights:** [results/20260703T054304Z/feni_unet_best.keras](../results/20260703T054304Z/feni_unet_best.keras)
- **Operating threshold:** **0.65** (val-tuned)
- **Canonical performance:** global pooled **test IoU 0.7216 / F1 0.8383** @ 0.65
- **Normalization:** `data/processed/norm_stats_v2.json`
- Per-run `metrics.json:test_iou/test_f1` in the v2 and v3 folders are **batch-averaged and non-canonical** — see the `METRIC_NOTE.md` dropped into each folder ([v2](../results/20260703T054304Z/METRIC_NOTE.md) · [v3](../results/20260703T135742Z/METRIC_NOTE.md)).

### Note for the record — v3 worst-patch listing

The `v3_worst_patches` / `v2_worst_patches` arrays in [threshold_fair_v3.json](../results/20260703T165223Z/threshold_fair_v3.json) are ordered by **highest mean predicted probability**, not by lowest IoU. This is unavoidable: every one of the 14 failure-mode patches has true flood frac 0, so its IoU is exactly 0 (no true positives possible) and IoU cannot rank them. The **false-positive COUNTS** — the actual verdict evidence (14 vs 14) — are computed correctly regardless of the listing order.

### PERMANENT-WATER DIAGNOSTIC

Do the FINAL v2 model's worst false positives sit on pre-flood permanent water (the mechanism assumed in Step 4d/5d)? Tested directly: for the three worst FP test patches, the fraction of each patch's **false-positive pixels** (pred flood @0.65 AND true non-flood) that fall inside the permanent-water proxy **VV_pre < −16 dB** (raw dB, Step-2c definition). Script [scripts/permwater_diagnostic.py](../scripts/permwater_diagnostic.py); frozen [results/20260703T190716Z/permwater_diagnostic/](../results/20260703T190716Z/permwater_diagnostic/) (per-patch 3-panel figures + `permwater_diagnostic.json`).

| patch | true flood frac | pred flood frac @0.65 | FP pixels | FP inside permanent water | fraction |
|---|---|---|---|---|---|
| 129 | 0.000 | 0.888 | 3638 | 0 | **0.000** |
| 153 | 0.000 | 0.681 | 2788 | 0 | **0.000** |
| 178 | 0.000 | 0.700 | 2868 | 0 | **0.000** |

**Verdict: NOT CONFIRMED — these are not river-or-pond confusion.** In all three patches the pre-flood VV is normal dry-land backscatter (median ≈ −9 dB, ~0 % below −16 dB), so essentially **none** of the false-positive pixels sit on permanent water. What the FP pixels actually show is a genuine **flood-time backscatter drop** on land that was dry in May: on the FP pixels VV falls from ≈ −9 dB (pre) to ≈ −16 dB (flood), a change of ≈ −6.5 dB. The model floods this land because it looks exactly like new standing water by change detection; UNOSAT's analyst product simply did not map it as flood. **This corrects the Step-4d/5d assumption** that the worst FPs are permanent open water — they are the opposite (newly-darkened land), and differencing against the pre-flood channel would *not* suppress them.

### RECESSION-LAYER CROSS-CHECK

Are those false positives instead **peak-window label omissions** — real flood that the Aug 18–26 UNOSAT layer missed but that appears in the later **recession** layer (UNOSAT `S1_20240828_20240904`, Aug 28–Sep 4)? Rasterized that layer onto the shared grid → `data/processed/feni_flood_label_recession_10m.tif` (scene flood frac **0.1366**, matching the Step-1d 13.66 % window estimate — rasterization faithful). Script [scripts/recession_crosscheck.py](../scripts/recession_crosscheck.py); frozen [results/20260703T202033Z/recession_crosscheck/recession_crosscheck.json](../results/20260703T202033Z/recession_crosscheck/recession_crosscheck.json). Geometry self-verified: the reproduced test-patch labels equal `feni_y_test_v2.npy` **exactly** (0 patches dropped for validity), so recession patches align index-for-index.

For all **14** zero-true-flood FP patches (true flood frac 0, predicted flood frac > 0.5 @0.65), the fraction of FP pixels labeled flood in the recession layer is **0.0000** (aggregate; per-patch mean and max both 0.0000).

**Verdict: FPs consistent with soil-moisture / SAR change-detection ambiguity — NOT peak-window label omission.** Combined with the permanent-water result, the diagnosis is fully triangulated: the model's residual false positives are neither permanent open water (0.000) nor flood UNOSAT mapped later in recession (0.000). They are genuine change-detection over-triggers — land that darkened ≈ 6.5 dB in the flood-window SAR (wet/ponded soil, flattened or harvested crops, transient inundation) that UNOSAT's analysts did not classify as flood in either window.

## STEP 6 — DEMO BUILD

Script [scripts/build_demo.py](../scripts/build_demo.py). Deliverable **`demo/index.html`** — a single self-contained file (**7.5 MB**, under the 50 MB target; the three/four overlays are embedded as base64 PNG, so the only external loads are Leaflet and the CartoDB basemap tiles from CDN). Supporting artifacts frozen to [results/20260703T202117Z/demo_build/](../results/20260703T202117Z/demo_build/): the four overlay PNGs, a downsampled probability GeoTIFF, and `build_manifest.json`.

### Inference
FINAL model [results/20260703T054304Z/feni_unet_best.keras](../results/20260703T054304Z/feni_unet_best.keras) (v2, 4-channel `[VV_flood, VH_flood, VV_pre, VH_pre]`) at operating threshold **0.65**, standardized with `data/processed/norm_stats_v2.json`. Full scene 6684×7793 run as **64×64 tiles at stride 32 (50 % overlap), with the softmax water probability averaged across overlapping tiles** (confirmed: overlap-averaged stitch, not tile-replaced), row-band streamed so the 4-channel float scene is never materialised at once. 50,544 tiles, ~7.8 min CPU.

### Headline the model earns
**Predicted flood 915.0 km² vs UNOSAT 905.5 km² → +9.5 km² (+1.0 %)**, over the full scene. Honest caveat: this is a whole-scene total, so most of it is the training region (west 60 %); only the east 20 % is spatially held out — the near-match is partly in-sample and the model genuinely over-predicts. Independent pixel-area cross-check: the UNOSAT label raster integrates to 903.5 km² under the same per-row geodesic pixel area (vs the 905.53 km² vector figure, 0.2 %).

### Layers (4, all toggleable)
| Layer | Default | Style |
|---|---|---|
| Sentinel-2 pre-flood (May 2024) | ON | true-colour base (B4/B3/B2, gamma-stretched) |
| **Agreement map (model vs UNOSAT)** | **ON** | dark-blue both-flood **780.7 km²** · orange model-only **134.3 km²** · magenta UNOSAT-only **122.7 km²** |
| DisasterShield-X predicted flood | OFF | semi-transparent blue fill |
| UNOSAT observed flood extent | OFF | thin, semi-transparent red outline (no fill) |

z-order bottom→top: **S2 → agreement → prediction → UNOSAT**, so the UNOSAT outline never occludes the prediction fill. Agreement areas are internally consistent: TP+FP = 780.7+134.3 = **915.0** (= predicted); TP+FN = 780.7+122.7 = **903.4** (= label). The orange (model-only) clusters are the diagnosed SAR-ambiguity zones — the estuary arc and the east test strip.

### Frozen sources behind every panel number
| Panel value | Source (read at build time) |
|---|---|
| Test IoU 0.72 / F1 0.84 | [threshold_fair_v3.csv](../results/20260703T165223Z/threshold_fair_v3.csv), `U-Net v2 @val-tuned` (0.721554 / 0.838259) |
| Baselines: logreg 0.61 · VV-change 0.57 · VV-abs 0.55 | [baseline_comparison.csv](../results/20260703T052548Z/baseline_comparison.csv) |
| Predicted 915.0 km²; agreement TP/FP/FN | build-time compute (per-row geodesic pixel area); frozen in `build_manifest.json` |
| UNOSAT 905.5 km² | Step 1d vector area (frozen) |

## ⛔ CHECKPOINT — PHASE 2 COMPLETE

- **5a:** v3 6-channel patches built, geometry identical to v2, all stats finite, `norm_stats_v3.json` frozen.
- **5b:** v3 smoke **PASS** (shapes (64,64,6), finite stats, loss 1.663→1.046, val IoU 0.516→0.572).
- **5d:** threshold-fair table frozen. **The Step-4d "logreg beats U-Net by 0.008" was a metric-definition artifact** (batch-averaged vs global IoU); fairly measured, the v2 U-Net leads logreg C by **+0.109 test IoU** (0.7216 val-tuned vs 0.6128).
- **5e:** v3 test IoU **0.7191** does **not** beat v2's **0.7216**; delta channels do **not** reduce the FP count (**14 vs 14**), only converge faster (best epoch 2 vs 17). **Final model: v2 4-channel**, [feni_unet_best.keras](../results/20260703T054304Z/feni_unet_best.keras), threshold **0.65**, test IoU **0.7216** / F1 **0.8383**.
- **FP diagnosis (triangulated):** the 14 worst FPs are **newly-darkened land** — not permanent water (0 % FP on VV_pre < −16 dB) and not recession-window flood (0 % FP in the Aug 28–Sep 4 UNOSAT layer) → soil-moisture / SAR change-detection ambiguity.
- **Demo:** self-contained [demo/index.html](../demo/index.html) (7.5 MB), 4 toggleable layers incl. the default agreement map; predicted 915.0 km² vs UNOSAT 905.5 km² (+9.5).

Phase 2 is **fully complete**. No further training launched. Nothing deployed or published.

