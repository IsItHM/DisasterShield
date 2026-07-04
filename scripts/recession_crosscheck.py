"""RECESSION-LAYER CROSS-CHECK (Phase 2 closeout).

Question: are the FINAL v2 model's permanent-FP test patches (true flood frac 0, high
predicted flood @0.65) explained by peak-window LABEL OMISSION? Test against the LATER UNOSAT
recession layer S1_20240828_20240904 (Aug 28 - Sep 04). If the model's FP pixels are flood in
the recession layer, the peak Aug18-26 label likely omitted flood the model (and the later
UNOSAT product) saw. If near zero, the FPs are soil-moisture / SAR change-detection ambiguity.

Steps:
  1. Rasterize S1_20240828_20240904 onto the shared grid -> feni_flood_label_recession_10m.tif.
  2. Reproduce the v2 test-patch geometry (stride 64, straddle drop, center-column split) and
     collect the recession-label test patches; SELF-VERIFY by rebuilding the peak-label test
     patches and asserting they equal feni_y_test_v2.npy (proves index alignment).
  3. v2 U-Net @0.65 predictions on feni_X_test_v2.npy; identify the zero-true-flood FP patches
     (true flood frac 0, predicted flood frac > 0.5).
  4. For each, fraction of FP pixels (pred flood AND true non-flood) that are flood in the
     recession layer. Freeze per-patch JSON + aggregate verdict.

Numbers only from frozen files. Seed 42. No training.
"""
import os, json, datetime
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

ROOT = r"d:\Hamim\DisasterShield"
LAB = os.path.join(ROOT, r"data\labels_unosat")
PROC = os.path.join(ROOT, r"data\processed")
PEAK_LABEL = os.path.join(PROC, "feni_flood_label_10m.tif")
RECESS_SHP = os.path.join(LAB, "S1_20240828_20240904_FloodExtent_Bangladesh.shp")
RECESS_OUT = os.path.join(PROC, "feni_flood_label_recession_10m.tif")
UNET = os.path.join(ROOT, r"results\20260703T054304Z\feni_unet_best.keras")
NORM = os.path.join(PROC, "norm_stats_v2.json")

PATCH, STRIDE = 64, 64
THRESH = 0.65
CONFIRM_FRAC = 0.25   # aggregate FP-in-recession >= this => peak-window omission likely

# ── 1. Rasterize recession layer onto the shared grid (same transform as peak label) ──
with rasterio.open(PEAK_LABEL) as ds:
    H, W = ds.height, ds.width
    transform, crs, bounds = ds.transform, ds.crs, ds.bounds
    peak = ds.read(1).astype(np.uint8)

gdf = gpd.read_file(RECESS_SHP, bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top))
gdf = gdf.to_crs(crs)
recess = rasterize(((g, 1) for g in gdf.geometry if g is not None),
                   out_shape=(H, W), transform=transform, fill=0, dtype="uint8", all_touched=False)
with rasterio.open(RECESS_OUT, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                   crs=crs, transform=transform, compress="lzw", nodata=0) as dst:
    dst.write(recess, 1)
print(f"recession raster saved: flood frac {recess.mean():.6f} "
      f"(peak {peak.mean():.6f}); features in window {len(gdf)}")

# ── 2. Reproduce v2 test-patch geometry; collect peak + recession label patches ──
c1, c2 = 0.60 * W, 0.80 * W
peak_patches, recess_patches = [], []
for r in range(0, H - PATCH + 1, STRIDE):
    for c in range(0, W - PATCH + 1, STRIDE):
        if (c < c1 < c + PATCH) or (c < c2 < c + PATCH):      # straddle drop
            continue
        center_c = c + PATCH / 2.0
        if center_c < c1:      sp = "train"
        elif center_c < c2:    sp = "val"
        else:                  sp = "test"
        if sp != "test":
            continue
        peak_patches.append(peak[r:r + PATCH, c:c + PATCH].copy())
        recess_patches.append(recess[r:r + PATCH, c:c + PATCH].copy())
peak_patches = np.stack(peak_patches).astype(np.uint8)
recess_patches = np.stack(recess_patches).astype(np.uint8)

y_test = np.load(os.path.join(PROC, "feni_y_test_v2.npy")).astype(np.uint8)
assert peak_patches.shape == y_test.shape, (peak_patches.shape, y_test.shape)
assert np.array_equal(peak_patches, y_test), "peak-label geometry does NOT match feni_y_test_v2 — alignment broken"
print(f"alignment OK: {peak_patches.shape[0]} test patches match feni_y_test_v2 exactly "
      f"(validity skipped 0, confirmed)")

# ── 3. v2 U-Net @0.65 predictions ──
ns = json.load(open(NORM))
M = np.array(ns["mean"], np.float32); S = np.array(ns["std"], np.float32)
X = np.load(os.path.join(PROC, "feni_X_test_v2.npy")).astype(np.float32)
Xs = ((np.where(np.isfinite(X), X, M) - M) / S).astype(np.float32)
model = tf.keras.models.load_model(UNET, compile=False)
prob = model.predict(Xs, batch_size=32, verbose=0)[..., 1]
pred = (prob > THRESH).astype(np.uint8)

true_frac = (y_test == 1).reshape(len(y_test), -1).mean(1)
pred_frac = pred.reshape(len(pred), -1).mean(1)
fp_idx = np.where((true_frac == 0.0) & (pred_frac > 0.5))[0]
print(f"zero-true-flood FP patches (true frac 0 & pred frac>0.5 @{THRESH}): n={len(fp_idx)}")

# ── 4. FP-pixel fraction that is flood in the recession layer ──
records = []
tot_fp, tot_fp_in = 0, 0
for i in fp_idx:
    fp = (pred[i] == 1) & (y_test[i] == 0)          # true==0 here, so FP == all predicted flood
    rec = recess_patches[i] == 1
    n_fp = int(fp.sum()); n_in = int((fp & rec).sum())
    tot_fp += n_fp; tot_fp_in += n_in
    records.append({
        "patch_index": int(i),
        "true_flood_frac": float(true_frac[i]),
        "pred_flood_frac_at_0.65": float(pred_frac[i]),
        "fp_pixels": n_fp,
        "fp_in_recession_flood": n_in,
        "frac_fp_in_recession": (n_in / n_fp) if n_fp else float("nan"),
        "recession_flood_frac_of_patch": float(rec.mean()),
    })
agg = tot_fp_in / tot_fp if tot_fp else float("nan")
per_patch = np.array([r["frac_fp_in_recession"] for r in records])
verdict = ("peak-window label omission likely explains part of the FP set"
           if agg >= CONFIRM_FRAC else
           "FPs consistent with soil-moisture / SAR change-detection ambiguity")
print(f"AGGREGATE frac FP pixels in recession flood = {agg:.4f}  "
      f"(per-patch mean {np.nanmean(per_patch):.4f}, max {np.nanmax(per_patch):.4f})")
print("VERDICT:", verdict)

ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = os.path.join(ROOT, "results", ts, "recession_crosscheck")
os.makedirs(out_dir, exist_ok=True)
summary = {
    "timestamp_utc": ts, "seed": SEED, "model": UNET, "threshold": THRESH,
    "recession_layer": os.path.basename(RECESS_SHP),
    "recession_raster": RECESS_OUT,
    "recession_flood_frac_scene": float(recess.mean()),
    "peak_flood_frac_scene": float(peak.mean()),
    "n_fp_patches": int(len(fp_idx)),
    "fp_patch_selection": "true flood frac 0 AND pred flood frac > 0.5 @0.65",
    "aggregate_frac_fp_in_recession": agg,
    "per_patch_mean_frac": float(np.nanmean(per_patch)),
    "per_patch_max_frac": float(np.nanmax(per_patch)),
    "confirm_fraction": CONFIRM_FRAC,
    "verdict": verdict,
    "records": records,
}
json.dump(summary, open(os.path.join(out_dir, "recession_crosscheck.json"), "w"), indent=2)
print("OUT:", out_dir)
