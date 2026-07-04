"""STEP 3 - 64x64 non-overlapping patches, 4ch S1 change-detection, spatial thirds split.
Channels: [VV_flood, VH_flood, VV_pre, VH_pre] (dB). Split by longitude == column (regular EPSG:4326 grid):
train=west 60%, val=mid 20%, test=east 20%. Patches straddling a boundary are dropped. Skip <50% valid.
RAM: peak ~ (H*W*4 ch float32) 833 MB + patch arrays ~0.8 GB. One raster opened at a time for the read.
"""
import os, json, datetime, argparse
import numpy as np
import rasterio

DATA = r"d:\Hamim\DisasterShield\data\Feni_2024_10m"
PROC = r"d:\Hamim\DisasterShield\data\processed"

ap = argparse.ArgumentParser()
ap.add_argument("--flood", default=os.path.join(DATA, "Feni_S1_Flood_Aug2024_10m.tif"),
                help="path to S1 flood raster used as the flood channels")
ap.add_argument("--tag", default="v1", help="version tag for output filenames")
args = ap.parse_args()

S1_FLOOD = args.flood
TAG = args.tag
S1_PRE = os.path.join(DATA, "Feni_S1_PreFlood_May2024_10m.tif")
LABEL = os.path.join(PROC, "feni_flood_label_10m.tif")

PATCH = 64
STRIDE = 64
MIN_VALID = 0.50
SEED = 42
np.random.seed(SEED)

with rasterio.open(S1_FLOOD) as ds:
    H, W = ds.height, ds.width
    transform = ds.transform
    crs = ds.crs
    bounds = ds.bounds

est_gb = H * W * 4 * 4 / 1e9
print(f"grid {W}x{H}; channels array est {est_gb:.2f} GB float32")

# --- load 4 channels one band at a time into (H,W,4) float32 ---
chan = np.empty((H, W, 4), dtype=np.float32)
with rasterio.open(S1_FLOOD) as ds:
    chan[:, :, 0] = ds.read(1).astype(np.float32)  # VV flood
    chan[:, :, 1] = ds.read(2).astype(np.float32)  # VH flood
with rasterio.open(S1_PRE) as ds:
    chan[:, :, 2] = ds.read(1).astype(np.float32)  # VV pre
    chan[:, :, 3] = ds.read(2).astype(np.float32)  # VH pre
with rasterio.open(LABEL) as ds:
    y_full = ds.read(1).astype(np.uint8)

# valid = all 4 channels finite AND not all-zero (0-fill = SAR mask/edge)
finite = np.all(np.isfinite(chan), axis=2)
nonzero = np.any(chan != 0, axis=2)
valid_full = finite & nonzero

# --- longitude split boundaries as fractional columns (regular grid: lon linear in col) ---
c1 = 0.60 * W   # west|mid  = 4675.8
c2 = 0.80 * W   # mid|east  = 6234.4
def lon_of_col(c):
    return bounds.left + (c / W) * (bounds.right - bounds.left)

splits = {"train": [], "val": [], "test": []}     # channel patches
splits_y = {"train": [], "val": [], "test": []}   # label patches
counts = {"train": 0, "val": 0, "test": 0}
skipped_invalid = 0
skipped_straddle = 0

for r in range(0, H - PATCH + 1, STRIDE):
    for c in range(0, W - PATCH + 1, STRIDE):
        # straddle test: boundary column inside [c, c+PATCH)
        if (c < c1 < c + PATCH) or (c < c2 < c + PATCH):
            skipped_straddle += 1
            continue
        vfrac = valid_full[r:r + PATCH, c:c + PATCH].mean()
        if vfrac < MIN_VALID:
            skipped_invalid += 1
            continue
        center_c = c + PATCH / 2.0
        if center_c < c1:
            sp = "train"
        elif center_c < c2:
            sp = "val"
        else:
            sp = "test"
        splits[sp].append(chan[r:r + PATCH, c:c + PATCH, :].copy())
        splits_y[sp].append(y_full[r:r + PATCH, c:c + PATCH].copy())
        counts[sp] += 1

del chan  # free 833 MB before stacking

manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "tag": TAG, "flood_input": os.path.basename(S1_FLOOD),
    "channels": ["VV_flood", "VH_flood", "VV_pre", "VH_pre"],
    "units": "dB", "patch": PATCH, "stride": STRIDE, "min_valid": MIN_VALID, "seed": SEED,
    "grid": {"width": W, "height": H, "crs": str(crs),
             "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top]},
    "split_boundaries": {
        "method": "longitude thirds by column (west60/mid20/east20)",
        "col_west_mid": c1, "col_mid_east": c2,
        "lon_west_mid": lon_of_col(c1), "lon_mid_east": lon_of_col(c2),
        "train_cols": [0, c1], "val_cols": [c1, c2], "test_cols": [c2, W],
    },
    "skipped_straddle": skipped_straddle, "skipped_invalid_lt50pct": skipped_invalid,
    "splits": {},
}

for sp in ["train", "val", "test"]:
    X = np.stack(splits[sp], axis=0).astype(np.float32) if counts[sp] else np.empty((0, PATCH, PATCH, 4), np.float32)
    y = np.stack(splits_y[sp], axis=0).astype(np.uint8) if counts[sp] else np.empty((0, PATCH, PATCH), np.uint8)
    np.save(os.path.join(PROC, f"feni_X_{sp}_{TAG}.npy"), X)
    np.save(os.path.join(PROC, f"feni_y_{sp}_{TAG}.npy"), y)
    flood_frac = float(y.mean()) if y.size else 0.0
    manifest["splits"][sp] = {
        "n_patches": int(counts[sp]), "X_shape": list(X.shape), "y_shape": list(y.shape),
        "flood_pixel_fraction": round(flood_frac, 6),
        "flood_pixels": int(y.sum()), "total_pixels": int(y.size),
    }
    del X, y, splits[sp], splits_y[sp]

with open(os.path.join(PROC, f"feni_patches_manifest_{TAG}.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
