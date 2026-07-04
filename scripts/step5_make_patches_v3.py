"""STEP 5a - v3 6-channel physics-informed patches.

Channels (6): [VV_flood, VH_flood, VV_pre, VH_pre, dVV, dVH] in raw dB.
  dVV = VV_flood - VV_pre  (computed in RAW dB, BEFORE any normalization)
  dVH = VH_flood - VH_pre

Identical grid / split / straddle / min-valid / seed to step3_make_patches.py (v2):
  64x64 non-overlapping, longitude thirds split (train west 60 / val mid 20 / test east 20),
  straddling patches dropped, skip patches <50% valid. Seed 42. Flood input = v2 window-matched.

Also computes TRAIN-ONLY per-channel mean/std for all 6 channels (nanmean/nanstd, same
convention as train_feni.py) -> data/processed/norm_stats_v3.json.

Outputs: data/processed/feni_{X,y}_{train,val,test}_v3.npy ; feni_patches_manifest_v3.json ;
         data/processed/norm_stats_v3.json.
RAM: base 4ch array 833 MB + 2 delta planes (~417 MB) = ~1.25 GB peak on the full grid.
"""
import os, json, datetime, argparse
import numpy as np
import rasterio

DATA = r"d:\Hamim\DisasterShield\data\Feni_2024_10m"
PROC = r"d:\Hamim\DisasterShield\data\processed"

ap = argparse.ArgumentParser()
ap.add_argument("--flood", default=os.path.join(DATA, "Feni_S1_Flood_18to26Aug2024_10m.tif"),
                help="S1 flood raster used as the flood channels (default = v2 window-matched)")
ap.add_argument("--tag", default="v3", help="version tag for output filenames")
args = ap.parse_args()

S1_FLOOD = args.flood
TAG = args.tag
S1_PRE = os.path.join(DATA, "Feni_S1_PreFlood_May2024_10m.tif")
LABEL = os.path.join(PROC, "feni_flood_label_10m.tif")

CHANNELS = ["VV_flood", "VH_flood", "VV_pre", "VH_pre", "dVV", "dVH"]
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

est_gb = H * W * 6 * 4 / 1e9
print(f"grid {W}x{H}; 6ch channels array est {est_gb:.2f} GB float32")

# --- load 6 channels into (H,W,6) float32; deltas in RAW dB BEFORE normalization ---
chan = np.empty((H, W, 6), dtype=np.float32)
with rasterio.open(S1_FLOOD) as ds:
    chan[:, :, 0] = ds.read(1).astype(np.float32)  # VV flood
    chan[:, :, 1] = ds.read(2).astype(np.float32)  # VH flood
with rasterio.open(S1_PRE) as ds:
    chan[:, :, 2] = ds.read(1).astype(np.float32)  # VV pre
    chan[:, :, 3] = ds.read(2).astype(np.float32)  # VH pre
chan[:, :, 4] = chan[:, :, 0] - chan[:, :, 2]      # dVV = VV_flood - VV_pre (raw dB)
chan[:, :, 5] = chan[:, :, 1] - chan[:, :, 3]      # dVH = VH_flood - VH_pre (raw dB)
with rasterio.open(LABEL) as ds:
    y_full = ds.read(1).astype(np.uint8)

# valid mask uses the 4 BASE channels only (identical to step3): finite AND not all-zero.
# Deltas are derived from the base channels, so they add no new validity constraint.
base = chan[:, :, :4]
finite = np.all(np.isfinite(base), axis=2)
nonzero = np.any(base != 0, axis=2)
valid_full = finite & nonzero

# --- longitude split boundaries as fractional columns (identical to step3) ---
c1 = 0.60 * W   # west|mid
c2 = 0.80 * W   # mid|east
def lon_of_col(c):
    return bounds.left + (c / W) * (bounds.right - bounds.left)

splits = {"train": [], "val": [], "test": []}
splits_y = {"train": [], "val": [], "test": []}
counts = {"train": 0, "val": 0, "test": 0}
skipped_invalid = 0
skipped_straddle = 0

for r in range(0, H - PATCH + 1, STRIDE):
    for c in range(0, W - PATCH + 1, STRIDE):
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

del chan

manifest = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "tag": TAG, "flood_input": os.path.basename(S1_FLOOD),
    "channels": CHANNELS,
    "delta_note": "dVV/dVH computed in RAW dB (VV_flood-VV_pre, VH_flood-VH_pre) BEFORE normalization",
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

# stack splits, save arrays; keep the TRAIN stack around to compute norm stats
X_train_for_stats = None
for sp in ["train", "val", "test"]:
    X = np.stack(splits[sp], axis=0).astype(np.float32) if counts[sp] else np.empty((0, PATCH, PATCH, 6), np.float32)
    y = np.stack(splits_y[sp], axis=0).astype(np.uint8) if counts[sp] else np.empty((0, PATCH, PATCH), np.uint8)
    np.save(os.path.join(PROC, f"feni_X_{sp}_{TAG}.npy"), X)
    np.save(os.path.join(PROC, f"feni_y_{sp}_{TAG}.npy"), y)
    flood_frac = float(y.mean()) if y.size else 0.0
    manifest["splits"][sp] = {
        "n_patches": int(counts[sp]), "X_shape": list(X.shape), "y_shape": list(y.shape),
        "flood_pixel_fraction": round(flood_frac, 6),
        "flood_pixels": int(y.sum()), "total_pixels": int(y.size),
    }
    if sp == "train":
        X_train_for_stats = X  # retain
    else:
        del X
    del y, splits[sp], splits_y[sp]

# --- TRAIN-ONLY norm stats for all 6 channels (nanmean/nanstd, matches train_feni.py) ---
flat = X_train_for_stats.reshape(-1, 6)
n_nonfinite_train = int((~np.isfinite(flat)).sum())
means = np.nanmean(flat, axis=0)
stds = np.nanstd(flat, axis=0)
stds = np.where(stds < 1e-6, 1.0, stds)
norm_stats = {
    "channels": CHANNELS, "tag": TAG, "seed": SEED,
    "computed_on": "train split only (nanmean/nanstd)",
    "mean": [float(m) for m in means],
    "std": [float(s) for s in stds],
    "nonfinite_pixels_filled_with_mean": True,
    "n_nonfinite_train_channel_values": n_nonfinite_train,
}
with open(os.path.join(PROC, f"norm_stats_{TAG}.json"), "w") as f:
    json.dump(norm_stats, f, indent=2)
manifest["norm_stats"] = norm_stats

with open(os.path.join(PROC, f"feni_patches_manifest_{TAG}.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
print("\nnorm means:", norm_stats["mean"])
print("norm stds :", norm_stats["std"])
print("non-finite train channel-values:", n_nonfinite_train)
