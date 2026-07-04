"""STEP 2 - Rasterize UNOSAT flood label to the shared S1 grid, sanity checks, permanent-water noise.
RAM-safe: read only what is needed; label raster is uint8 (~52 MB), VV band float32 (~208 MB)."""
import os, json, datetime
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = r"d:\Hamim\DisasterShield\data\Feni_2024_10m"
LAB = r"d:\Hamim\DisasterShield\data\labels_unosat"
PROC = r"d:\Hamim\DisasterShield\data\processed"
S1_FLOOD = os.path.join(DATA, "Feni_S1_Flood_Aug2024_10m.tif")
S1_PRE = os.path.join(DATA, "Feni_S1_PreFlood_May2024_10m.tif")
LABEL_SHP = os.path.join(LAB, "S1_20240818_20240826_FloodExtent_Bangladesh.shp")
LABEL_OUT = os.path.join(PROC, "feni_flood_label_10m.tif")
VV_PERM_THRESH_DB = -16.0

os.makedirs(PROC, exist_ok=True)
ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RES = os.path.join(r"d:\Hamim\DisasterShield\results", ts)
os.makedirs(RES, exist_ok=True)

report = {"timestamp_utc": ts}

# --- grid from the chosen input raster (S1 flood) ---
with rasterio.open(S1_FLOOD) as ds:
    H, W = ds.height, ds.width
    transform = ds.transform
    crs = ds.crs
    bounds = ds.bounds
report["grid"] = {"width": W, "height": H, "crs": str(crs),
                  "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top]}

# --- read only polygons intersecting the raster window (bbox filter -> huge memory saving) ---
bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
gdf = gpd.read_file(LABEL_SHP, bbox=bbox)
gdf = gdf.to_crs(crs)
report["label_source"] = os.path.basename(LABEL_SHP)
report["n_features_in_window"] = int(len(gdf))

# --- rasterize (1 = flood) onto exact grid ---
shapes = ((geom, 1) for geom in gdf.geometry if geom is not None)
label = rasterize(shapes, out_shape=(H, W), transform=transform,
                  fill=0, dtype="uint8", all_touched=False)

with rasterio.open(LABEL_OUT, "w", driver="GTiff", height=H, width=W, count=1,
                   dtype="uint8", crs=crs, transform=transform, compress="lzw",
                   nodata=0) as dst:
    dst.write(label, 1)
report["label_raster"] = LABEL_OUT

# --- 2b: flooded-pixel fraction ---
n_flood = int(label.sum())
n_tot = int(label.size)
report["flood_pixel_fraction"] = round(n_flood / n_tot, 6)
report["flood_pixels"] = n_flood
report["total_pixels"] = n_tot

# --- 2b: overlay PNG at reduced resolution (decimated read of VV flood) ---
DECIM = 8
oh, ow = H // DECIM, W // DECIM
with rasterio.open(S1_FLOOD) as ds:
    vv_small = ds.read(1, out_shape=(oh, ow)).astype(np.float32)  # VV dB, downsampled
lab_small = label[::DECIM, ::DECIM][:oh, :ow]
fig, ax = plt.subplots(1, 1, figsize=(10, 9))
ax.imshow(vv_small, cmap="gray", vmin=-22, vmax=0)
mask = np.ma.masked_where(lab_small == 0, lab_small)
ax.imshow(mask, cmap="autumn", alpha=0.45)
ax.set_title(f"UNOSAT flood label (red) over S1 VV dB (gray) — {report['flood_pixel_fraction']*100:.2f}% flood")
ax.axis("off")
overlay_png = os.path.join(RES, "label_overlay_check.png")
fig.savefig(overlay_png, dpi=110, bbox_inches="tight")
plt.close(fig)
report["overlay_png"] = overlay_png

# --- 2c: permanent-water noise (S1 pre-flood VV < -16 dB, fraction OUTSIDE UNOSAT flood polygons) ---
with rasterio.open(S1_PRE) as ds:
    vv_pre = ds.read(1).astype(np.float32)  # VV dB full res
finite = np.isfinite(vv_pre)
perm_water = finite & (vv_pre < VV_PERM_THRESH_DB)
del vv_pre
n_perm = int(perm_water.sum())
perm_outside = perm_water & (label == 0)
n_perm_out = int(perm_outside.sum())
report["permanent_water"] = {
    "proxy": "S1 pre-flood VV < -16 dB",
    "perm_water_pixels": n_perm,
    "perm_water_fraction_of_scene": round(n_perm / n_tot, 6),
    "perm_water_pixels_outside_UNOSAT": n_perm_out,
    "fraction_perm_water_outside_UNOSAT": round(n_perm_out / n_perm, 6) if n_perm else None,
}

with open(os.path.join(RES, "step2_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
