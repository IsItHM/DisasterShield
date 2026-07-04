"""
DisasterShield-X — DEMO BUILD (Phase 2 closeout, Part 2).

Produces a single self-contained `demo/index.html`:
  (a) Full-scene inference with the FINAL v2 4-channel U-Net over the entire 7793x6684
      Feni scene. Overlapping 64x64 tiles at stride 32; probabilities averaged in overlaps;
      threshold 0.65 for the binary flood map. Row-band streaming -> the full 4-channel float
      array is never materialised at once. Stitched probability map saved as a downsampled
      GeoTIFF.
  (b) Folium map with three toggleable layers: Sentinel-2 pre-flood RGB, predicted flood
      (blue), UNOSAT observed flood extent (red outline).
  (c) Fixed info panel; metric numbers read AT BUILD TIME from the frozen CSVs.
  (d) Headline: predicted flooded km2 over the full scene vs UNOSAT 905.53 km2.

Numbers only from frozen files. No training. Seed 42.

Inputs (frozen):
  model   results/20260703T054304Z/feni_unet_best.keras   (v2, FINAL)
  norm    data/processed/norm_stats_v2.json
  S1 v2   data/Feni_2024_10m/Feni_S1_Flood_18to26Aug2024_10m.tif  (VV,VH flood)
  S1 pre  data/Feni_2024_10m/Feni_S1_PreFlood_May2024_10m.tif     (VV,VH pre)
  S2 pre  data/Feni_2024_10m/Feni_S2_PreFlood_May2024_10m.tif     (B2,B3,B4,B8)
  label   data/processed/feni_flood_label_10m.tif                 (UNOSAT, 1=flood)
  metrics results/20260703T165223Z/threshold_fair_v3.csv (U-Net v2 @val-tuned row)
          results/20260703T052548Z/baseline_comparison.csv
"""
import os, io, csv, json, base64, datetime, time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
import folium

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

ROOT = r"d:\Hamim\DisasterShield"
MODEL = os.path.join(ROOT, r"results\20260703T054304Z\feni_unet_best.keras")
NORM = os.path.join(ROOT, r"data\processed\norm_stats_v2.json")
S1_FLOOD = os.path.join(ROOT, r"data\Feni_2024_10m\Feni_S1_Flood_18to26Aug2024_10m.tif")
S1_PRE = os.path.join(ROOT, r"data\Feni_2024_10m\Feni_S1_PreFlood_May2024_10m.tif")
S2_PRE = os.path.join(ROOT, r"data\Feni_2024_10m\Feni_S2_PreFlood_May2024_10m.tif")
LABEL = os.path.join(ROOT, r"data\processed\feni_flood_label_10m.tif")
THRESH_CSV = os.path.join(ROOT, r"results\20260703T165223Z\threshold_fair_v3.csv")
BASE_CSV = os.path.join(ROOT, r"results\20260703T052548Z\baseline_comparison.csv")

TILE = 64
STRIDE = 32
THRESHOLD = 0.65
OVERLAY_LONGSIDE = 1500          # overlay long-side px; auto-reduced if HTML > SIZE_LIMIT
SIZE_LIMIT_MB = 50
UNOSAT_KM2 = 905.53              # frozen vector area, Step 1d

DEMO_DIR = os.path.join(ROOT, "demo")
os.makedirs(DEMO_DIR, exist_ok=True)
TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
BUILD_DIR = os.path.join(ROOT, "results", TS, "demo_build")
os.makedirs(BUILD_DIR, exist_ok=True)


# ── per-row geodesic pixel area (WGS84), for the km2 headline ────────────────
def per_row_pixel_area_m2(H, top_lat, a_deg):
    lat = top_lat - (np.arange(H) + 0.5) * a_deg
    phi = np.radians(lat)
    m_deg_lat = 111132.92 - 559.82*np.cos(2*phi) + 1.175*np.cos(4*phi) - 0.0023*np.cos(6*phi)
    m_deg_lon = 111412.84*np.cos(phi) - 93.5*np.cos(3*phi) + 0.118*np.cos(5*phi)
    return (a_deg * m_deg_lon) * (a_deg * m_deg_lat)   # m^2 / pixel, length H


# ── (a) full-scene inference, row-band streaming ────────────────────────────
def full_scene_inference(model, means, stds):
    with rasterio.open(S1_FLOOD) as ds:
        H, W = ds.height, ds.width
        bounds = ds.bounds
        a_deg = ds.transform.a
    ty_list = sorted(set(list(range(0, H - TILE, STRIDE)) + [H - TILE]))
    tx_list = sorted(set(list(range(0, W - TILE, STRIDE)) + [W - TILE]))
    tx_arr = np.array(tx_list)
    print(f"scene {H}x{W}  strips={len(ty_list)}  tiles/strip={len(tx_list)}  "
          f"total tiles={len(ty_list)*len(tx_list)}")

    prob_sum = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.uint16)
    fds = rasterio.open(S1_FLOOD); pds = rasterio.open(S1_PRE)
    t0 = time.time()
    for i, ty in enumerate(ty_list):
        win = Window(0, ty, W, TILE)
        fl = fds.read(window=win).astype(np.float32)     # (2,64,W) VV,VH flood
        pr = pds.read(window=win).astype(np.float32)     # (2,64,W) VV,VH pre
        band = np.stack([fl[0], fl[1], pr[0], pr[1]], axis=-1)   # (64,W,4)
        tiles = np.stack([band[:, tx:tx + TILE, :] for tx in tx_list], 0)  # (n,64,64,4)
        tiles = np.where(np.isfinite(tiles), tiles, means)        # mean-fill non-finite
        tiles = ((tiles - means) / stds).astype(np.float32)       # standardize (train stats)
        probs = model.predict_on_batch(tiles)[..., 1]             # (n,64,64) water prob
        for j, tx in enumerate(tx_list):
            prob_sum[ty:ty + TILE, tx:tx + TILE] += probs[j]
            count[ty:ty + TILE, tx:tx + TILE] += 1
        if i % 20 == 0 or i == len(ty_list) - 1:
            el = time.time() - t0
            print(f"  strip {i+1:3d}/{len(ty_list)}  ty={ty:5d}  {el:6.1f}s", flush=True)
    fds.close(); pds.close()
    prob = prob_sum / np.maximum(count, 1)
    del prob_sum, count
    return prob, H, W, bounds, a_deg


def to_data_uri(rgba):
    im = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), buf.getbuffer().nbytes


def build_pred_overlay(prob, W2, H2):
    prob_ds = np.array(Image.fromarray(prob).resize((W2, H2), Image.BILINEAR))
    pred = prob_ds > THRESHOLD
    rgba = np.zeros((H2, W2, 4), np.uint8)
    rgba[pred] = [20, 90, 255, 130]           # semi-transparent blue FILL
    return rgba, pred


def build_s2_overlay(W2, H2):
    with rasterio.open(S2_PRE) as ds:
        rgb = ds.read([3, 2, 1], out_shape=(3, H2, W2),
                      resampling=Resampling.bilinear).astype(np.float32)  # B4,B3,B2
    valid = ~((rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0))
    out = np.zeros((H2, W2, 3), np.uint8)
    for b in range(3):
        v = rgb[b][valid]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        norm = np.clip((rgb[b] - lo) / max(hi - lo, 1e-6), 0, 1)
        out[..., b] = (np.power(norm, 0.8) * 255).astype(np.uint8)   # gamma 0.8 brighten
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    return np.dstack([out, alpha]), float(valid.mean())


def build_label_overlay(W2, H2):
    with rasterio.open(LABEL) as ds:
        lab = ds.read(1, out_shape=(H2, W2), resampling=Resampling.nearest)
    lab = (lab == 1)
    edge = lab & ~binary_erosion(lab, iterations=1)
    edge = binary_dilation(edge, iterations=1)         # 1-px outline
    rgba = np.zeros((H2, W2, 4), np.uint8)
    rgba[edge] = [220, 20, 20, 150]                    # thin, semi-transparent red OUTLINE (no fill)
    return rgba, lab


def build_agreement_overlay(pred, lab):
    """Per-pixel model-vs-UNOSAT agreement: TP dark blue, model-only orange, UNOSAT-only magenta."""
    H2, W2 = pred.shape
    rgba = np.zeros((H2, W2, 4), np.uint8)
    rgba[pred & lab] = [10, 40, 140, 190]      # both agree: flooded (true positive)
    rgba[pred & ~lab] = [255, 140, 0, 190]     # model only (the diagnosed disagreement zones)
    rgba[~pred & lab] = [220, 0, 190, 190]     # UNOSAT only (missed flood)
    return rgba, int((pred & lab).sum()), int((pred & ~lab).sum()), int((~pred & lab).sum())


def label_area_km2(H, top_lat, a_deg):
    """UNOSAT label pixel area cross-check, per-row geodesic."""
    area_row = per_row_pixel_area_m2(H, top_lat, a_deg)
    with rasterio.open(LABEL) as ds:
        total = 0.0
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win)
            rows = np.arange(win.row_off, win.row_off + win.height)
            total += float((arr == 1).sum(axis=1) @ area_row[rows])
    return total / 1e6


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    t_start = time.time()
    ns = json.load(open(NORM))
    means = np.array(ns["mean"], np.float32)
    stds = np.array(ns["std"], np.float32)
    print("loading FINAL v2 model:", MODEL)
    model = tf.keras.models.load_model(MODEL, compile=False)

    prob, H, W, bounds, a_deg = full_scene_inference(model, means, stds)
    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

    # full-res binary + km2 (per-row geodesic area)
    area_row = per_row_pixel_area_m2(H, north, a_deg)
    binary = prob > THRESHOLD
    pred_km2 = float((binary.sum(axis=1) @ area_row) / 1e6)
    lab_km2 = label_area_km2(H, north, a_deg)
    pred_px = int(binary.sum())
    with rasterio.open(LABEL) as ds:
        lab_full = ds.read(1) == 1
    tp_km2 = float(((binary & lab_full).sum(axis=1) @ area_row) / 1e6)
    fp_km2 = float(((binary & ~lab_full).sum(axis=1) @ area_row) / 1e6)
    fn_km2 = float(((~binary & lab_full).sum(axis=1) @ area_row) / 1e6)
    del lab_full
    print(f"predicted flood: {pred_km2:.1f} km2 ({pred_px:,} px)  |  "
          f"label check: {lab_km2:.1f} km2 (UNOSAT vector {UNOSAT_KM2})")
    print(f"agreement full-res km2: TP(both)={tp_km2:.1f} FP(model-only)={fp_km2:.1f} "
          f"FN(UNOSAT-only)={fn_km2:.1f}")

    # downsampled probability GeoTIFF (frozen artifact)
    f = max(H, W) / OVERLAY_LONGSIDE
    W2, H2 = round(W / f), round(H / f)
    prob_small = np.array(Image.fromarray(prob).resize((W2, H2), Image.BILINEAR)).astype(np.float32)
    tif_path = os.path.join(BUILD_DIR, "pred_prob_downsampled.tif")
    with rasterio.open(tif_path, "w", driver="GTiff", height=H2, width=W2, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(west, south, east, north, W2, H2),
                       compress="lzw") as dst:
        dst.write(prob_small, 1)
    print("saved", tif_path)

    # overlays
    pred_rgba, pred_ds = build_pred_overlay(prob, W2, H2)
    del prob
    s2_rgba, s2_valid = build_s2_overlay(W2, H2)
    lab_rgba, lab_ds = build_label_overlay(W2, H2)
    agr_rgba, _, _, _ = build_agreement_overlay(pred_ds, lab_ds)
    for name, arr in [("overlay_pred", pred_rgba), ("overlay_s2", s2_rgba),
                      ("overlay_label", lab_rgba), ("overlay_agreement", agr_rgba)]:
        Image.fromarray(arr, "RGBA").save(os.path.join(BUILD_DIR, name + ".png"), optimize=True)
    pred_uri, n1 = to_data_uri(pred_rgba)
    s2_uri, n2 = to_data_uri(s2_rgba)
    lab_uri, n3 = to_data_uri(lab_rgba)
    agr_uri, n4 = to_data_uri(agr_rgba)
    print(f"overlay sizes KB  pred={n1/1e3:.0f}  s2={n2/1e3:.0f}  label={n3/1e3:.0f}  "
          f"agreement={n4/1e3:.0f}  (overlay grid {W2}x{H2}, S2 valid {s2_valid*100:.1f}%)")

    # ── (c) info-panel numbers read at build time ───────────────────────────
    tf_rows = read_csv_rows(THRESH_CSV)
    v2 = next(r for r in tf_rows if r["method"] == "U-Net v2 @val-tuned")
    iou, f1 = float(v2["test_IoU"]), float(v2["test_F1"])
    b = read_csv_rows(BASE_CSV)
    bget = lambda p: next(r for r in b if r["method"].startswith(p))
    A, B, C = bget("A_"), bget("B_"), bget("C_")
    diff = pred_km2 - UNOSAT_KM2
    diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"

    panel = f"""
<div style="position:fixed;top:12px;left:12px;z-index:9999;width:370px;
 background:rgba(255,255,255,0.94);padding:13px 15px;border-radius:9px;
 font-family:Arial,Helvetica,sans-serif;color:#1a1a1a;
 box-shadow:0 1px 8px rgba(0,0,0,0.35);font-size:12px;line-height:1.45;">
 <div style="font-size:15px;font-weight:700;margin-bottom:2px;">
   DisasterShield-X &mdash; Aug 2024 Feni flood, Bangladesh</div>
 <div style="color:#555;margin-bottom:7px;">Sentinel-1 change detection &middot; U-Net &middot; 10 m</div>
 <div style="background:#eef4ff;border-left:4px solid #1450ff;padding:6px 8px;margin-bottom:7px;">
   <b>Predicted flood: {pred_km2:.1f} km&sup2;</b> over the full scene<br>
   UNOSAT observed: {UNOSAT_KM2:.1f} km&sup2; &nbsp;(<b>{diff_str} km&sup2;</b>)</div>
 <div><b>Test IoU {iou:.2f} / F1 {f1:.2f}</b> vs UNOSAT ground truth
   (spatially held-out region)</div>
 <div style="color:#555;margin-top:3px;">Baselines (test IoU): logreg {float(C['test_IoU']):.2f}
   &middot; VV-change {float(A['test_IoU']):.2f} &middot; VV-abs {float(B['test_IoU']):.2f}</div>
 <div style="margin-top:7px;padding-top:6px;border-top:1px solid #ddd;color:#444;font-style:italic;">
   Trained W / validated mid / tested E of the same event &mdash; single-event model;
   multi-event generalization in progress.</div>
 <div style="margin-top:7px;padding-top:6px;border-top:1px solid #ddd;font-size:11px;color:#444;">
   <b>Agreement map</b> (default layer):<br>
   <span style="color:#0a2882;">&#9632;</span> both agree: flooded ({tp_km2:.0f} km&sup2;) &nbsp;
   <span style="color:#ff8c00;">&#9632;</span> model only ({fp_km2:.0f} km&sup2;)<br>
   <span style="color:#dc00be;">&#9632;</span> UNOSAT only ({fn_km2:.0f} km&sup2;)<br>
   Toggle on: <span style="color:#1450ff;">&#9632;</span> predicted flood (blue fill) &middot;
   <span style="color:#dc1414;">&#9633;</span> UNOSAT outline &middot; base S2 May 2024</div>
</div>"""

    # ── (b) folium map ──────────────────────────────────────────────────────
    fbounds = [[south, west], [north, east]]
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=10,
                   tiles="CartoDB positron", control_scale=True)
    # z-order bottom->top: S2 base -> agreement -> prediction -> UNOSAT outline
    folium.raster_layers.ImageOverlay(s2_uri, bounds=fbounds, opacity=1.0, show=True,
        name="Sentinel-2 pre-flood (May 2024)").add_to(m)
    folium.raster_layers.ImageOverlay(agr_uri, bounds=fbounds, opacity=1.0, show=True,
        name="Agreement map (model vs UNOSAT)").add_to(m)
    folium.raster_layers.ImageOverlay(pred_uri, bounds=fbounds, opacity=1.0, show=False,
        name="DisasterShield-X predicted flood").add_to(m)
    folium.raster_layers.ImageOverlay(lab_uri, bounds=fbounds, opacity=1.0, show=False,
        name="UNOSAT observed flood extent").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(panel))
    m.fit_bounds(fbounds)

    out_html = os.path.join(DEMO_DIR, "index.html")
    m.save(out_html)
    size_mb = os.path.getsize(out_html) / 1e6
    print(f"saved {out_html}  ({size_mb:.1f} MB)")

    manifest = {
        "timestamp_utc": TS, "seed": SEED, "model": MODEL, "threshold": THRESHOLD,
        "tile": TILE, "stride": STRIDE, "scene": [H, W], "overlay_grid": [H2, W2],
        "predicted_flood_km2": pred_km2, "predicted_flood_pixels": pred_px,
        "unosat_vector_km2": UNOSAT_KM2, "label_raster_km2_check": lab_km2,
        "diff_km2": diff, "s2_valid_frac": s2_valid,
        "agreement_km2": {"both_flood_TP": tp_km2, "model_only_FP": fp_km2,
                          "unosat_only_FN": fn_km2},
        "layers": ["Sentinel-2 pre-flood (May 2024)", "Agreement map (model vs UNOSAT)",
                   "DisasterShield-X predicted flood", "UNOSAT observed flood extent"],
        "info_panel_sources": {
            "iou_f1": {"file": THRESH_CSV, "row": "U-Net v2 @val-tuned",
                       "test_IoU": iou, "test_F1": f1},
            "baselines": {"file": BASE_CSV,
                          "A_test_IoU": float(A["test_IoU"]),
                          "B_test_IoU": float(B["test_IoU"]),
                          "C_test_IoU": float(C["test_IoU"])}},
        "html": out_html, "html_mb": size_mb,
        "geotiff": tif_path,
        "build_seconds": time.time() - t_start,
    }
    json.dump(manifest, open(os.path.join(BUILD_DIR, "build_manifest.json"), "w"), indent=2)
    print("BUILD_DIR:", BUILD_DIR)
    print("DONE in %.0fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
