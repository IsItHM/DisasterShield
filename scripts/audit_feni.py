"""STEP 1 - Feni 10m data audit (read-only). RAM-safe: windowed reads, one raster at a time."""
import os, json
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box

DATA = r"d:\Hamim\DisasterShield\data\Feni_2024_10m"
LABELS = r"d:\Hamim\DisasterShield\data\labels_unosat"
RASTERS = [
    "Feni_S2_Flood_Aug2024_10m.tif",
    "Feni_S2_PreFlood_May2024_10m.tif",
    "Feni_S1_Flood_Aug2024_10m.tif",
    "Feni_S1_PreFlood_May2024_10m.tif",
]

def pixel_size_m(ds):
    # transform gives units of CRS. If projected (m) use directly; if geographic (deg) approx.
    a = ds.transform
    px, py = abs(a.a), abs(a.e)
    if ds.crs and ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2.0
        mx = px * 111320.0 * np.cos(np.radians(lat))
        my = py * 110540.0
        return mx, my, "approx-from-degrees"
    return px, py, "crs-units(m)"

def valid_fraction(path, blockrows=512):
    """Fraction of pixels valid across all bands: finite, nonzero, not nodata.
    A pixel counts valid if ALL bands are finite and not equal to nodata; 'nonzero' = not all-band-zero."""
    with rasterio.open(path) as ds:
        H, W, B = ds.height, ds.width, ds.count
        nodata = ds.nodata
        total = 0
        valid = 0
        for row in range(0, H, blockrows):
            h = min(blockrows, H - row)
            win = Window(0, row, W, h)
            arr = ds.read(window=win).astype(np.float32)  # (B,h,W)
            finite = np.all(np.isfinite(arr), axis=0)
            nz = np.any(arr != 0, axis=0)
            mask = finite & nz
            if nodata is not None:
                notnd = np.all(arr != np.float32(nodata), axis=0)
                mask &= notnd
            valid += int(mask.sum())
            total += mask.size
            del arr
        return valid / total, total

def main():
    out = {"rasters": {}, "labels": {}}
    for name in RASTERS:
        path = os.path.join(DATA, name)
        with rasterio.open(path) as ds:
            mx, my, note = pixel_size_m(ds)
            out["rasters"][name] = {
                "width": ds.width, "height": ds.height, "bands": ds.count,
                "dtype": ds.dtypes[0], "crs": str(ds.crs),
                "pixel_size_m": [round(mx, 3), round(my, 3)], "pixel_note": note,
                "nodata": ds.nodata,
                "bounds": [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top],
                "file_size_MB": round(os.path.getsize(path) / 1e6, 1),
            }
    # valid fractions
    for key, name in [("S2_flood", "Feni_S2_Flood_Aug2024_10m.tif"),
                      ("S1_flood", "Feni_S1_Flood_Aug2024_10m.tif")]:
        vf, tot = valid_fraction(os.path.join(DATA, name))
        out["rasters"][name]["valid_fraction"] = round(vf, 5)

    # labels
    ref_name = "Feni_S1_Flood_Aug2024_10m.tif"
    with rasterio.open(os.path.join(DATA, ref_name)) as ds:
        rb = ds.bounds
        rcrs = ds.crs
        raster_box_wgs = None
        rbox = box(rb.left, rb.bottom, rb.right, rb.top)

    for shp in ["S1_20240818_20240826_FloodExtent_Bangladesh.shp",
                "S1_20240828_20240904_FloodExtent_Bangladesh.shp",
                "AnalysisExtent_Bangladesh.shp"]:
        gdf = gpd.read_file(os.path.join(LABELS, shp))
        # count geometry parts (features may be MultiPolygons)
        parts = 0
        for geom in gdf.geometry:
            if geom is None:
                continue
            parts += len(geom.geoms) if geom.geom_type.startswith("Multi") else 1
        info = {"crs": str(gdf.crs), "n_features": len(gdf), "n_polygon_parts": parts}
        # reproject to raster CRS for overlap (bbox), and to UTM46N for area
        gdf_r = gdf.to_crs(rcrs)
        area_km2 = float(gdf.to_crs("EPSG:32646").geometry.area.sum()) / 1e6
        info["total_area_km2"] = round(area_km2, 3)
        lb = gdf_r.total_bounds  # minx miny maxx maxy
        info["bounds_in_rasterCRS"] = [float(x) for x in lb]
        lbox = box(*lb)
        inter = rbox.intersection(lbox).area
        info["overlap_pct_of_raster_bbox"] = round(100 * inter / rbox.area, 2)
        info["overlap_pct_of_label_bbox"] = round(100 * inter / lbox.area, 2) if lbox.area > 0 else None
        out["labels"][shp] = info

    out["reference_raster_for_labels"] = ref_name
    out["reference_raster_bounds"] = [rb.left, rb.bottom, rb.right, rb.top]
    out["reference_raster_crs"] = str(rcrs)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
