"""AUDIT v2 flood raster + grid comparison against the shared reference grid."""
import os, json
import numpy as np
import rasterio
from rasterio.windows import Window

DATA = r"d:\Hamim\DisasterShield\data\Feni_2024_10m"
V2 = os.path.join(DATA, "Feni_S1_Flood_18to26Aug2024_10m.tif")
REF = os.path.join(DATA, "Feni_S1_Flood_Aug2024_10m.tif")  # shared reference grid


def valid_fraction(path, blockrows=512):
    with rasterio.open(path) as ds:
        H, W = ds.height, ds.width
        nodata = ds.nodata
        total = valid = 0
        for row in range(0, H, blockrows):
            h = min(blockrows, H - row)
            arr = ds.read(window=Window(0, row, W, h)).astype(np.float32)
            finite = np.all(np.isfinite(arr), axis=0)
            nz = np.any(arr != 0, axis=0)
            mask = finite & nz
            if nodata is not None:
                mask &= np.all(arr != np.float32(nodata), axis=0)
            valid += int(mask.sum())
            total += mask.size
            del arr
        return valid / total


out = {}
with rasterio.open(V2) as ds:
    out["v2"] = {
        "path": V2, "width": ds.width, "height": ds.height, "bands": ds.count,
        "dtype": ds.dtypes[0], "crs": str(ds.crs), "descriptions": ds.descriptions,
        "nodata": ds.nodata,
        "bounds": [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top],
        "transform": list(ds.transform)[:6],
        "file_size_MB": round(os.path.getsize(V2) / 1e6, 1),
    }
    v2_shape = (ds.height, ds.width)
    v2_tr = tuple(ds.transform)[:6]
    v2_crs = ds.crs

with rasterio.open(REF) as ds:
    out["ref"] = {
        "width": ds.width, "height": ds.height, "crs": str(ds.crs),
        "bounds": [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top],
        "transform": list(ds.transform)[:6],
    }
    ref_shape = (ds.height, ds.width)
    ref_tr = tuple(ds.transform)[:6]
    ref_crs = ds.crs

# grid comparison
dims_match = (v2_shape == ref_shape)
crs_match = (str(v2_crs) == str(ref_crs))
tr_diff = [abs(a - b) for a, b in zip(v2_tr, ref_tr)]
tr_match = all(d < 1e-9 for d in tr_diff)
out["grid_comparison"] = {
    "dims_match": dims_match, "crs_match": crs_match,
    "transform_max_abs_diff": max(tr_diff), "transform_match_1e-9": tr_match,
    "aligned": bool(dims_match and crs_match and tr_match),
}
out["v2_valid_fraction"] = round(valid_fraction(V2), 5)
print(json.dumps(out, indent=2))
