import os
import geopandas as gpd
from shapely.geometry import box
import rasterio
LAB = r"d:\Hamim\DisasterShield\data\labels_unosat"
with rasterio.open(r"d:\Hamim\DisasterShield\data\Feni_2024_10m\Feni_S1_Flood_Aug2024_10m.tif") as ds:
    b = ds.bounds
rbox = box(b.left, b.bottom, b.right, b.top)
rw = gpd.GeoSeries([rbox], crs="EPSG:4326").to_crs("EPSG:32646").area.iloc[0] / 1e6
for shp in ["S1_20240818_20240826_FloodExtent_Bangladesh.shp",
            "S1_20240828_20240904_FloodExtent_Bangladesh.shp"]:
    g = gpd.read_file(os.path.join(LAB, shp))
    clip = g.clip(rbox)
    a = clip.to_crs("EPSG:32646").geometry.area.sum() / 1e6
    print(shp)
    print("  flood_km2_inside_Feni_window:", round(a, 2),
          "| Feni_window_km2:", round(rw, 1),
          "| flood_pct_of_window:", round(100 * a / rw, 2))
