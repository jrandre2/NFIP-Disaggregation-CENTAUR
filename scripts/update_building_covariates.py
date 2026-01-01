#!/usr/bin/env python3
"""
Update building covariates (slope, distance to SFHA) for prepared counties.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import geopandas as gpd
import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_WORK = PROJECT_ROOT / "data_work"

ELEVATION_DIR = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Elevation/USGS_3DEP_10m")

COUNTY_NFHL = {
    "dodge": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
    "douglas": Path("/Users/jesseandrews/Downloads/31055C_20250325/S_FLD_HAZ_AR.shp"),
    "cass": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
    "dakota": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
}


def compute_elevation_and_slope(points_ll: gpd.GeoSeries) -> tuple[list, list]:
    try:
        import rasterio
    except ImportError as exc:
        print("rasterio not available, skipping elevation/slope")
        n = len(points_ll)
        return [None] * n, [None] * n

    xs = np.array([p.x for p in points_ll])
    ys = np.array([p.y for p in points_ll])
    dem_files = list(ELEVATION_DIR.glob("*.tif"))
    elevations = [None] * len(points_ll)
    slopes = [None] * len(points_ll)

    for dem_file in dem_files:
        with rasterio.open(dem_file) as src:
            bounds = src.bounds
            arr = src.read(1, masked=True).astype("float64")
            if hasattr(arr, "filled"):
                arr = arr.filled(float("nan"))
            lat_center = (bounds.top + bounds.bottom) / 2
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))
            xres_m = src.res[0] * meters_per_deg_lon
            yres_m = src.res[1] * meters_per_deg_lat
            dy, dx = np.gradient(arr, yres_m, xres_m)
            slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

            in_tile = (
                (xs >= bounds.left) & (xs <= bounds.right) &
                (ys >= bounds.bottom) & (ys <= bounds.top)
            )
            idxs = np.where(in_tile)[0]
            if idxs.size == 0:
                continue
            from rasterio.transform import rowcol
            rows, cols = rowcol(src.transform, xs[idxs], ys[idxs])
            rows = np.array(rows)
            cols = np.array(cols)
            valid = (
                (rows >= 0) & (rows < arr.shape[0]) &
                (cols >= 0) & (cols < arr.shape[1])
            )
            idxs = idxs[valid]
            rows = rows[valid]
            cols = cols[valid]
            elev_vals = arr[rows, cols]
            slope_vals = slope[rows, cols]
            for i, elev, s in zip(idxs, elev_vals, slope_vals):
                if elevations[i] is None and not np.isnan(elev):
                    elevations[i] = float(elev)
                if slopes[i] is None and not np.isnan(s):
                    slopes[i] = float(s)

    return elevations, slopes


def compute_distance_to_sfha(points_proj: gpd.GeoSeries, nfhl_path: Path) -> list:
    if not nfhl_path.exists():
        return [None] * len(points_proj)

    bounds = tuple(points_proj.to_crs("EPSG:4326").total_bounds)
    nfhl = gpd.read_file(nfhl_path, bbox=bounds)
    if nfhl.empty:
        return [None] * len(points_proj)

    sfha = nfhl[nfhl["SFHA_TF"] == "T"].copy()
    if sfha.empty:
        return [None] * len(points_proj)

    target_crs = points_proj.crs
    sfha_proj = sfha.to_crs(target_crs)
    sfha_union = sfha_proj.geometry.union_all()
    return points_proj.distance(sfha_union).astype(float).tolist()


def update_county(
    county_key: str,
    base_dir: Path,
    skip_slope: bool,
    skip_distance: bool,
) -> None:
    in_path = base_dir / county_key / "buildings_prepared.gpkg"
    if not in_path.exists():
        print(f"Missing buildings file: {in_path}")
        return

    gdf = gpd.read_file(in_path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    # Centroids in lon/lat for DEM sampling
    geom_ll = gdf.geometry.to_crs("EPSG:26914")
    if not skip_slope:
        centroids_ll = geom_ll.centroid.to_crs("EPSG:4326")
        elev, slope = compute_elevation_and_slope(centroids_ll)
        gdf["ELEVATION"] = elev
        gdf["SlopeDeg"] = slope

    # Distance to SFHA in projected CRS for meters
    if not skip_distance:
        points_proj = geom_ll.centroid
        dist = compute_distance_to_sfha(points_proj, COUNTY_NFHL[county_key])
        gdf["DistToSFHA_m"] = dist

    gdf.to_file(in_path, driver="GPKG")
    gdf.drop(columns=["geometry"], errors="ignore").to_csv(
        base_dir / county_key / "buildings_prepared.csv", index=False
    )
    print(f"Updated covariates for {county_key}: {in_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update building covariates for prepared counties.")
    parser.add_argument("--counties", default="dodge,douglas",
                        help="Comma-separated county keys to update.")
    parser.add_argument("--skip-slope", action="store_true",
                        help="Skip slope/elevation update.")
    parser.add_argument("--skip-distance", action="store_true",
                        help="Skip distance-to-SFHA update.")
    parser.add_argument("--base-dir", default=str(DATA_WORK / "revision"),
                        help="Base directory with county subfolders.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    for key in [c.strip() for c in args.counties.split(",") if c.strip()]:
        if key not in COUNTY_NFHL:
            print(f"Skipping unknown county: {key}")
            continue
        update_county(key, base_dir, args.skip_slope, args.skip_distance)


if __name__ == "__main__":
    main()
