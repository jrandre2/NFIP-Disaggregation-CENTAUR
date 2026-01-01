#!/usr/bin/env python3
"""
Prepare county-specific datasets for revision analyses (claims, policies, buildings).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_WORK = PROJECT_ROOT / "data_work"

CLAIMS_CSV = PROJECT_ROOT / "scripts/NEFloodMitigation/Data/NE_FEMA_Claims.csv"
POLICIES_CSV = DATA_RAW / "nfip_policies_subset.csv"

BUILDINGS_GEOJSON = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Building_Footprints/Nebraska.geojson")
PARCELS_GDB = Path("/Volumes/T9/Projects/Freeze and Flight/statewide parcel/NE_2023_statewideparcels.gdb")
ELEVATION_DIR = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Elevation/USGS_3DEP_10m")
INUNDATION_GPKG = Path("/Volumes/T9/Projects/Freeze and Flight/data_work/inund_union.gpkg")

ZCTA_SHP = Path("/Volumes/T9/Projects/ML Vision Broadband/tl_2022_us_zcta520.shp")
CBG_DIR = DATA_RAW / "cbg"
CBG_SHP = CBG_DIR / "tl_2020_31_bg.shp"


COUNTIES = {
    "dodge": {
        "fips": 31053,
        "county_id": "053",
        "bounds": (-97.0, 41.4, -96.3, 42.0),
        "nfhl": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
        "boundary": None,
    },
    "douglas": {
        "fips": 31055,
        "county_id": "055",
        "bounds": None,
        "nfhl": Path("/Users/jesseandrews/Downloads/31055C_20250325/S_FLD_HAZ_AR.shp"),
        "boundary": Path("/Volumes/T9/Projects/Freeze and Flight/data_work/douglas_county.gpkg"),
    },
    "cass": {
        "fips": 31025,
        "county_id": "025",
        "bounds": None,
        "nfhl": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
        "boundary": None,
    },
    "dakota": {
        "fips": 31043,
        "county_id": "043",
        "bounds": None,
        "nfhl": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
        "boundary": None,
    },
}


def ensure_cbg_shp() -> Path:
    if CBG_SHP.exists():
        return CBG_SHP

    CBG_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_31_bg.zip"
    zip_path = CBG_DIR / "tl_2020_31_bg.zip"
    print(f"Downloading NE block groups to {zip_path}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CBG_DIR)
    print(f"Extracted CBG shapefile to {CBG_DIR}")
    return CBG_SHP


def load_county_boundary(county: Dict) -> gpd.GeoDataFrame:
    if county["boundary"]:
        gdf = gpd.read_file(county["boundary"])
        return gdf

    cbg = gpd.read_file(ensure_cbg_shp())
    county_fips = str(county["fips"])[-3:]
    boundary = cbg[cbg["COUNTYFP"] == county_fips].dissolve(by="COUNTYFP")
    return boundary


def prepare_claims(
    county: Dict,
    out_dir: Path,
    event_year: int,
    event_month: int | None,
    event_label: str | None,
) -> Path:
    df = pd.read_csv(CLAIMS_CSV)
    df["dateOfLoss"] = pd.to_datetime(df["dateOfLoss"])

    mask = (
        (df["countyCode"] == county["fips"]) &
        (df["occupancyType"] == 1)
    )
    mask &= df["dateOfLoss"].dt.year == event_year
    if event_month is not None:
        mask &= df["dateOfLoss"].dt.month == event_month
    if event_label:
        mask &= df["floodEvent"].astype(str).str.contains(event_label, na=False)
    claims = df[mask].copy()
    claims = claims.rename(columns={
        "reportedZipCode": "ZIP",
        "floodZoneCurrent": "FloodZone",
        "baseFloodElevation": "BFE",
        "buildingReplacementCost": "COST",
        "originalConstructionDate": "OrigConstDate",
        "censusBlockGroupFips": "CBG",
    })

    claims["ZIP"] = claims["ZIP"].apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else None)
    claims["FloodZone"] = claims["FloodZone"].str.upper()
    claims["OrigConstDate"] = pd.to_datetime(claims["OrigConstDate"], errors="coerce")
    claims["OrigYear"] = claims["OrigConstDate"].dt.year
    claims["ZCTA"] = claims["ZIP"]
    claims["CBG"] = claims["CBG"].apply(lambda x: str(int(x)).zfill(12) if pd.notna(x) else None)
    claims["LAT_RND"] = claims["latitude"].round(1)
    claims["LON_RND"] = claims["longitude"].round(1)

    for col in ["netBuildingPaymentAmount", "netContentsPaymentAmount", "netIccPaymentAmount",
                "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
                "amountPaidOnIncreasedCostOfComplianceClaim"]:
        if col not in claims.columns:
            claims[col] = 0.0

    net_total = (
        claims["netBuildingPaymentAmount"].fillna(0.0) +
        claims["netContentsPaymentAmount"].fillna(0.0) +
        claims["netIccPaymentAmount"].fillna(0.0)
    )
    gross_total = (
        claims["amountPaidOnBuildingClaim"].fillna(0.0) +
        claims["amountPaidOnContentsClaim"].fillna(0.0) +
        claims["amountPaidOnIncreasedCostOfComplianceClaim"].fillna(0.0)
    )
    claims["TotalPayment"] = net_total.where(net_total > 0, gross_total)

    claims["ClaimID"] = range(1, len(claims) + 1)

    out_path = out_dir / "claims_prepared.csv"
    claims.to_csv(out_path, index=False)
    print(f"Saved claims: {out_path} ({len(claims)})")
    return out_path


def prepare_policies(
    county: Dict,
    out_dir: Path,
    event_year: int,
    event_month: int | None,
) -> Path | None:
    if not POLICIES_CSV.exists():
        print(f"Policies CSV not found: {POLICIES_CSV}")
        return None

    df = pd.read_csv(POLICIES_CSV)
    df["policyEffectiveDate"] = pd.to_datetime(df["policyEffectiveDate"], errors="coerce")
    df["policyTerminationDate"] = pd.to_datetime(df["policyTerminationDate"], errors="coerce")

    if event_month is None:
        start = pd.Timestamp(f"{event_year}-01-01T00:00:00Z")
        end = pd.Timestamp(f"{event_year}-12-31T23:59:59Z")
    else:
        start = pd.Timestamp(f"{event_year}-{event_month:02d}-01T00:00:00Z")
        end = (start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)

    mask = (
        (df["countyCode"].astype(str) == str(county["fips"])) &
        (df["occupancyType"] == 1) &
        (df["policyEffectiveDate"] <= end) &
        ((df["policyTerminationDate"].isna()) | (df["policyTerminationDate"] >= start))
    )
    policies = df[mask].copy()
    policies = policies.rename(columns={
        "reportedZipCode": "ZIP",
        "floodZoneCurrent": "FloodZone",
        "baseFloodElevation": "BFE",
        "buildingReplacementCost": "COST",
        "originalConstructionDate": "OrigConstDate",
        "censusBlockGroupFips": "CBG",
    })

    policies["ZIP"] = policies["ZIP"].apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else None)
    policies["FloodZone"] = policies["FloodZone"].str.upper()
    policies["OrigConstDate"] = pd.to_datetime(policies["OrigConstDate"], errors="coerce")
    policies["OrigYear"] = policies["OrigConstDate"].dt.year
    policies["ZCTA"] = policies["ZIP"]
    policies["CBG"] = policies["CBG"].apply(lambda x: str(int(x)).zfill(12) if pd.notna(x) else None)
    policies["LAT_RND"] = policies["latitude"].round(1)
    policies["LON_RND"] = policies["longitude"].round(1)
    policies["PolicyID"] = range(1, len(policies) + 1)

    out_path = out_dir / "policies_prepared.csv"
    policies.to_csv(out_path, index=False)
    print(f"Saved policies: {out_path} ({len(policies)})")
    return out_path


def extract_elevation_and_slope(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        import rasterio
        import math
    except ImportError as exc:
        print("rasterio not available, skipping elevation")
        buildings["ELEVATION"] = None
        buildings["SlopeDeg"] = None
        return buildings

    centroids = buildings["centroid"].to_crs("EPSG:4326")
    coords = [(p.x, p.y) for p in centroids]
    dem_files = list(ELEVATION_DIR.glob("*.tif"))
    elevations = [None] * len(buildings)
    slopes = [None] * len(buildings)

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

            for i, (x, y) in enumerate(coords):
                if elevations[i] is None and bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
                    row, col = src.index(x, y)
                    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                        elev = arr[row, col]
                        if not np.isnan(elev):
                            elevations[i] = float(elev)
                        slope_val = slope[row, col]
                        if not np.isnan(slope_val):
                            slopes[i] = float(slope_val)

    buildings["ELEVATION"] = elevations
    buildings["SlopeDeg"] = slopes
    return buildings


def compute_distance_to_sfha(buildings: gpd.GeoDataFrame, nfhl: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sfha = nfhl[nfhl["SFHA_TF"] == "T"].copy()
    if sfha.empty:
        buildings["DistToSFHA_m"] = None
        return buildings

    target_crs = "EPSG:26914"
    sfha_proj = sfha.to_crs(target_crs)
    points_proj = buildings["centroid"].to_crs(target_crs)
    sfha_union = sfha_proj.geometry.union_all()
    buildings["DistToSFHA_m"] = points_proj.distance(sfha_union).astype(float)
    return buildings


def prepare_buildings(county: Dict, out_dir: Path) -> Tuple[Path, Path]:
    if county["bounds"]:
        bounds = county["bounds"]
    else:
        boundary = load_county_boundary(county)
        if boundary.crs is None:
            boundary.set_crs("EPSG:4326", inplace=True)
        if boundary.crs.to_string() != "EPSG:4326":
            boundary = boundary.to_crs("EPSG:4326")
        bounds = tuple(boundary.total_bounds)

    print(f"Loading buildings for bounds: {bounds}")
    buildings = gpd.read_file(BUILDINGS_GEOJSON, bbox=bounds)
    buildings["BldgID"] = range(1, len(buildings) + 1)
    # Compute centroids in projected CRS for accuracy
    if buildings.crs is None:
        buildings.set_crs("EPSG:4326", inplace=True)
    buildings_proj = buildings.to_crs("EPSG:26914")
    centroids_proj = buildings_proj.geometry.centroid
    buildings["centroid"] = gpd.GeoSeries(centroids_proj, crs="EPSG:26914").to_crs(buildings.crs)

    parcels = gpd.read_file(
        PARCELS_GDB,
        layer="StatewideParcels_Current",
        where=f"County_ID = '{county['county_id']}'"
    )
    if buildings.crs != parcels.crs:
        parcels = parcels.to_crs(buildings.crs)

    centroids = gpd.GeoDataFrame(
        buildings[["BldgID"]],
        geometry=buildings["centroid"],
        crs=buildings.crs
    )
    joined = gpd.sjoin(centroids, parcels, how="left", predicate="within")
    parcel_attrs = joined[[
        "BldgID", "Parcel_ID", "Ph_Zip5",
        "Total_Assessed_Value", "Classification_Code", "BuildingYear"
    ]].drop_duplicates(subset="BldgID")
    buildings = buildings.merge(parcel_attrs, on="BldgID", how="left")
    buildings = buildings.rename(columns={
        "Ph_Zip5": "ZIP",
        "Total_Assessed_Value": "Total_Asse",
        "Classification_Code": "ClassCode",
        "BuildingYear": "BuildYear",
    })

    buildings = extract_elevation_and_slope(buildings)

    nfhl = gpd.read_file(county["nfhl"], bbox=bounds)
    if buildings.crs != nfhl.crs:
        nfhl = nfhl.to_crs(buildings.crs)

    fz_join = gpd.sjoin(
        centroids,
        nfhl[["FLD_ZONE", "SFHA_TF", "geometry"]],
        how="left",
        predicate="within"
    )
    fz_attrs = fz_join[["BldgID", "FLD_ZONE", "SFHA_TF"]].drop_duplicates(subset="BldgID")
    buildings = buildings.merge(fz_attrs, on="BldgID", how="left")
    buildings = buildings.rename(columns={"FLD_ZONE": "FloodZone", "SFHA_TF": "SFHA"})
    buildings["FloodZone"] = buildings["FloodZone"].str.upper()
    buildings = compute_distance_to_sfha(buildings, nfhl)

    # ZCTA + CBG joins (use EPSG:4326 centroids)
    centroids_ll = buildings["centroid"].to_crs("EPSG:4326")
    points_ll = gpd.GeoDataFrame(buildings[["BldgID"]], geometry=centroids_ll, crs="EPSG:4326")

    if ZCTA_SHP.exists():
        zcta = gpd.read_file(ZCTA_SHP, bbox=bounds)
        if zcta.crs is None:
            zcta.set_crs("EPSG:4269", inplace=True)
        if zcta.crs.to_string() != "EPSG:4326":
            zcta = zcta.to_crs("EPSG:4326")
        zcta_join = gpd.sjoin(points_ll, zcta[["ZCTA5CE20", "geometry"]],
                              how="left", predicate="within")
        buildings = buildings.merge(
            zcta_join[["BldgID", "ZCTA5CE20"]].drop_duplicates("BldgID"),
            on="BldgID", how="left"
        )
        buildings = buildings.rename(columns={"ZCTA5CE20": "ZCTA"})

    cbg_path = ensure_cbg_shp()
    cbg = gpd.read_file(cbg_path, bbox=bounds)
    if cbg.crs is None:
        cbg.set_crs("EPSG:4269", inplace=True)
    if cbg.crs.to_string() != "EPSG:4326":
        cbg = cbg.to_crs("EPSG:4326")
    cbg_join = gpd.sjoin(points_ll, cbg[["GEOID", "geometry"]],
                         how="left", predicate="within")
    buildings = buildings.merge(
        cbg_join[["BldgID", "GEOID"]].drop_duplicates("BldgID"),
        on="BldgID", how="left"
    )
    buildings = buildings.rename(columns={"GEOID": "CBG"})

    buildings["LAT"] = points_ll.geometry.y
    buildings["LON"] = points_ll.geometry.x
    buildings["LAT_RND"] = buildings["LAT"].round(1)
    buildings["LON_RND"] = buildings["LON"].round(1)

    # Save all buildings
    all_path = out_dir / "buildings_all.gpkg"
    buildings.drop(columns=["centroid"], errors="ignore").to_file(all_path, driver="GPKG")

    # Largest-per-parcel filter
    with_parcel = buildings[buildings["Parcel_ID"].notna()].copy()
    idx = with_parcel.groupby("Parcel_ID")["Total_Asse"].idxmax()
    largest = with_parcel.loc[idx]
    no_parcel = buildings[buildings["Parcel_ID"].isna()]
    largest = pd.concat([largest, no_parcel], ignore_index=True)

    prepared_path = out_dir / "buildings_prepared.gpkg"
    largest.drop(columns=["centroid"], errors="ignore").to_file(prepared_path, driver="GPKG")
    largest.drop(columns=["geometry", "centroid"], errors="ignore").to_csv(
        out_dir / "buildings_prepared.csv", index=False
    )

    print(f"Saved buildings: {prepared_path} ({len(largest)})")
    return all_path, prepared_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare county datasets for revision analysis.")
    parser.add_argument("--counties", default="dodge,douglas",
                        help="Comma-separated county keys to prepare (default: dodge,douglas).")
    parser.add_argument("--event-year", type=int, default=2019,
                        help="Event year for claim filtering (default: 2019).")
    parser.add_argument("--event-month", type=int, default=3,
                        help="Event month for claim filtering (default: 3). Use 0 to disable.")
    parser.add_argument("--event-label", default=None,
                        help="Optional floodEvent label filter (substring match).")
    parser.add_argument("--out-root", default=str(DATA_WORK / "revision"),
                        help="Output root directory (default: data_work/revision).")
    parser.add_argument("--skip-policies", action="store_true",
                        help="Skip policy preparation (useful for non-2019 events).")
    parser.add_argument("--skip-inundation", action="store_true",
                        help="Skip inundation copy (useful for non-2019 events).")
    args = parser.parse_args()

    event_month = None if args.event_month == 0 else args.event_month
    out_root = Path(args.out_root)

    for key in [c.strip() for c in args.counties.split(",") if c.strip()]:
        county = COUNTIES[key]
        out_dir = out_root / key
        out_dir.mkdir(parents=True, exist_ok=True)

        prepare_claims(county, out_dir, args.event_year, event_month, args.event_label)
        if not args.skip_policies:
            prepare_policies(county, out_dir, args.event_year, event_month)
        prepare_buildings(county, out_dir)

        if not args.skip_inundation and INUNDATION_GPKG.exists():
            inund_out = out_dir / "inundation.gpkg"
            if not inund_out.exists():
                gpd.read_file(INUNDATION_GPKG).to_file(inund_out, driver="GPKG")
            print(f"Saved inundation: {inund_out}")


if __name__ == "__main__":
    main()
