#!/usr/bin/env python3
"""
s00b_download_nfhl.py - Download FEMA NFHL Flood Zones from REST API
====================================================================

Downloads flood hazard zones (S_FLD_HAZ_AR) for Dodge County, Nebraska
directly from FEMA's public ArcGIS REST API.

No manual download required!

Output: data_raw/nfhl/dodge_county_flood_zones.gpkg
"""

import requests
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape, box
import time

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
NFHL_DIR = DATA_RAW / "nfhl"

# FEMA NFHL REST API
NFHL_API_BASE = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
FLOOD_ZONES_LAYER = 28  # S_FLD_HAZ_AR

# Actual Dodge County bounds (tighter)
DODGE_BOUNDS = {
    "xmin": -96.92,
    "ymin": 41.52,
    "xmax": -96.32,
    "ymax": 41.92
}

# Key fields to retrieve (reduces payload size)
OUT_FIELDS = "OBJECTID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH,VELOCITY,FLD_AR_ID"


def query_flood_zones_paginated(bounds: dict, batch_size: int = 1000) -> list:
    """Query FEMA API for flood zones within bounds using pagination."""

    url = f"{NFHL_API_BASE}/{FLOOD_ZONES_LAYER}/query"

    # Build geometry envelope
    geometry = json.dumps({
        "xmin": bounds["xmin"],
        "ymin": bounds["ymin"],
        "xmax": bounds["xmax"],
        "ymax": bounds["ymax"],
        "spatialReference": {"wkid": 4326}
    })

    all_features = []
    offset = 0

    print(f"Querying FEMA NFHL API (Layer {FLOOD_ZONES_LAYER})...")
    print(f"  Bounds: {bounds}")

    while True:
        params = {
            "geometry": geometry,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": OUT_FIELDS,
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": batch_size,
            "resultOffset": offset
        }

        print(f"  Fetching records {offset} to {offset + batch_size}...", end=" ")

        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"\n  HTTP Error: {e}")
            if offset == 0:
                # Try with smaller batch
                print("  Retrying with smaller batch size...")
                return query_flood_zones_paginated(bounds, batch_size=500)
            break

        data = response.json()

        if "features" not in data or len(data["features"]) == 0:
            print("no more records.")
            break

        n_features = len(data["features"])
        print(f"got {n_features} features.")
        all_features.extend(data["features"])

        if n_features < batch_size:
            # Last page
            break

        offset += batch_size
        time.sleep(0.5)  # Be nice to the API

    print(f"  Total features retrieved: {len(all_features)}")
    return all_features


def check_if_more_records(bounds: dict) -> int:
    """Check how many total records exist in bounds."""

    url = f"{NFHL_API_BASE}/{FLOOD_ZONES_LAYER}/query"

    geometry = json.dumps({
        "xmin": bounds["xmin"],
        "ymin": bounds["ymin"],
        "xmax": bounds["xmax"],
        "ymax": bounds["ymax"],
        "spatialReference": {"wkid": 4326}
    })

    params = {
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f": "json"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data.get("count", 0)


def features_to_gdf(features: list) -> gpd.GeoDataFrame:
    """Convert GeoJSON features to GeoDataFrame."""

    if not features:
        return gpd.GeoDataFrame()

    # Build GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
    return gdf


def download_flood_zones() -> gpd.GeoDataFrame:
    """Download all flood zones for Dodge County area."""

    # Check record count first
    total_count = check_if_more_records(DODGE_BOUNDS)
    print(f"\nTotal flood zone features in area: {total_count}")

    if total_count == 0:
        print("No flood zones found in specified area.")
        return gpd.GeoDataFrame()

    # Query features with pagination
    features = query_flood_zones_paginated(DODGE_BOUNDS, batch_size=1000)

    if len(features) < total_count:
        print(f"\n  Note: Retrieved {len(features)} of {total_count} features.")

    # Convert to GeoDataFrame
    gdf = features_to_gdf(features)

    if len(gdf) > 0:
        print(f"\nFlood zone breakdown:")
        if "FLD_ZONE" in gdf.columns:
            print(gdf["FLD_ZONE"].value_counts().to_string())

    return gdf


def main():
    """Download and save NFHL flood zones."""

    print("=" * 60)
    print("NFHL Flood Zone Download - Dodge County, Nebraska")
    print("=" * 60)

    # Create output directory
    NFHL_DIR.mkdir(parents=True, exist_ok=True)

    # Download flood zones
    gdf = download_flood_zones()

    if len(gdf) == 0:
        print("\nNo data downloaded. Check FEMA API availability.")
        return

    # Save to GeoPackage
    output_path = NFHL_DIR / "dodge_county_flood_zones.gpkg"
    gdf.to_file(output_path, driver="GPKG")
    print(f"\nSaved: {output_path}")
    print(f"  Features: {len(gdf)}")
    print(f"  Columns: {list(gdf.columns)}")

    # Also save field summary
    print("\n" + "=" * 60)
    print("Key fields for analysis:")
    print("=" * 60)
    for col in ["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "STATIC_BFE"]:
        if col in gdf.columns:
            print(f"\n{col}:")
            print(gdf[col].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
