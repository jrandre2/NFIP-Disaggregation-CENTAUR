#!/usr/bin/env python3
"""
s00b_download_nfhl.py - Download FEMA NFHL Flood Zones from REST API
====================================================================

Downloads flood hazard zones (S_FLD_HAZ_AR) for a Nebraska county
(default Dodge; use --county douglas) directly from FEMA's public ArcGIS
REST API. No manual download required.

Output: data_raw/nfhl/<county>_county_flood_zones.gpkg
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

# County FIPS (within Nebraska, state 31) for deriving query bounds from the
# Census block-group shapefile in data_raw/cbg.
COUNTY_FIPS = {"dodge": "053", "douglas": "055"}
CBG_SHP = DATA_RAW / "cbg" / "tl_2020_31_bg.shp"

# Generous fallback bounds if the CBG shapefile is unavailable.
FALLBACK_BOUNDS = {
    "dodge": DODGE_BOUNDS,
    "douglas": {"xmin": -96.48, "ymin": 41.15, "xmax": -95.85, "ymax": 41.45},
}


def bounds_for_county(county: str, pad: float = 0.03) -> dict:
    """County query envelope, derived from the CBG shapefile when available."""
    fips3 = COUNTY_FIPS.get(county)
    if fips3 and CBG_SHP.exists():
        cbg = gpd.read_file(CBG_SHP)
        sub = cbg[cbg["COUNTYFP"] == fips3]
        if len(sub):
            if sub.crs is not None and sub.crs.to_string() != "EPSG:4326":
                sub = sub.to_crs("EPSG:4326")
            minx, miny, maxx, maxy = sub.total_bounds
            return {"xmin": minx - pad, "ymin": miny - pad,
                    "xmax": maxx + pad, "ymax": maxy + pad}
    if county in FALLBACK_BOUNDS:
        return FALLBACK_BOUNDS[county]
    raise SystemExit(f"Unknown county '{county}' and no CBG bounds available.")

# Key fields to retrieve (reduces payload size)
OUT_FIELDS = "OBJECTID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH,VELOCITY,FLD_AR_ID"


def query_flood_zones_paginated(bounds: dict, batch_size: int = 200) -> list:
    """Query FEMA NFHL flood zones within bounds via OBJECTID batching."""

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
    seen_ids = set()

    print(f"Querying FEMA NFHL API (Layer {FLOOD_ZONES_LAYER})...")
    print(f"  Bounds: {bounds}")

    def _request_json(params: dict) -> dict:
        """GET a query and return parsed JSON, retrying transient errors."""
        for attempt in range(4):
            try:
                response = requests.get(url, params=params, timeout=180)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == 3:
                    raise
                print(f"(retry {attempt + 1} after {e})", end=" ")
                time.sleep(2 * (attempt + 1))
        return {}

    def _envelope(b: dict) -> str:
        return json.dumps({"xmin": b["xmin"], "ymin": b["ymin"],
                           "xmax": b["xmax"], "ymax": b["ymax"],
                           "spatialReference": {"wkid": 4326}})

    def _count(b: dict) -> int:
        d = _request_json({"geometry": _envelope(b),
                           "geometryType": "esriGeometryEnvelope",
                           "spatialRel": "esriSpatialRelIntersects",
                           "returnCountOnly": "true", "f": "json"})
        return int(d.get("count", 0))

    # FEMA's server returns HTTP 500 for resultOffset paging (and for long
    # objectIds lists) when geometry is requested; only a single offset-0 query
    # is reliable. So recursively subdivide the envelope into tiles small enough
    # (<= SAFE features) to retrieve in one offset-0 request, then merge and
    # de-duplicate by OBJECTID. This is the standard bulk-extract pattern for
    # ArcGIS servers whose pagination is broken for geometry queries.
    SAFE = 450

    def _subdivide(b: dict, depth: int) -> None:
        mx = (b["xmin"] + b["xmax"]) / 2.0
        my = (b["ymin"] + b["ymax"]) / 2.0
        _fetch({"xmin": b["xmin"], "ymin": b["ymin"], "xmax": mx, "ymax": my}, depth + 1)
        _fetch({"xmin": mx, "ymin": b["ymin"], "xmax": b["xmax"], "ymax": my}, depth + 1)
        _fetch({"xmin": b["xmin"], "ymin": my, "xmax": mx, "ymax": b["ymax"]}, depth + 1)
        _fetch({"xmin": mx, "ymin": my, "xmax": b["xmax"], "ymax": b["ymax"]}, depth + 1)

    def _fetch(b: dict, depth: int = 0) -> None:
        try:
            n = _count(b)
        except requests.exceptions.RequestException:
            n = SAFE + 1  # if even the count fails, force subdivision
        if n == 0:
            return
        if n <= SAFE:
            try:
                data = _request_json({"geometry": _envelope(b),
                                      "geometryType": "esriGeometryEnvelope",
                                      "spatialRel": "esriSpatialRelIntersects",
                                      "outFields": OUT_FIELDS,
                                      "returnGeometry": "true",
                                      "f": "geojson",
                                      "resultRecordCount": 500})
            except requests.exceptions.RequestException as e:
                # A few small tiles 500 on the geometry fetch (complex polygons
                # the server fails to serialize). Subdivide to isolate them.
                if depth < 14:
                    print(f"  fetch error on tile [{n} feat]; subdividing")
                    _subdivide(b, depth)
                else:
                    print(f"  WARNING: dropping tile [{n} feat] at max depth: {e}")
                return
            new = 0
            for feat in data.get("features", []):
                oid = (feat.get("properties") or {}).get("OBJECTID")
                if oid is not None and oid in seen_ids:
                    continue
                if oid is not None:
                    seen_ids.add(oid)
                all_features.append(feat)
                new += 1
            print(f"  tile [{n} feat] -> +{new} (running total {len(all_features)})")
            time.sleep(0.2)
            return
        # Too many features to fetch in one offset-0 request: subdivide.
        if depth < 14:
            _subdivide(b, depth)
        else:
            print(f"  WARNING: dropping dense tile [{n} feat] at max depth")

    _fetch(bounds)
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


def download_flood_zones(bounds: dict) -> gpd.GeoDataFrame:
    """Download all flood zones within the given bounds."""

    # Check record count first
    total_count = check_if_more_records(bounds)
    print(f"\nTotal flood zone features in area: {total_count}")

    if total_count == 0:
        print("No flood zones found in specified area.")
        return gpd.GeoDataFrame()

    # Query features with pagination. With returnGeometry=true this server
    # errors on large page sizes, so use a conservative batch that succeeds.
    features = query_flood_zones_paginated(bounds, batch_size=500)

    if len(features) < total_count:
        print(f"\n  Note: Retrieved {len(features)} of {total_count} features.")

    # Convert to GeoDataFrame
    gdf = features_to_gdf(features)

    # Defensive de-duplication (resultOffset paging should not overlap).
    if len(gdf) and "OBJECTID" in gdf.columns:
        gdf = gdf.drop_duplicates(subset="OBJECTID").reset_index(drop=True)

    if len(gdf) > 0:
        print(f"\nFlood zone breakdown:")
        if "FLD_ZONE" in gdf.columns:
            print(gdf["FLD_ZONE"].value_counts().to_string())

    return gdf


def main():
    """Download and save NFHL flood zones for a county."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FEMA NFHL flood zones.")
    parser.add_argument("--county", default="dodge",
                        help="County key (dodge, douglas, ...). Default: dodge.")
    args = parser.parse_args()
    county = args.county.lower()

    print("=" * 60)
    print(f"NFHL Flood Zone Download - {county.title()} County, Nebraska")
    print("=" * 60)

    # Create output directory
    NFHL_DIR.mkdir(parents=True, exist_ok=True)

    # Download flood zones within the county envelope
    bounds = bounds_for_county(county)
    gdf = download_flood_zones(bounds)

    if len(gdf) == 0:
        print("\nNo data downloaded. Check FEMA API availability.")
        return

    # Save to GeoPackage (county-specific, portable under data_raw/nfhl/)
    output_path = NFHL_DIR / f"{county}_county_flood_zones.gpkg"
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
