#!/usr/bin/env python3
"""
s00_prepare_nfip.py - Prepare Data for NFIP Claims Disaggregation
==================================================================

This script prepares all input data for the bootstrap disaggregation analysis:
1. Filter FEMA claims to Dodge County, March 2019, residential
2. Load building footprints for study area
3. Join buildings with parcel data for assessed values
4. Extract elevation for each building
5. Join buildings with NFHL flood zones
6. Prepare inundation layer for validation

Output files saved to data_work/:
- claims_prepared.parquet
- buildings_prepared.parquet
- inundation.gpkg
"""

import os
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Point

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_WORK = PROJECT_ROOT / "data_work"

# External data paths
CLAIMS_CSV = PROJECT_ROOT / "scripts/NEFloodMitigation/Data/NE_FEMA_Claims.csv"
BUILDINGS_GEOJSON = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Building_Footprints/Nebraska.geojson")
PARCELS_GDB = Path("/Volumes/T9/Projects/Freeze and Flight/statewide parcel/NE_2023_statewideparcels.gdb")
ELEVATION_DIR = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Elevation/USGS_3DEP_10m")
INUNDATION_GPKG = Path("/Volumes/T9/Projects/Freeze and Flight/data_work/inund_union.gpkg")
NFHL_SHP = DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp"

# Study area parameters
DODGE_COUNTY_FIPS = 31053
DODGE_COUNTY_ID = "053"
STUDY_YEAR = 2019
STUDY_MONTH = 3

# Dodge County approximate bounds (for filtering buildings)
DODGE_BOUNDS = box(-97.0, 41.4, -96.3, 42.0)


def prepare_claims():
    """Load and filter NFIP claims to study area."""
    print("Loading FEMA claims...")
    df = pd.read_csv(CLAIMS_CSV)
    df['dateOfLoss'] = pd.to_datetime(df['dateOfLoss'])

    # Filter to Dodge County, March 2019, residential
    mask = (
        (df['countyCode'] == DODGE_COUNTY_FIPS) &
        (df['dateOfLoss'].dt.year == STUDY_YEAR) &
        (df['dateOfLoss'].dt.month == STUDY_MONTH) &
        (df['occupancyType'] == 1)  # Residential
    )
    claims = df[mask].copy()

    # Select and rename relevant columns
    claims = claims.rename(columns={
        'reportedZipCode': 'ZIP',
        'floodZoneCurrent': 'FloodZone',
        'baseFloodElevation': 'BFE',
        'buildingReplacementCost': 'COST',
        'originalConstructionDate': 'OrigConstDate'
    })

    # Clean ZIP codes
    claims['ZIP'] = claims['ZIP'].apply(lambda x: str(int(x)).zfill(5) if pd.notna(x) else None)

    # Standardize flood zones
    claims['FloodZone'] = claims['FloodZone'].str.upper()

    # Parse construction year
    claims['OrigConstDate'] = pd.to_datetime(claims['OrigConstDate'], errors='coerce')
    claims['OrigYear'] = claims['OrigConstDate'].dt.year

    print(f"  Total claims: {len(claims)}")
    print(f"  Flood zones: {claims['FloodZone'].value_counts().to_dict()}")
    print(f"  ZIP codes: {claims['ZIP'].value_counts().head(5).to_dict()}")

    # Create claim ID
    claims['ClaimID'] = range(1, len(claims) + 1)

    return claims[['ClaimID', 'ZIP', 'FloodZone', 'BFE', 'COST', 'OrigYear',
                   'latitude', 'longitude']]


def prepare_buildings():
    """Load and filter building footprints for Dodge County."""
    print("Loading Nebraska building footprints...")
    print("  (This may take a few minutes for 300MB file)")

    # Read with bounding box filter
    buildings = gpd.read_file(
        BUILDINGS_GEOJSON,
        bbox=(-97.0, 41.4, -96.3, 42.0)  # Dodge County bounds
    )

    print(f"  Buildings in Dodge County area: {len(buildings)}")

    # Add building ID
    buildings['BldgID'] = range(1, len(buildings) + 1)

    # Calculate centroid for spatial joins
    buildings['centroid'] = buildings.geometry.centroid

    return buildings


def join_parcels(buildings):
    """Join buildings with parcel data to get assessed values."""
    print("Loading Dodge County parcels...")

    parcels = gpd.read_file(
        PARCELS_GDB,
        layer='StatewideParcels_Current',
        where=f"County_ID = '{DODGE_COUNTY_ID}'"
    )

    print(f"  Parcels loaded: {len(parcels)}")

    # Ensure same CRS
    if buildings.crs != parcels.crs:
        parcels = parcels.to_crs(buildings.crs)

    # Create centroid GeoDataFrame for spatial join
    centroids = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['centroid'],
        crs=buildings.crs
    )

    print("  Performing spatial join...")
    joined = gpd.sjoin(centroids, parcels, how='left', predicate='within')

    # Get relevant parcel attributes
    parcel_attrs = joined[['BldgID', 'Parcel_ID', 'Ph_Zip5', 'Total_Assessed_Value',
                           'Classification_Code', 'BuildingYear']].drop_duplicates(subset='BldgID')

    # Merge back to buildings
    buildings = buildings.merge(parcel_attrs, on='BldgID', how='left')

    # Rename columns
    buildings = buildings.rename(columns={
        'Ph_Zip5': 'ZIP',
        'Total_Assessed_Value': 'Total_Asse',
        'Classification_Code': 'ClassCode',
        'BuildingYear': 'BuildYear'
    })

    print(f"  Buildings with parcel match: {buildings['Parcel_ID'].notna().sum()}")

    return buildings


def extract_elevation(buildings):
    """Extract elevation values for building centroids from DEM."""
    print("Extracting elevation from DEM...")

    try:
        import rasterio
        from rasterstats import point_query

        # Get centroid coordinates
        centroids = buildings['centroid'].to_crs('EPSG:4326')
        coords = [(p.x, p.y) for p in centroids]

        # Find relevant DEM tiles
        dem_files = list(ELEVATION_DIR.glob("*.tif"))
        print(f"  Found {len(dem_files)} DEM tiles")

        elevations = [None] * len(buildings)

        for dem_file in dem_files:
            with rasterio.open(dem_file) as src:
                dem_bounds = src.bounds

                # Find points within this tile
                for i, (x, y) in enumerate(coords):
                    if elevations[i] is None:
                        if (dem_bounds.left <= x <= dem_bounds.right and
                            dem_bounds.bottom <= y <= dem_bounds.top):
                            try:
                                val = list(src.sample([(x, y)]))[0][0]
                                if val != src.nodata:
                                    elevations[i] = float(val)
                            except:
                                pass

        buildings['ELEVATION'] = elevations
        valid_elev = sum(1 for e in elevations if e is not None)
        print(f"  Buildings with elevation: {valid_elev}")

    except ImportError:
        print("  WARNING: rasterio/rasterstats not available, skipping elevation")
        buildings['ELEVATION'] = None

    return buildings


def join_flood_zones(buildings):
    """Join buildings with NFHL flood zones."""
    print("Loading NFHL flood zones...")

    if not NFHL_SHP.exists():
        print(f"  WARNING: NFHL shapefile not found at {NFHL_SHP}")
        print("  Skipping flood zone assignment")
        buildings['FloodZone'] = None
        buildings['SFHA'] = None
        return buildings

    nfhl = gpd.read_file(NFHL_SHP)
    print(f"  Flood zone polygons: {len(nfhl)}")

    # Ensure same CRS
    if buildings.crs != nfhl.crs:
        nfhl = nfhl.to_crs(buildings.crs)

    # Create centroid GeoDataFrame for spatial join
    centroids = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['centroid'],
        crs=buildings.crs
    )

    print("  Performing spatial join with flood zones...")
    joined = gpd.sjoin(centroids, nfhl[['FLD_ZONE', 'ZONE_SUBTY', 'SFHA_TF', 'geometry']],
                       how='left', predicate='within')

    # Get flood zone attributes (take first match if multiple)
    fz_attrs = joined[['BldgID', 'FLD_ZONE', 'SFHA_TF']].drop_duplicates(subset='BldgID')

    # Merge back to buildings
    buildings = buildings.merge(fz_attrs, on='BldgID', how='left')

    # Rename and clean up
    buildings = buildings.rename(columns={
        'FLD_ZONE': 'FloodZone',
        'SFHA_TF': 'SFHA'
    })

    # Standardize flood zones to match claims format
    buildings['FloodZone'] = buildings['FloodZone'].str.upper()

    # Count matches
    n_with_fz = buildings['FloodZone'].notna().sum()
    print(f"  Buildings with flood zone: {n_with_fz}")
    print(f"  Flood zone breakdown:")
    print(buildings['FloodZone'].value_counts().to_string())

    return buildings


def filter_largest_per_parcel(buildings):
    """Keep only the largest building per parcel."""
    print("Filtering to largest building per parcel...")

    # Use assessed value as proxy for building size
    buildings_with_parcel = buildings[buildings['Parcel_ID'].notna()].copy()

    # Group by parcel and keep max value
    idx = buildings_with_parcel.groupby('Parcel_ID')['Total_Asse'].idxmax()
    largest = buildings_with_parcel.loc[idx]

    # Add back buildings without parcels
    no_parcel = buildings[buildings['Parcel_ID'].isna()]

    result = pd.concat([largest, no_parcel], ignore_index=True)
    print(f"  Buildings after filter: {len(result)}")

    return result


def prepare_inundation():
    """Load and prepare inundation layer for validation."""
    print("Loading inundation layer...")

    inund = gpd.read_file(INUNDATION_GPKG)
    print(f"  Inundation polygons: {len(inund)}")
    print(f"  CRS: {inund.crs}")

    return inund


def main():
    """Run full data preparation pipeline."""
    print("=" * 60)
    print("NFIP Claims Disaggregation - Data Preparation")
    print("=" * 60)

    # Create output directory
    DATA_WORK.mkdir(exist_ok=True)

    # 1. Prepare claims
    claims = prepare_claims()
    claims_path = DATA_WORK / "claims_prepared.csv"
    claims.to_csv(claims_path, index=False)
    print(f"  Saved: {claims_path}")

    # 2. Prepare buildings
    buildings = prepare_buildings()

    # 3. Join with parcels
    buildings = join_parcels(buildings)

    # 4. Extract elevation
    buildings = extract_elevation(buildings)

    # 5. Join with NFHL flood zones
    buildings = join_flood_zones(buildings)

    # 6. Filter to largest per parcel
    buildings = filter_largest_per_parcel(buildings)

    # Save buildings
    # Drop centroid column before saving (geometry column issue)
    buildings_save = buildings.drop(columns=['centroid'], errors='ignore')
    buildings_path = DATA_WORK / "buildings_prepared.gpkg"
    buildings_save.to_file(buildings_path, driver="GPKG")
    print(f"  Saved: {buildings_path}")

    # Also save as CSV for easier inspection
    buildings_df = pd.DataFrame(buildings_save.drop(columns=['geometry']))
    buildings_df.to_csv(DATA_WORK / "buildings_prepared.csv", index=False)

    # 7. Prepare inundation
    inund = prepare_inundation()
    inund_path = DATA_WORK / "inundation.gpkg"
    inund.to_file(inund_path, driver="GPKG")
    print(f"  Saved: {inund_path}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Claims: {len(claims)}")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Output directory: {DATA_WORK}")


if __name__ == "__main__":
    main()
