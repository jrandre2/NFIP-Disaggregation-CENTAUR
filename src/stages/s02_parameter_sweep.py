#!/usr/bin/env python3
"""
s02_parameter_sweep.py - Parameter Sweep for Table 1 Recreation
================================================================

This script tests all 4 configurations from Table 1 of the paper:
1. ZIP only + All buildings
2. ZIP only + Largest per parcel
3. ZIP + FZ + All buildings
4. ZIP + FZ + Largest per parcel (baseline)

Each configuration is evaluated with full validation metrics.

Output: data_work/parameter_sweep_results.csv
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"
DATA_RAW = PROJECT_ROOT / "data_raw"

# Input files
CLAIMS_CSV = DATA_WORK / "claims_prepared.csv"
BUILDINGS_GPKG = DATA_WORK / "buildings_prepared.gpkg"
BUILDINGS_ALL_GPKG = DATA_WORK / "buildings_all.gpkg"  # All buildings before largest filter
INUNDATION_GPKG = DATA_WORK / "inundation.gpkg"

# Source data for rebuilding all buildings
BUILDINGS_GEOJSON = Path("/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Building_Footprints/Nebraska.geojson")
PARCELS_GDB = Path("/Volumes/T9/Projects/Freeze and Flight/statewide parcel/NE_2023_statewideparcels.gdb")
NFHL_SHP = DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp"

# Table 1 configurations
CONFIGS = [
    {
        'name': 'ZIP_All',
        'label': 'ZIP only + All buildings',
        'use_flood_zone': False,
        'largest_only': False,
    },
    {
        'name': 'ZIP_Largest',
        'label': 'ZIP only + Largest per parcel',
        'use_flood_zone': False,
        'largest_only': True,
    },
    {
        'name': 'ZIP_FZ_All',
        'label': 'ZIP + Flood Zone + All buildings',
        'use_flood_zone': True,
        'largest_only': False,
    },
    {
        'name': 'ZIP_FZ_Largest',
        'label': 'ZIP + Flood Zone + Largest per parcel',
        'use_flood_zone': True,
        'largest_only': True,
    },
]

# Bootstrap settings
N_ITERATIONS = 1000
SEED = 42


def load_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load claims, all buildings (before largest filter), and inundation data."""
    print("Loading data...")

    claims = pd.read_csv(CLAIMS_CSV)
    inundation = gpd.read_file(INUNDATION_GPKG)

    # Load or create the "all buildings" dataset
    if BUILDINGS_ALL_GPKG.exists():
        print("  Loading cached all-buildings dataset...")
        buildings = gpd.read_file(BUILDINGS_ALL_GPKG)
    else:
        print("  Creating all-buildings dataset (this may take a few minutes)...")
        buildings = create_all_buildings_dataset()

    print(f"  Claims: {len(claims)}")
    print(f"  Buildings (all): {len(buildings)}")

    return claims, buildings, inundation


def create_all_buildings_dataset() -> gpd.GeoDataFrame:
    """Create the all-buildings dataset without the largest-per-parcel filter."""

    # Load raw buildings
    print("    Loading Nebraska building footprints...")
    buildings = gpd.read_file(
        BUILDINGS_GEOJSON,
        bbox=(-97.0, 41.4, -96.3, 42.0)
    )
    print(f"    Raw buildings: {len(buildings)}")

    # Add building ID
    buildings['BldgID'] = range(1, len(buildings) + 1)

    # Calculate centroid
    buildings['centroid'] = buildings.geometry.centroid

    # Load and join parcels
    print("    Loading parcels...")
    parcels = gpd.read_file(
        PARCELS_GDB,
        layer='StatewideParcels_Current',
        where="County_ID = '053'"
    )

    if buildings.crs != parcels.crs:
        parcels = parcels.to_crs(buildings.crs)

    # Spatial join for parcel attributes
    print("    Joining with parcels...")
    centroids = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['centroid'],
        crs=buildings.crs
    )
    joined = gpd.sjoin(centroids, parcels, how='left', predicate='within')

    parcel_attrs = joined[['BldgID', 'Parcel_ID', 'Ph_Zip5', 'Total_Assessed_Value',
                           'Classification_Code', 'BuildingYear']].drop_duplicates(subset='BldgID')
    buildings = buildings.merge(parcel_attrs, on='BldgID', how='left')
    buildings = buildings.rename(columns={
        'Ph_Zip5': 'ZIP',
        'Total_Assessed_Value': 'Total_Asse',
        'Classification_Code': 'ClassCode',
        'BuildingYear': 'BuildYear'
    })

    # Load and join flood zones
    print("    Joining with flood zones...")
    nfhl = gpd.read_file(NFHL_SHP)
    if buildings.crs != nfhl.crs:
        nfhl = nfhl.to_crs(buildings.crs)

    centroids = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['centroid'],
        crs=buildings.crs
    )
    fz_joined = gpd.sjoin(centroids, nfhl[['FLD_ZONE', 'SFHA_TF', 'geometry']],
                          how='left', predicate='within')
    fz_attrs = fz_joined[['BldgID', 'FLD_ZONE', 'SFHA_TF']].drop_duplicates(subset='BldgID')
    buildings = buildings.merge(fz_attrs, on='BldgID', how='left')
    buildings = buildings.rename(columns={'FLD_ZONE': 'FloodZone', 'SFHA_TF': 'SFHA'})
    buildings['FloodZone'] = buildings['FloodZone'].str.upper()

    # Save for future use
    print("    Saving all-buildings dataset...")
    buildings_save = buildings.drop(columns=['centroid'], errors='ignore')
    buildings_save.to_file(BUILDINGS_ALL_GPKG, driver="GPKG")
    print(f"    Saved: {BUILDINGS_ALL_GPKG}")

    return buildings


def filter_buildings(buildings: gpd.GeoDataFrame, largest_only: bool) -> gpd.GeoDataFrame:
    """Filter buildings based on configuration."""
    if not largest_only:
        return buildings.copy()

    # Keep largest building per parcel (by assessed value)
    buildings_with_parcel = buildings[buildings['Parcel_ID'].notna()].copy()

    if len(buildings_with_parcel) == 0:
        return buildings.copy()

    # Group by parcel and keep max value
    idx = buildings_with_parcel.groupby('Parcel_ID')['Total_Asse'].idxmax()
    largest = buildings_with_parcel.loc[idx]

    # Add back buildings without parcels
    no_parcel = buildings[buildings['Parcel_ID'].isna()]

    result = pd.concat([largest, no_parcel], ignore_index=True)
    return result


def build_indices(buildings: gpd.GeoDataFrame, use_flood_zone: bool) -> Tuple[Dict, Dict]:
    """Build lookup indices."""
    idx_zip_fz = defaultdict(list)
    idx_zip = defaultdict(list)

    for _, row in buildings.iterrows():
        bid = row['BldgID']
        zip_code = str(row['ZIP']) if pd.notna(row['ZIP']) else None
        fz = str(row['FloodZone']).upper() if pd.notna(row['FloodZone']) else None

        if zip_code:
            idx_zip[zip_code].append(bid)
            if fz:
                idx_zip_fz[(zip_code, fz)].append(bid)

    return idx_zip_fz, idx_zip


def find_candidates(claim: pd.Series,
                    idx_zip_fz: Dict,
                    idx_zip: Dict,
                    use_flood_zone: bool) -> List[int]:
    """Find candidate buildings for a claim."""
    zip_code = str(claim['ZIP']) if pd.notna(claim['ZIP']) else None
    fz = str(claim['FloodZone']).upper() if pd.notna(claim['FloodZone']) else None

    if not zip_code:
        return []

    if use_flood_zone and fz:
        # Normalize flood zone (A04 -> A, AE -> AE)
        fz_normalized = fz.rstrip('0123456789') if fz.startswith('A') else fz

        # Try exact match first
        candidates = idx_zip_fz.get((zip_code, fz), [])

        # If no match, try normalized
        if not candidates and fz != fz_normalized:
            candidates = idx_zip_fz.get((zip_code, fz_normalized), [])

        # B zone -> X zone fallback
        if not candidates and fz == 'B':
            candidates = idx_zip_fz.get((zip_code, 'X'), [])
    else:
        candidates = idx_zip.get(zip_code, [])

    return candidates


def run_bootstrap(claims: pd.DataFrame,
                  buildings: gpd.GeoDataFrame,
                  config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """Run bootstrap for a single configuration."""

    # Filter buildings
    filtered_buildings = filter_buildings(buildings, config['largest_only'])
    print(f"    Buildings after filter: {len(filtered_buildings)}")

    # Build indices
    idx_zip_fz, idx_zip = build_indices(filtered_buildings, config['use_flood_zone'])

    # Initialize
    rng = np.random.default_rng(seed=SEED)
    building_counts = defaultdict(int)
    n_matched = 0

    # Process claims
    for _, claim in claims.iterrows():
        candidates = find_candidates(claim, idx_zip_fz, idx_zip, config['use_flood_zone'])

        if len(candidates) > 0:
            n_matched += 1
            draws = rng.choice(candidates, N_ITERATIONS, replace=True)
            for bid in draws:
                building_counts[bid] += 1

    # Convert to results
    total_draws = n_matched * N_ITERATIONS

    results = []
    for bid, count in building_counts.items():
        bldg = filtered_buildings[filtered_buildings['BldgID'] == bid].iloc[0]
        results.append({
            'BldgID': bid,
            'draw_count': count,
            'probability': count / total_draws if total_draws > 0 else 0,
        })

    results_df = pd.DataFrame(results)

    stats = {
        'n_matched': n_matched,
        'n_unmatched': len(claims) - n_matched,
        'match_rate': n_matched / len(claims),
        'n_buildings_scored': len(results),
    }

    return results_df, filtered_buildings, stats


def validate(results: pd.DataFrame,
             buildings: gpd.GeoDataFrame,
             inundation: gpd.GeoDataFrame) -> Dict:
    """Calculate validation metrics."""

    # Ensure same CRS
    if buildings.crs != inundation.crs:
        buildings = buildings.to_crs(inundation.crs)

    # Find inundated buildings
    buildings['rep_point'] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['rep_point'],
        crs=buildings.crs
    )

    inundated = gpd.sjoin(points, inundation[['geometry']], how='inner', predicate='within')
    inundated_ids = set(inundated['BldgID'].unique())

    # Build y_true and y_scores for all buildings with probability
    y_true = []
    y_scores = []

    for _, row in results.iterrows():
        bid = row['BldgID']
        y_true.append(1 if bid in inundated_ids else 0)
        y_scores.append(row['probability'])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Calculate metrics
    if sum(y_true) > 0 and sum(y_true) < len(y_true):
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        brier = brier_score_loss(y_true, y_scores)

        # Log loss (clip probabilities to avoid log(0))
        y_scores_clipped = np.clip(y_scores, 1e-10, 1 - 1e-10)
        logloss = log_loss(y_true, y_scores_clipped)

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'log_loss': logloss,
            'n_inundated': sum(y_true),
            'n_scored': len(y_true),
            'prevalence': sum(y_true) / len(y_true),
        }

    return {}


def main():
    """Run parameter sweep."""
    print("=" * 60)
    print("NFIP Disaggregation - Parameter Sweep (Table 1)")
    print("=" * 60)

    # Load data
    claims, buildings, inundation = load_data()

    # Run each configuration
    all_results = []

    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {config['label']}")
        print(f"  Flood zone matching: {config['use_flood_zone']}")
        print(f"  Largest only: {config['largest_only']}")
        print(f"{'='*60}")

        # Run bootstrap
        results_df, filtered_buildings, stats = run_bootstrap(claims, buildings, config)

        print(f"    Claims matched: {stats['n_matched']}/{len(claims)}")
        print(f"    Buildings scored: {stats['n_buildings_scored']}")

        # Validate
        metrics = validate(results_df, filtered_buildings, inundation)

        if metrics:
            print(f"\n    Validation Metrics:")
            print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"      PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"      Brier Score: {metrics['brier_score']:.4f}")
            print(f"      Log-Loss: {metrics['log_loss']:.4f}")

            # Store results
            all_results.append({
                'config_name': config['name'],
                'config_label': config['label'],
                'use_flood_zone': config['use_flood_zone'],
                'largest_only': config['largest_only'],
                'n_claims': len(claims),
                'n_matched': stats['n_matched'],
                'match_rate': stats['match_rate'],
                'n_buildings_scored': stats['n_buildings_scored'],
                **metrics
            })

    # Save results
    results_path = DATA_WORK / "parameter_sweep_results.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Table 1 Recreation")
    print("=" * 60)
    print(f"\n{'Config':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'Brier':>10} {'Log-Loss':>10}")
    print("-" * 65)

    for r in all_results:
        print(f"{r['config_name']:<25} {r['roc_auc']:>10.4f} {r['pr_auc']:>10.4f} "
              f"{r['brier_score']:>10.4f} {r['log_loss']:>10.4f}")

    # Compare to paper values
    print("\n" + "=" * 60)
    print("Comparison to Paper Table 1")
    print("=" * 60)
    paper_values = {
        'ZIP_All': {'roc_auc': 0.707, 'brier': 0.116},
        'ZIP_Largest': {'roc_auc': 0.706, 'brier': 0.092},
        'ZIP_FZ_All': {'roc_auc': 0.902, 'brier': 0.097},
        'ZIP_FZ_Largest': {'roc_auc': 0.950, 'brier': 0.069},
    }

    print(f"\n{'Config':<20} {'Our ROC':>10} {'Paper ROC':>12} {'Diff':>8}")
    print("-" * 55)

    for r in all_results:
        paper = paper_values.get(r['config_name'], {})
        if paper:
            diff = r['roc_auc'] - paper['roc_auc']
            print(f"{r['config_name']:<20} {r['roc_auc']:>10.4f} {paper['roc_auc']:>12.3f} "
                  f"{diff:>+8.4f}")


if __name__ == "__main__":
    main()
