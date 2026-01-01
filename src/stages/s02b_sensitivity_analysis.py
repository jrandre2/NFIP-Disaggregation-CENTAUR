#!/usr/bin/env python3
"""
s02b_sensitivity_analysis.py - Parameter Sensitivity Analysis
==============================================================

Tests sensitivity of results to various parameters:
1. Bootstrap iterations: [100, 500, 1000, 5000, 10000]
2. Elevation tolerance: [0.25, 0.5, 1.0, 2.0, None] feet
3. Value tolerance: [5%, 10%, 15%, 20%, None]
4. Inundation buffer: [0, 50, 100, 250, 500] feet

Output: data_work/sensitivity/
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
from itertools import product

from sklearn.metrics import roc_auc_score, brier_score_loss

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"
SENSITIVITY_DIR = DATA_WORK / "sensitivity"

# Input files
CLAIMS_CSV = DATA_WORK / "claims_prepared.csv"
BUILDINGS_GPKG = DATA_WORK / "buildings_prepared.gpkg"
INUNDATION_GPKG = DATA_WORK / "inundation.gpkg"

# Parameter ranges to test
ITERATION_VALUES = [100, 500, 1000, 5000]
ELEV_TOLERANCES = [None, 0.25, 0.5, 1.0, 2.0]  # None = disabled
BUFFER_DISTANCES = [0, 50, 100, 250]  # feet

# Fixed baseline config
BASELINE_CONFIG = {
    'use_flood_zone': True,
    'largest_only': True,
    'elev_tolerance': 0.5,
    'n_iterations': 1000,
}


def load_data():
    """Load data."""
    print("Loading data...")
    claims = pd.read_csv(CLAIMS_CSV)
    buildings = gpd.read_file(BUILDINGS_GPKG)
    inundation = gpd.read_file(INUNDATION_GPKG)
    print(f"  Claims: {len(claims)}, Buildings: {len(buildings)}")
    return claims, buildings, inundation


def build_indices(buildings):
    """Build lookup indices."""
    bldg_dict = {}
    idx_zip_fz = defaultdict(list)

    for _, row in buildings.iterrows():
        bid = row['BldgID']
        zip_code = str(row['ZIP']) if pd.notna(row['ZIP']) else None
        fz = str(row['FloodZone']).upper() if pd.notna(row['FloodZone']) else None
        elev = float(row['ELEVATION']) if pd.notna(row.get('ELEVATION')) else None
        value = float(row['Total_Asse']) if pd.notna(row.get('Total_Asse')) else None

        bldg_dict[bid] = {'ELEVATION': elev, 'Total_Asse': value}

        if zip_code and fz:
            idx_zip_fz[(zip_code, fz)].append(bid)

    return bldg_dict, idx_zip_fz


def get_candidates(claim, idx_zip_fz, bldg_dict, elev_tol):
    """Get candidates with optional elevation filter."""
    zip_code = str(claim['ZIP']) if pd.notna(claim['ZIP']) else None
    fz = str(claim['FloodZone']).upper() if pd.notna(claim['FloodZone']) else None

    if not zip_code or not fz:
        return []

    fz_normalized = fz.rstrip('0123456789') if fz.startswith('A') else fz
    candidates = idx_zip_fz.get((zip_code, fz), [])
    if not candidates and fz != fz_normalized:
        candidates = idx_zip_fz.get((zip_code, fz_normalized), [])
    if not candidates and fz == 'B':
        candidates = idx_zip_fz.get((zip_code, 'X'), [])

    candidates = list(candidates)

    # Apply elevation filter if enabled
    if elev_tol is not None and pd.notna(claim.get('BFE')):
        bfe = float(claim['BFE'])
        filtered = [bid for bid in candidates
                    if bldg_dict[bid]['ELEVATION'] is not None
                    and abs(bldg_dict[bid]['ELEVATION'] - bfe) <= elev_tol]
        if filtered:
            candidates = filtered

    return candidates


def run_bootstrap(claims, bldg_dict, idx_zip_fz, n_iter, elev_tol, seed=42):
    """Run bootstrap with given parameters."""
    rng = np.random.default_rng(seed=seed)
    building_counts = defaultdict(int)
    n_matched = 0

    for _, claim in claims.iterrows():
        candidates = get_candidates(claim, idx_zip_fz, bldg_dict, elev_tol)
        if candidates:
            n_matched += 1
            draws = rng.choice(candidates, n_iter, replace=True)
            for bid in draws:
                building_counts[bid] += 1

    # Convert to probabilities
    total_draws = n_matched * n_iter
    results = {bid: count / total_draws for bid, count in building_counts.items()}

    return results, n_matched


def get_ground_truth(buildings, inundation, buffer_ft=0):
    """Get inundated building IDs with optional buffer."""
    if buildings.crs != inundation.crs:
        buildings = buildings.to_crs(inundation.crs)

    # Buffer inundation if needed
    if buffer_ft > 0:
        # Convert feet to meters (approximate)
        buffer_m = buffer_ft * 0.3048
        inundation = inundation.copy()
        inundation['geometry'] = inundation.geometry.buffer(buffer_m)

    buildings['rep_point'] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['rep_point'],
        crs=buildings.crs
    )

    inundated = gpd.sjoin(points, inundation[['geometry']], how='inner', predicate='within')
    return set(inundated['BldgID'].unique())


def calculate_metrics(results, ground_truth):
    """Calculate ROC-AUC and Brier score."""
    y_true = []
    y_scores = []

    for bid, prob in results.items():
        y_true.append(1 if bid in ground_truth else 0)
        y_scores.append(prob)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if sum(y_true) > 0 and sum(y_true) < len(y_true):
        roc_auc = roc_auc_score(y_true, y_scores)
        brier = brier_score_loss(y_true, y_scores)
        return {'roc_auc': roc_auc, 'brier_score': brier, 'n_inundated': sum(y_true)}

    return None


def test_iteration_convergence(claims, buildings, inundation, bldg_dict, idx_zip_fz):
    """Test convergence with different bootstrap iterations."""
    print("\n" + "=" * 60)
    print("Testing Bootstrap Iteration Convergence")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)
    results = []

    for n_iter in ITERATION_VALUES:
        print(f"  Testing N={n_iter}...", end=" ")
        probs, n_matched = run_bootstrap(claims, bldg_dict, idx_zip_fz, n_iter,
                                         BASELINE_CONFIG['elev_tolerance'])
        metrics = calculate_metrics(probs, ground_truth)
        if metrics:
            print(f"ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}")
            results.append({
                'n_iterations': n_iter,
                'roc_auc': metrics['roc_auc'],
                'brier_score': metrics['brier_score'],
                'n_matched': n_matched,
                'n_buildings_scored': len(probs),
            })

    df = pd.DataFrame(results)
    df.to_csv(SENSITIVITY_DIR / "iteration_convergence.csv", index=False)
    return df


def test_elevation_sensitivity(claims, buildings, inundation, bldg_dict, idx_zip_fz):
    """Test sensitivity to elevation tolerance."""
    print("\n" + "=" * 60)
    print("Testing Elevation Tolerance Sensitivity")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)
    results = []

    for elev_tol in ELEV_TOLERANCES:
        label = f"{elev_tol} ft" if elev_tol else "Disabled"
        print(f"  Testing elev_tol={label}...", end=" ")
        probs, n_matched = run_bootstrap(claims, bldg_dict, idx_zip_fz,
                                         BASELINE_CONFIG['n_iterations'], elev_tol)
        metrics = calculate_metrics(probs, ground_truth)
        if metrics:
            print(f"ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}")
            results.append({
                'elev_tolerance': elev_tol if elev_tol else 'None',
                'roc_auc': metrics['roc_auc'],
                'brier_score': metrics['brier_score'],
                'n_matched': n_matched,
                'n_buildings_scored': len(probs),
            })

    df = pd.DataFrame(results)
    df.to_csv(SENSITIVITY_DIR / "elevation_sensitivity.csv", index=False)
    return df


def test_buffer_sensitivity(claims, buildings, inundation, bldg_dict, idx_zip_fz):
    """Test sensitivity to inundation buffer distance."""
    print("\n" + "=" * 60)
    print("Testing Inundation Buffer Sensitivity")
    print("=" * 60)

    # Run baseline bootstrap once
    probs, n_matched = run_bootstrap(claims, bldg_dict, idx_zip_fz,
                                     BASELINE_CONFIG['n_iterations'],
                                     BASELINE_CONFIG['elev_tolerance'])

    results = []

    for buffer_ft in BUFFER_DISTANCES:
        print(f"  Testing buffer={buffer_ft} ft...", end=" ")
        ground_truth = get_ground_truth(buildings, inundation, buffer_ft)
        metrics = calculate_metrics(probs, ground_truth)
        if metrics:
            print(f"ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}, "
                  f"Inundated={metrics['n_inundated']}")
            results.append({
                'buffer_ft': buffer_ft,
                'roc_auc': metrics['roc_auc'],
                'brier_score': metrics['brier_score'],
                'n_inundated': metrics['n_inundated'],
            })

    df = pd.DataFrame(results)
    df.to_csv(SENSITIVITY_DIR / "buffer_sensitivity.csv", index=False)
    return df


def main():
    """Run all sensitivity analyses."""
    print("=" * 60)
    print("NFIP Disaggregation - Sensitivity Analysis")
    print("=" * 60)

    # Create output directory
    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    claims, buildings, inundation = load_data()

    # Build indices once
    bldg_dict, idx_zip_fz = build_indices(buildings)

    # Run sensitivity tests
    iter_df = test_iteration_convergence(claims, buildings, inundation, bldg_dict, idx_zip_fz)
    elev_df = test_elevation_sensitivity(claims, buildings, inundation, bldg_dict, idx_zip_fz)
    buffer_df = test_buffer_sensitivity(claims, buildings, inundation, bldg_dict, idx_zip_fz)

    # Summary
    print("\n" + "=" * 60)
    print("Sensitivity Analysis Complete")
    print("=" * 60)

    print("\nIteration Convergence:")
    print(iter_df.to_string(index=False))

    print("\nElevation Tolerance:")
    print(elev_df.to_string(index=False))

    print("\nInundation Buffer:")
    print(buffer_df.to_string(index=False))

    print(f"\nResults saved to: {SENSITIVITY_DIR}")


if __name__ == "__main__":
    main()
