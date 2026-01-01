#!/usr/bin/env python3
"""
s01_bootstrap_disagg.py - Bootstrap Disaggregation of NFIP Claims
==================================================================

This script implements the probabilistic disaggregation algorithm to link
anonymous NFIP claims to building footprints using available attributes:
- ZIP code (required)
- Flood zone (optional)
- Elevation tolerance (optional)
- Assessed value tolerance (optional)

Bootstrap sampling (N iterations) assigns each claim to candidate buildings
multiple times, producing probability estimates for each building.

Output files saved to data_work/:
- bootstrap_results.csv - Building-level probabilities
- disagg_summary.csv - Summary statistics
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"

# Input files
CLAIMS_CSV = DATA_WORK / "claims_prepared.csv"
BUILDINGS_GPKG = DATA_WORK / "buildings_prepared.gpkg"
INUNDATION_GPKG = DATA_WORK / "inundation.gpkg"

# Default configuration (baseline model from paper)
DEFAULT_CONFIG = {
    'use_flood_zone': True,       # Match on flood zone
    'elev_tolerance': 0.5,        # Elevation tolerance in feet (None = disabled)
    'val_tolerance': None,        # Value tolerance % (None = disabled)
    'year_tolerance': None,       # Construction year tolerance in years (None = disabled)
    'n_iterations': 1000,         # Bootstrap iterations
    'seed': 42,                   # Random seed for reproducibility
}


def load_data() -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load prepared claims and buildings data."""
    print("Loading prepared data...")

    claims = pd.read_csv(CLAIMS_CSV)
    buildings = gpd.read_file(BUILDINGS_GPKG)

    print(f"  Claims: {len(claims)}")
    print(f"  Buildings: {len(buildings)}")

    return claims, buildings


def build_indices(buildings: gpd.GeoDataFrame) -> Tuple[Dict, Dict]:
    """Build lookup indices for efficient candidate matching."""
    print("Building spatial indices...")

    # Index by (ZIP, FloodZone)
    idx_zip_fz = defaultdict(list)
    # Index by ZIP only
    idx_zip = defaultdict(list)

    for _, row in buildings.iterrows():
        bid = row['BldgID']
        zip_code = str(row['ZIP']) if pd.notna(row['ZIP']) else None
        fz = str(row['FloodZone']).upper() if pd.notna(row['FloodZone']) else None

        if zip_code:
            idx_zip[zip_code].append(bid)
            if fz:
                idx_zip_fz[(zip_code, fz)].append(bid)

    print(f"  ZIP+FZ groups: {len(idx_zip_fz)}")
    print(f"  ZIP groups: {len(idx_zip)}")

    return idx_zip_fz, idx_zip


def find_candidates(claim: pd.Series,
                    buildings: gpd.GeoDataFrame,
                    idx_zip_fz: Dict,
                    idx_zip: Dict,
                    config: Dict) -> List[int]:
    """Find candidate buildings for a claim based on matching criteria."""

    zip_code = str(claim['ZIP']) if pd.notna(claim['ZIP']) else None
    fz = str(claim['FloodZone']).upper() if pd.notna(claim['FloodZone']) else None

    if not zip_code:
        return []

    # Step 1: Get initial candidates from index
    if config['use_flood_zone'] and fz:
        # Normalize flood zone names (A04 -> A, AE -> AE, etc.)
        fz_normalized = fz.rstrip('0123456789') if fz.startswith('A') else fz

        # Try exact match first
        candidates = idx_zip_fz.get((zip_code, fz), [])

        # If no exact match, try normalized match
        if not candidates and fz != fz_normalized:
            candidates = idx_zip_fz.get((zip_code, fz_normalized), [])

        # Special handling: B zone is minimal risk, similar to X
        if not candidates and fz == 'B':
            candidates = idx_zip_fz.get((zip_code, 'X'), [])
    else:
        candidates = idx_zip.get(zip_code, [])

    if not candidates:
        return []

    # Step 2: Apply elevation filter (if enabled)
    if config['elev_tolerance'] is not None and pd.notna(claim.get('BFE')):
        bfe = float(claim['BFE'])
        tol = float(config['elev_tolerance'])

        filtered = []
        for bid in candidates:
            bldg = buildings[buildings['BldgID'] == bid].iloc[0]
            elev = bldg.get('ELEVATION')
            if pd.notna(elev):
                if abs(float(elev) - bfe) <= tol:
                    filtered.append(bid)

        # Only apply filter if it doesn't eliminate all candidates
        if filtered:
            candidates = filtered

    # Step 3: Apply value filter (if enabled)
    if config['val_tolerance'] is not None and pd.notna(claim.get('COST')):
        cost = float(claim['COST'])
        tol_pct = float(config['val_tolerance']) / 100.0
        lo, hi = cost * (1 - tol_pct), cost * (1 + tol_pct)

        filtered = []
        for bid in candidates:
            bldg = buildings[buildings['BldgID'] == bid].iloc[0]
            val = bldg.get('Total_Asse')
            if pd.notna(val) and float(val) > 0:
                if lo <= float(val) <= hi:
                    filtered.append(bid)

        # Only apply filter if it doesn't eliminate all candidates
        if filtered:
            candidates = filtered

    # Step 4: Apply construction-year filter (if enabled)
    if config.get('year_tolerance') is not None and pd.notna(claim.get('OrigYear')):
        claim_year = int(claim['OrigYear'])
        tol_years = int(config['year_tolerance'])

        filtered = []
        for bid in candidates:
            bldg = buildings[buildings['BldgID'] == bid].iloc[0]
            byear = bldg.get('BuildYear')
            if pd.notna(byear):
                if abs(int(byear) - claim_year) <= tol_years:
                    filtered.append(bid)

        # Only apply filter if it doesn't eliminate all candidates
        if filtered:
            candidates = filtered

    return candidates


def run_bootstrap(claims: pd.DataFrame,
                  buildings: gpd.GeoDataFrame,
                  config: Dict) -> pd.DataFrame:
    """Run bootstrap disaggregation."""
    print(f"\nRunning bootstrap with {config['n_iterations']} iterations...")
    print(f"  Config: flood_zone={config['use_flood_zone']}, "
          f"elev_tol={config['elev_tolerance']}, "
          f"val_tol={config['val_tolerance']}, "
          f"year_tol={config.get('year_tolerance')}")

    # Build indices
    idx_zip_fz, idx_zip = build_indices(buildings)

    # Initialize random generator
    rng = np.random.default_rng(seed=config['seed'])

    # Track assignments
    building_counts = defaultdict(int)
    claim_stats = []

    # Process each claim
    n_matched = 0
    n_unmatched = 0

    for _, claim in claims.iterrows():
        candidates = find_candidates(claim, buildings, idx_zip_fz, idx_zip, config)

        if len(candidates) > 0:
            n_matched += 1
            # Bootstrap sample
            draws = rng.choice(candidates, config['n_iterations'], replace=True)
            for bid in draws:
                building_counts[bid] += 1

            # Record claim stats
            claim_stats.append({
                'ClaimID': claim['ClaimID'],
                'ZIP': claim['ZIP'],
                'FloodZone': claim['FloodZone'],
                'n_candidates': len(candidates),
                'matched': True
            })
        else:
            n_unmatched += 1
            claim_stats.append({
                'ClaimID': claim['ClaimID'],
                'ZIP': claim['ZIP'],
                'FloodZone': claim['FloodZone'],
                'n_candidates': 0,
                'matched': False
            })

    print(f"\n  Claims matched: {n_matched} ({100*n_matched/len(claims):.1f}%)")
    print(f"  Claims unmatched: {n_unmatched} ({100*n_unmatched/len(claims):.1f}%)")

    # Convert counts to probabilities
    total_draws = n_matched * config['n_iterations']

    results = []
    for bid, count in building_counts.items():
        bldg = buildings[buildings['BldgID'] == bid].iloc[0]
        results.append({
            'BldgID': bid,
            'draw_count': count,
            'probability': count / total_draws if total_draws > 0 else 0,
            'ZIP': bldg['ZIP'],
            'FloodZone': bldg['FloodZone'],
            'ELEVATION': bldg.get('ELEVATION'),
            'Total_Asse': bldg.get('Total_Asse'),
            'BuildYear': bldg.get('BuildYear'),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('probability', ascending=False)

    # Save claim stats
    claim_stats_df = pd.DataFrame(claim_stats)
    claim_stats_df.to_csv(DATA_WORK / "claim_matching_stats.csv", index=False)

    return results_df


def validate_with_inundation(results: pd.DataFrame,
                              buildings: gpd.GeoDataFrame) -> Dict:
    """Validate results against observed inundation extent."""
    print("\nValidating against inundation layer...")

    if not INUNDATION_GPKG.exists():
        print("  Inundation layer not found, skipping validation")
        return {}

    inund = gpd.read_file(INUNDATION_GPKG)

    # Ensure same CRS
    if buildings.crs != inund.crs:
        buildings = buildings.to_crs(inund.crs)

    # Check which buildings intersect inundation
    print("  Performing spatial intersection...")

    # Use representative point for intersection test
    buildings['rep_point'] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['rep_point'],
        crs=buildings.crs
    )

    # Spatial join
    inundated = gpd.sjoin(points, inund[['geometry']], how='inner', predicate='within')
    inundated_ids = set(inundated['BldgID'].unique())

    print(f"  Buildings in inundation zone: {len(inundated_ids)}")

    # Create ground truth labels
    all_bldg_ids = set(buildings['BldgID'].values)
    buildings_in_results = set(results['BldgID'].values)

    # Binary classification: was building inundated?
    y_true = []
    y_scores = []

    for _, row in results.iterrows():
        bid = row['BldgID']
        y_true.append(1 if bid in inundated_ids else 0)
        y_scores.append(row['probability'])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Calculate metrics
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_curve,
        auc,
        brier_score_loss,
        log_loss
    )

    if sum(y_true) > 0 and sum(y_true) < len(y_true):
        # We have both positive and negative cases
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        brier = brier_score_loss(y_true, y_scores)

        # Normalize probabilities for log loss
        y_scores_clipped = np.clip(y_scores, 1e-10, 1 - 1e-10)
        y_scores_norm = y_scores_clipped / y_scores_clipped.sum()

        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'n_buildings_scored': len(y_true),
            'n_inundated': sum(y_true),
            'prevalence': sum(y_true) / len(y_true)
        }

        print(f"\n  Validation Metrics:")
        print(f"    ROC-AUC: {roc_auc:.4f}")
        print(f"    PR-AUC: {pr_auc:.4f}")
        print(f"    Brier Score: {brier:.4f}")
        print(f"    Prevalence: {metrics['prevalence']:.4f}")
    else:
        print("  Cannot compute metrics: need both positive and negative cases")
        metrics = {}

    return metrics


def main():
    """Run full bootstrap disaggregation pipeline."""
    print("=" * 60)
    print("NFIP Claims Disaggregation - Bootstrap Analysis")
    print("=" * 60)

    # Load data
    claims, buildings = load_data()

    # Run bootstrap
    config = DEFAULT_CONFIG.copy()
    results = run_bootstrap(claims, buildings, config)

    # Save results
    results_path = DATA_WORK / "bootstrap_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    print(f"  Buildings with non-zero probability: {len(results)}")

    # Validate
    metrics = validate_with_inundation(results, buildings)

    if metrics:
        # Save metrics
        metrics_path = DATA_WORK / "validation_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Saved: {metrics_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Disaggregation complete!")
    print("=" * 60)
    print(f"\nTop 10 buildings by probability:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
