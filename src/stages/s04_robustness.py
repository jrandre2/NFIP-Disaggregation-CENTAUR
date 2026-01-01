#!/usr/bin/env python3
"""
s04_robustness.py - Robustness Checks for Bootstrap Disaggregation
===================================================================

Implements robustness checks:
1. Random seed stability (multiple seeds)
2. Jackknife (leave-one-claim-out)
3. Spatial cross-validation (leave-ZIP-out)
4. Bootstrap confidence intervals for metrics

Output: data_work/robustness/
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"
ROBUSTNESS_DIR = DATA_WORK / "robustness"

# Input files
CLAIMS_CSV = DATA_WORK / "claims_prepared.csv"
BUILDINGS_GPKG = DATA_WORK / "buildings_prepared.gpkg"
INUNDATION_GPKG = DATA_WORK / "inundation.gpkg"

# Baseline configuration
N_ITERATIONS = 1000
ELEV_TOLERANCE = 0.5


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

        bldg_dict[bid] = {'ELEVATION': elev}

        if zip_code and fz:
            idx_zip_fz[(zip_code, fz)].append(bid)

    return bldg_dict, idx_zip_fz


def get_candidates(claim, idx_zip_fz, bldg_dict, elev_tol):
    """Get candidate buildings for a claim."""
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

    # Apply elevation filter
    if elev_tol is not None and pd.notna(claim.get('BFE')):
        bfe = float(claim['BFE'])
        filtered = [bid for bid in candidates
                    if bldg_dict[bid]['ELEVATION'] is not None
                    and abs(bldg_dict[bid]['ELEVATION'] - bfe) <= elev_tol]
        if filtered:
            candidates = filtered

    return candidates


def run_bootstrap(claims, bldg_dict, idx_zip_fz, n_iter, elev_tol, seed=42):
    """Run bootstrap with given seed."""
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

    total_draws = n_matched * n_iter
    results = {bid: count / total_draws for bid, count in building_counts.items()}

    return results, n_matched


def get_ground_truth(buildings, inundation):
    """Get inundated building IDs."""
    if buildings.crs != inundation.crs:
        buildings = buildings.to_crs(inundation.crs)

    buildings['rep_point'] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[['BldgID']],
        geometry=buildings['rep_point'],
        crs=buildings.crs
    )

    inundated = gpd.sjoin(points, inundation[['geometry']], how='inner', predicate='within')
    return set(inundated['BldgID'].unique())


def calculate_metrics(results, ground_truth):
    """Calculate validation metrics."""
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
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'brier_score': brier}

    return None


def test_seed_stability(claims, buildings, inundation, bldg_dict, idx_zip_fz, n_seeds=20):
    """Test stability across different random seeds."""
    print("\n" + "=" * 60)
    print("Testing Random Seed Stability")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)
    results = []

    seeds = list(range(1, n_seeds + 1))
    for seed in seeds:
        print(f"  Seed {seed:2d}...", end=" ")
        probs, _ = run_bootstrap(claims, bldg_dict, idx_zip_fz, N_ITERATIONS, ELEV_TOLERANCE, seed=seed)
        metrics = calculate_metrics(probs, ground_truth)
        if metrics:
            print(f"ROC-AUC={metrics['roc_auc']:.4f}")
            results.append({
                'seed': seed,
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'brier_score': metrics['brier_score'],
            })

    df = pd.DataFrame(results)

    # Summary statistics
    print(f"\n  ROC-AUC: mean={df['roc_auc'].mean():.4f}, std={df['roc_auc'].std():.4f}")
    print(f"  PR-AUC:  mean={df['pr_auc'].mean():.4f}, std={df['pr_auc'].std():.4f}")
    print(f"  Brier:   mean={df['brier_score'].mean():.4f}, std={df['brier_score'].std():.4f}")

    df.to_csv(ROBUSTNESS_DIR / "seed_stability.csv", index=False)
    return df


def jackknife_claims(claims, buildings, inundation, bldg_dict, idx_zip_fz):
    """Leave-one-claim-out jackknife analysis."""
    print("\n" + "=" * 60)
    print("Jackknife Analysis (Leave-One-Claim-Out)")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)

    # Full model baseline
    full_probs, _ = run_bootstrap(claims, bldg_dict, idx_zip_fz, N_ITERATIONS, ELEV_TOLERANCE)
    full_metrics = calculate_metrics(full_probs, ground_truth)
    print(f"  Full model: ROC-AUC={full_metrics['roc_auc']:.4f}")

    results = []
    n_claims = len(claims)

    for i in range(n_claims):
        if (i + 1) % 50 == 0:
            print(f"  Processing claim {i + 1}/{n_claims}...")

        # Leave one claim out
        claims_loo = claims.drop(claims.index[i])
        claim_left_out = claims.iloc[i]

        probs, _ = run_bootstrap(claims_loo, bldg_dict, idx_zip_fz, N_ITERATIONS, ELEV_TOLERANCE)
        metrics = calculate_metrics(probs, ground_truth)

        if metrics:
            influence = full_metrics['roc_auc'] - metrics['roc_auc']
            results.append({
                'claim_index': i,
                'ClaimID': claim_left_out.get('ClaimID', i),
                'ZIP': claim_left_out.get('ZIP'),
                'FloodZone': claim_left_out.get('FloodZone'),
                'roc_auc': metrics['roc_auc'],
                'influence': influence,
            })

    df = pd.DataFrame(results)

    # Identify influential claims
    influential = df.nlargest(5, 'influence')
    print(f"\n  Most influential claims (positive = removing hurts model):")
    for _, row in influential.iterrows():
        print(f"    ClaimID {row['ClaimID']}: influence={row['influence']:.4f}")

    df.to_csv(ROBUSTNESS_DIR / "jackknife_claims.csv", index=False)
    return df


def spatial_cv(claims, buildings, inundation, bldg_dict, idx_zip_fz):
    """Leave-one-ZIP-out spatial cross-validation."""
    print("\n" + "=" * 60)
    print("Spatial Cross-Validation (Leave-ZIP-Out)")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)

    # Get unique ZIPs
    zip_codes = claims['ZIP'].dropna().unique()
    print(f"  Testing {len(zip_codes)} ZIP codes...")

    results = []

    for zip_code in zip_codes:
        # Leave one ZIP out
        claims_loo = claims[claims['ZIP'] != zip_code]
        n_left_out = len(claims[claims['ZIP'] == zip_code])

        if len(claims_loo) == 0:
            continue

        probs, n_matched = run_bootstrap(claims_loo, bldg_dict, idx_zip_fz, N_ITERATIONS, ELEV_TOLERANCE)
        metrics = calculate_metrics(probs, ground_truth)

        if metrics:
            print(f"  ZIP {zip_code}: n_out={n_left_out}, ROC-AUC={metrics['roc_auc']:.4f}")
            results.append({
                'zip_code': zip_code,
                'n_claims_out': n_left_out,
                'n_claims_in': len(claims_loo),
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'brier_score': metrics['brier_score'],
            })

    df = pd.DataFrame(results)

    print(f"\n  Mean ROC-AUC across folds: {df['roc_auc'].mean():.4f} (±{df['roc_auc'].std():.4f})")

    df.to_csv(ROBUSTNESS_DIR / "spatial_cv.csv", index=False)
    return df


def bootstrap_ci(claims, buildings, inundation, bldg_dict, idx_zip_fz, n_bootstrap=100):
    """Bootstrap confidence intervals for metrics."""
    print("\n" + "=" * 60)
    print("Bootstrap Confidence Intervals")
    print("=" * 60)

    ground_truth = get_ground_truth(buildings, inundation)
    rng = np.random.default_rng(seed=42)

    results = []

    for b in range(n_bootstrap):
        if (b + 1) % 20 == 0:
            print(f"  Bootstrap iteration {b + 1}/{n_bootstrap}...")

        # Resample claims with replacement
        claims_boot = claims.sample(n=len(claims), replace=True, random_state=rng.integers(1e9))

        probs, _ = run_bootstrap(claims_boot, bldg_dict, idx_zip_fz, N_ITERATIONS, ELEV_TOLERANCE,
                                  seed=rng.integers(1e9))
        metrics = calculate_metrics(probs, ground_truth)

        if metrics:
            results.append({
                'bootstrap_iter': b,
                'roc_auc': metrics['roc_auc'],
                'pr_auc': metrics['pr_auc'],
                'brier_score': metrics['brier_score'],
            })

    df = pd.DataFrame(results)

    # Calculate 95% CI
    for metric in ['roc_auc', 'pr_auc', 'brier_score']:
        lower = df[metric].quantile(0.025)
        upper = df[metric].quantile(0.975)
        mean = df[metric].mean()
        print(f"  {metric}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

    df.to_csv(ROBUSTNESS_DIR / "bootstrap_ci.csv", index=False)
    return df


def main():
    """Run all robustness checks."""
    print("=" * 60)
    print("NFIP Disaggregation - Robustness Checks")
    print("=" * 60)

    # Create output directory
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    claims, buildings, inundation = load_data()

    # Build indices
    bldg_dict, idx_zip_fz = build_indices(buildings)

    # Run robustness checks
    seed_df = test_seed_stability(claims, buildings, inundation, bldg_dict, idx_zip_fz, n_seeds=20)
    spatial_df = spatial_cv(claims, buildings, inundation, bldg_dict, idx_zip_fz)
    jackknife_df = jackknife_claims(claims, buildings, inundation, bldg_dict, idx_zip_fz)
    ci_df = bootstrap_ci(claims, buildings, inundation, bldg_dict, idx_zip_fz, n_bootstrap=100)

    # Summary
    print("\n" + "=" * 60)
    print("Robustness Summary")
    print("=" * 60)

    print("\nSeed Stability (20 seeds):")
    print(f"  ROC-AUC: {seed_df['roc_auc'].mean():.4f} +/- {seed_df['roc_auc'].std():.4f}")

    print("\nSpatial CV (Leave-ZIP-Out):")
    print(f"  ROC-AUC: {spatial_df['roc_auc'].mean():.4f} +/- {spatial_df['roc_auc'].std():.4f}")

    print("\nJackknife (Leave-One-Claim-Out):")
    print(f"  Max influence: {jackknife_df['influence'].abs().max():.4f}")

    print("\nBootstrap 95% CI:")
    for metric in ['roc_auc', 'pr_auc', 'brier_score']:
        lower = ci_df[metric].quantile(0.025)
        upper = ci_df[metric].quantile(0.975)
        print(f"  {metric}: ({lower:.4f}, {upper:.4f})")

    print(f"\nResults saved to: {ROBUSTNESS_DIR}")


if __name__ == "__main__":
    main()
