#!/usr/bin/env python3
"""
s02c_claim_diagnostics.py - Per-Claim Filter Chain Diagnostics
===============================================================

Recreates the diagnostics from NFIPPolicyDescriptivesBootstrap.py:
- Track candidate buildings at each filter stage
- Calculate bootstrap selection variance
- Summarize by flood zone and ZIP code

Output files in data_work/diagnostics/:
- filter_chain.csv - Per-claim filter diagnostics
- summary_by_fz.csv - Summary by flood zone
- summary_by_zip.csv - Summary by ZIP code
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"
DIAGNOSTICS_DIR = DATA_WORK / "diagnostics"

# Input files
CLAIMS_CSV = DATA_WORK / "claims_prepared.csv"
BUILDINGS_GPKG = DATA_WORK / "buildings_prepared.gpkg"

# Configuration (baseline model)
N_ITERATIONS = 1000
SEED = 42
USE_FLOOD_ZONE = True
ELEV_TOLERANCE = 0.5  # feet


def load_data():
    """Load claims and buildings data."""
    print("Loading data...")
    claims = pd.read_csv(CLAIMS_CSV)
    buildings = gpd.read_file(BUILDINGS_GPKG)
    print(f"  Claims: {len(claims)}")
    print(f"  Buildings: {len(buildings)}")
    return claims, buildings


def build_indices(buildings: gpd.GeoDataFrame):
    """Build lookup indices and building dict."""
    print("Building indices...")

    # Building lookup by ID
    bldg_dict = {}
    idx_zip_fz = defaultdict(list)
    idx_zip = defaultdict(list)

    for _, row in buildings.iterrows():
        bid = row['BldgID']
        zip_code = str(row['ZIP']) if pd.notna(row['ZIP']) else None
        fz = str(row['FloodZone']).upper() if pd.notna(row['FloodZone']) else None
        elev = float(row['ELEVATION']) if pd.notna(row.get('ELEVATION')) else None
        value = float(row['Total_Asse']) if pd.notna(row.get('Total_Asse')) else None

        bldg_dict[bid] = {
            'BldgID': bid,
            'ZIP': zip_code,
            'FloodZone': fz,
            'ELEVATION': elev,
            'Total_Asse': value,
        }

        if zip_code:
            idx_zip[zip_code].append(bid)
            if fz:
                idx_zip_fz[(zip_code, fz)].append(bid)

    print(f"  ZIP+FZ groups: {len(idx_zip_fz)}")
    print(f"  ZIP groups: {len(idx_zip)}")

    return bldg_dict, idx_zip_fz, idx_zip


def get_initial_candidates(claim, idx_zip_fz, idx_zip, use_flood_zone):
    """Get initial candidates based on ZIP and optionally flood zone."""
    zip_code = str(claim['ZIP']) if pd.notna(claim['ZIP']) else None
    fz = str(claim['FloodZone']).upper() if pd.notna(claim['FloodZone']) else None

    if not zip_code:
        return []

    if use_flood_zone and fz:
        fz_normalized = fz.rstrip('0123456789') if fz.startswith('A') else fz
        candidates = idx_zip_fz.get((zip_code, fz), [])
        if not candidates and fz != fz_normalized:
            candidates = idx_zip_fz.get((zip_code, fz_normalized), [])
        if not candidates and fz == 'B':
            candidates = idx_zip_fz.get((zip_code, 'X'), [])
    else:
        candidates = idx_zip.get(zip_code, [])

    return list(candidates)


def apply_elevation_filter(candidates, bldg_dict, claim_bfe, elev_tol):
    """Apply elevation tolerance filter."""
    if elev_tol is None or pd.isna(claim_bfe):
        return candidates

    filtered = []
    for bid in candidates:
        bldg = bldg_dict[bid]
        elev = bldg['ELEVATION']
        if elev is not None:
            if abs(elev - float(claim_bfe)) <= elev_tol:
                filtered.append(bid)

    # If filter eliminates all, keep original candidates
    return filtered if filtered else candidates


def run_diagnostics(claims, buildings):
    """Run per-claim diagnostics."""
    print("\nRunning per-claim diagnostics...")

    bldg_dict, idx_zip_fz, idx_zip = build_indices(buildings)
    rng = np.random.default_rng(seed=SEED)

    results = []

    for _, claim in claims.iterrows():
        claim_id = claim['ClaimID']
        zip_code = str(claim['ZIP']) if pd.notna(claim['ZIP']) else None
        fz = str(claim['FloodZone']).upper() if pd.notna(claim['FloodZone']) else None
        bfe = claim.get('BFE')

        # Stage 1: Initial candidates (ZIP + FZ)
        initial = get_initial_candidates(claim, idx_zip_fz, idx_zip, USE_FLOOD_ZONE)
        n_initial = len(initial)

        # Stage 2: After elevation filter
        after_elev = apply_elevation_filter(initial, bldg_dict, bfe, ELEV_TOLERANCE)
        n_after_elev = len(after_elev)

        # Stage 3: Final candidates (same as after_elev for now)
        final = after_elev
        n_final = len(final)

        # Bootstrap statistics
        n_unique_sel = 0
        var_sel = 0.0
        top_share = 0.0
        mean_count = 0.0
        median_count = 0.0

        if n_final > 0:
            draws = rng.choice(final, N_ITERATIONS, replace=True)
            unique_ids, counts = np.unique(draws, return_counts=True)
            n_unique_sel = len(unique_ids)
            var_sel = counts.var(ddof=0)
            top_share = counts.max() / N_ITERATIONS
            mean_count = counts.mean()
            median_count = np.median(counts)

        results.append({
            'ClaimID': claim_id,
            'ZIP': zip_code,
            'FloodZone': fz,
            'BFE': bfe,
            'n_initial': n_initial,
            'n_after_elev': n_after_elev,
            'n_final': n_final,
            'n_unique_sel': n_unique_sel,
            'var_sel': round(var_sel, 3),
            'top_share': round(top_share, 4),
            'mean_count': round(mean_count, 2),
            'median_count': round(median_count, 2),
            'matched': n_final > 0
        })

    return pd.DataFrame(results)


def summarize_by_group(df, group_col):
    """Summarize diagnostics by a grouping column."""
    summary = df.groupby(group_col).agg({
        'n_initial': ['mean', 'median', 'min', 'max'],
        'n_final': ['mean', 'median', 'min', 'max'],
        'var_sel': ['mean', 'median', 'std', 'min', 'max'],
        'top_share': ['mean', 'median'],
        'matched': ['sum', 'count']
    }).round(3)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary['match_rate'] = (summary['matched_sum'] / summary['matched_count']).round(3)

    return summary.reset_index()


def main():
    """Run full diagnostics pipeline."""
    print("=" * 60)
    print("NFIP Disaggregation - Per-Claim Diagnostics")
    print("=" * 60)

    # Create output directory
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    claims, buildings = load_data()

    # Run diagnostics
    df = run_diagnostics(claims, buildings)

    # Save per-claim results
    filter_chain_path = DIAGNOSTICS_DIR / "filter_chain.csv"
    df.to_csv(filter_chain_path, index=False)
    print(f"\nSaved: {filter_chain_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Per-Claim Summary")
    print("=" * 60)
    print(f"  Total claims: {len(df)}")
    print(f"  Matched claims: {df['matched'].sum()} ({100*df['matched'].mean():.1f}%)")
    print(f"  Mean initial candidates: {df['n_initial'].mean():.1f}")
    print(f"  Mean final candidates: {df['n_final'].mean():.1f}")
    print(f"  Mean top_share: {df['top_share'].mean():.3f}")
    print(f"  Mean variance: {df['var_sel'].mean():.2f}")

    # Summary by flood zone
    fz_summary = summarize_by_group(df, 'FloodZone')
    fz_path = DIAGNOSTICS_DIR / "summary_by_fz.csv"
    fz_summary.to_csv(fz_path, index=False)
    print(f"\nSaved: {fz_path}")

    print("\n" + "=" * 60)
    print("Summary by Flood Zone")
    print("=" * 60)
    print(fz_summary.to_string(index=False))

    # Summary by ZIP code
    zip_summary = summarize_by_group(df, 'ZIP')
    zip_path = DIAGNOSTICS_DIR / "summary_by_zip.csv"
    zip_summary.to_csv(zip_path, index=False)
    print(f"\nSaved: {zip_path}")

    print("\n" + "=" * 60)
    print("Summary by ZIP Code")
    print("=" * 60)
    print(zip_summary.to_string(index=False))

    # Distribution of candidate counts
    print("\n" + "=" * 60)
    print("Candidate Count Distribution")
    print("=" * 60)
    print(f"  Min candidates: {df['n_final'].min()}")
    print(f"  25th percentile: {df['n_final'].quantile(0.25):.0f}")
    print(f"  Median candidates: {df['n_final'].median():.0f}")
    print(f"  75th percentile: {df['n_final'].quantile(0.75):.0f}")
    print(f"  Max candidates: {df['n_final'].max()}")

    # Claims with few candidates (high confidence)
    few_candidates = df[df['n_final'] <= 5]
    print(f"\n  Claims with ≤5 candidates: {len(few_candidates)} ({100*len(few_candidates)/len(df):.1f}%)")

    # Claims with many candidates (low confidence)
    many_candidates = df[df['n_final'] > 100]
    print(f"  Claims with >100 candidates: {len(many_candidates)} ({100*len(many_candidates)/len(df):.1f}%)")


if __name__ == "__main__":
    main()
