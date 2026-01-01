#!/usr/bin/env python3
"""
Run revision experiments: ZIP vs ZCTA vs CBG, lat/long variants,
construction-year filters, and policy vs claim distribution comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import ks_2samp

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_WORK = PROJECT_ROOT / "data_work"

CBG_SHP = DATA_RAW / "cbg" / "tl_2020_31_bg.shp"

COUNTIES = {
    "dodge": {"fips": "053"},
    "douglas": {"fips": "055"},
    "cass": {"fips": "025"},
    "dakota": {"fips": "043"},
}

NFHL_PATHS = {
    "dodge": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
    "douglas": Path("/Users/jesseandrews/Downloads/31055C_20250325/S_FLD_HAZ_AR.shp"),
    "cass": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
    "dakota": DATA_RAW / "nfhl" / "S_FLD_HAZ_AR.shp",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def normalize_flood_zone(fz: str | None) -> str | None:
    if fz is None:
        return None
    fz = str(fz).upper()
    if fz.startswith("A"):
        return fz.rstrip("0123456789")
    if fz == "B":
        return "X"
    return fz


def build_indices(buildings: pd.DataFrame, key_field: str, use_flood_zone: bool) -> Dict:
    idx = {}
    for _, row in buildings.iterrows():
        key = row.get(key_field)
        if pd.isna(key):
            continue
        key = str(key)
        if use_flood_zone:
            fz = normalize_flood_zone(row.get("FloodZone"))
            if fz is None or pd.isna(fz):
                continue
            idx.setdefault((key, fz), []).append(int(row["BldgID"]))
        else:
            idx.setdefault(key, []).append(int(row["BldgID"]))
    return idx


def get_candidates(record: pd.Series,
                   idx: Dict,
                   key_field: str,
                   use_flood_zone: bool) -> List[int]:
    key = record.get(key_field)
    if pd.isna(key):
        return []
    key = str(key)
    if use_flood_zone:
        fz = normalize_flood_zone(record.get("FloodZone"))
        if fz is None:
            return []
        return idx.get((key, fz), [])
    return idx.get(key, [])


def apply_filters(candidates: List[int],
                  record: pd.Series,
                  bldg_dict: Dict[int, dict],
                  elev_tol: float | None,
                  val_tol: float | None,
                  year_tol: int | None,
                  slope_max: float | None,
                  dist_sfha_max: float | None) -> List[int]:
    if not candidates:
        return candidates

    # Elevation filter
    if elev_tol is not None and pd.notna(record.get("BFE")):
        bfe = float(record["BFE"])
        filtered = [bid for bid in candidates
                    if bldg_dict[bid]["ELEVATION"] is not None
                    and abs(bldg_dict[bid]["ELEVATION"] - bfe) <= elev_tol]
        if filtered:
            candidates = filtered

    # Value filter
    if val_tol is not None and pd.notna(record.get("COST")):
        cost = float(record["COST"])
        tol_pct = float(val_tol) / 100.0
        lo, hi = cost * (1 - tol_pct), cost * (1 + tol_pct)
        filtered = [bid for bid in candidates
                    if bldg_dict[bid]["Total_Asse"] is not None
                    and bldg_dict[bid]["Total_Asse"] > 0
                    and lo <= bldg_dict[bid]["Total_Asse"] <= hi]
        if filtered:
            candidates = filtered

    # Year filter
    if year_tol is not None and pd.notna(record.get("OrigYear")):
        ry = int(record["OrigYear"])
        filtered = []
        for bid in candidates:
            byear = bldg_dict[bid]["BuildYear"]
            if byear is None or pd.isna(byear) or byear == "":
                continue
            try:
                byear_int = int(byear)
            except Exception:
                continue
            if abs(byear_int - ry) <= year_tol:
                filtered.append(bid)
        if filtered:
            candidates = filtered

    # Slope filter (degrees)
    if slope_max is not None:
        filtered = [bid for bid in candidates
                    if bldg_dict[bid]["SlopeDeg"] is not None
                    and bldg_dict[bid]["SlopeDeg"] <= slope_max]
        if filtered:
            candidates = filtered

    # Distance to SFHA filter (meters), applied to Zone X claims only
    if dist_sfha_max is not None:
        fz = normalize_flood_zone(record.get("FloodZone"))
        if fz == "X":
            filtered = [bid for bid in candidates
                        if bldg_dict[bid]["DistToSFHA_m"] is not None
                        and bldg_dict[bid]["DistToSFHA_m"] <= dist_sfha_max]
            if filtered:
                candidates = filtered

    return candidates


def run_bootstrap(records: pd.DataFrame,
                  buildings: pd.DataFrame,
                  idx: Dict,
                  config: Dict,
                  boundary: gpd.GeoDataFrame | None = None) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed=config.get("seed", 42))
    bldg_dict = buildings.set_index("BldgID")[[
        "ELEVATION", "Total_Asse", "BuildYear",
        "LAT", "LON", "LAT_RND", "LON_RND",
        "SlopeDeg", "DistToSFHA_m"
    ]].to_dict(orient="index")

    boundary_union = boundary.geometry.union_all() if boundary is not None else None

    building_counts = {}
    matched = 0
    latlon_used = 0
    latlon_skipped = 0
    total_draws = 0

    for _, rec in records.iterrows():
        candidates = get_candidates(rec, idx, config["key_field"], config["use_flood_zone"])
        candidates = apply_filters(
            candidates,
            rec,
            bldg_dict,
            config.get("elev_tolerance"),
            config.get("val_tolerance"),
            config.get("year_tolerance"),
            config.get("slope_max"),
            config.get("dist_sfha_max"),
        )

        if not candidates:
            continue

        # Lat/long strict filter
        if config.get("latlon_mode") == "strict":
            use_lat = False
            if pd.notna(rec.get("latitude")) and pd.notna(rec.get("longitude")) and boundary_union is not None:
                if boundary_union.contains(Point(rec["longitude"], rec["latitude"])):
                    use_lat = True

            if use_lat:
                latlon_used += 1
                lr = round(float(rec["latitude"]), 1)
                lo = round(float(rec["longitude"]), 1)
                filtered = [bid for bid in candidates
                            if bldg_dict[bid]["LAT_RND"] == lr and bldg_dict[bid]["LON_RND"] == lo]
                if filtered:
                    candidates = filtered
            else:
                latlon_skipped += 1

        if not candidates:
            continue

        # Draw count (policyCount optional)
        draw_mult = 1
        if config.get("use_policy_count") and pd.notna(rec.get("policyCount")):
            try:
                draw_mult = max(1, int(rec.get("policyCount")))
            except Exception:
                draw_mult = 1

        n_draws = config["n_iterations"] * draw_mult
        matched += 1
        total_draws += n_draws

        # Weighted bootstrap
        if config.get("latlon_mode") == "weighted":
            use_lat = False
            if pd.notna(rec.get("latitude")) and pd.notna(rec.get("longitude")) and boundary_union is not None:
                if boundary_union.contains(Point(rec["longitude"], rec["latitude"])):
                    use_lat = True

            if use_lat:
                latlon_used += 1
                lat0 = float(rec["latitude"])
                lon0 = float(rec["longitude"])
                cand_lat = np.array([bldg_dict[bid]["LAT"] for bid in candidates])
                cand_lon = np.array([bldg_dict[bid]["LON"] for bid in candidates])
                dist = haversine_km(lat0, lon0, cand_lat, cand_lon)
                sigma = float(config.get("sigma_km", 10.0))
                weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                weights = weights / weights.sum()
                draws = rng.choice(candidates, n_draws, replace=True, p=weights)
            else:
                latlon_skipped += 1
                draws = rng.choice(candidates, n_draws, replace=True)
        else:
            draws = rng.choice(candidates, n_draws, replace=True)

        for bid in draws:
            building_counts[bid] = building_counts.get(bid, 0) + 1

    results = []
    for bid, count in building_counts.items():
        results.append({
            "BldgID": bid,
            "draw_count": count,
            "probability": count / total_draws if total_draws else 0,
        })

    stats = {
        "matched": matched,
        "total_draws": total_draws,
        "latlon_used": latlon_used,
        "latlon_skipped": latlon_skipped,
    }
    return pd.DataFrame(results), stats


def get_inundated_ids(buildings_gdf: gpd.GeoDataFrame, inund_gdf: gpd.GeoDataFrame) -> set:
    if buildings_gdf.crs != inund_gdf.crs:
        buildings_gdf = buildings_gdf.to_crs(inund_gdf.crs)
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["rep_point"] = buildings_gdf.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings_gdf[["BldgID"]],
        geometry=buildings_gdf["rep_point"],
        crs=buildings_gdf.crs,
    )
    inundated = gpd.sjoin(points, inund_gdf[["geometry"]], how="inner", predicate="within")
    return set(inundated["BldgID"].unique())


def calculate_metrics(results: pd.DataFrame, inundated_ids: set) -> Dict:
    y_true = []
    y_scores = []
    for _, row in results.iterrows():
        bid = row["BldgID"]
        y_true.append(1 if bid in inundated_ids else 0)
        y_scores.append(row["probability"])

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return {}

    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    brier = brier_score_loss(y_true, y_scores)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "n_buildings_scored": len(y_true),
        "n_inundated": int(sum(y_true)),
        "prevalence": float(sum(y_true)) / len(y_true),
    }


def summarize_distribution(results: pd.DataFrame) -> Dict:
    probs = results["probability"].values
    return {
        "mean": float(np.mean(probs)),
        "median": float(np.median(probs)),
        "p90": float(np.quantile(probs, 0.90)),
        "p95": float(np.quantile(probs, 0.95)),
        "p99": float(np.quantile(probs, 0.99)),
    }


def load_county_boundary(county_fips: str) -> gpd.GeoDataFrame:
    cbg = gpd.read_file(CBG_SHP)
    boundary = cbg[cbg["COUNTYFP"] == county_fips].dissolve(by="COUNTYFP")
    if boundary.crs.to_string() != "EPSG:4326":
        boundary = boundary.to_crs("EPSG:4326")
    return boundary


def run_payment_strata(claims: pd.DataFrame,
                       buildings_df: pd.DataFrame,
                       boundary: gpd.GeoDataFrame | None,
                       inundated_ids: set | None,
                       base_dir: Path) -> None:
    if "TotalPayment" not in claims.columns or inundated_ids is None:
        return

    q50 = claims["TotalPayment"].quantile(0.5)
    q90 = claims["TotalPayment"].quantile(0.9)

    strata = [
        ("low", claims[claims["TotalPayment"] <= q50]),
        ("mid", claims[(claims["TotalPayment"] > q50) & (claims["TotalPayment"] <= q90)]),
        ("high", claims[claims["TotalPayment"] > q90]),
    ]

    idx = build_indices(buildings_df, "ZIP", True)
    rows = []
    for label, subset in strata:
        if subset.empty:
            continue
        config = {
            "key_field": "ZIP",
            "use_flood_zone": True,
            "elev_tolerance": 0.5,
            "val_tolerance": None,
            "year_tolerance": None,
            "n_iterations": 1000,
            "seed": 42,
        }
        results, stats = run_bootstrap(subset, buildings_df, idx, config, boundary=boundary)
        metrics = calculate_metrics(results, inundated_ids)
        rows.append({
            "stratum": label,
            "n_claims": len(subset),
            **stats,
            **metrics,
        })

    if rows:
        pd.DataFrame(rows).to_csv(base_dir / "metrics_payment_strata.csv", index=False)


def run_for_county(county_key: str, base_dir_root: Path, skip_policies: bool, skip_inundation: bool) -> None:
    base_dir = base_dir_root / county_key
    claims_path = base_dir / "claims_prepared.csv"
    buildings_path = base_dir / "buildings_prepared.gpkg"
    if not claims_path.exists() or not buildings_path.exists():
        return

    claims = pd.read_csv(claims_path, dtype={"ZIP": str, "ZCTA": str, "CBG": str})
    buildings = gpd.read_file(buildings_path)

    policies = None
    policies_path = base_dir / "policies_prepared.csv"
    if not skip_policies and policies_path.exists():
        policies = pd.read_csv(policies_path, dtype={"ZIP": str, "ZCTA": str, "CBG": str})

    inund = None
    inund_path = base_dir / "inundation.gpkg"
    if not skip_inundation and inund_path.exists():
        inund = gpd.read_file(inund_path)

    county_meta = COUNTIES.get(county_key)
    boundary = None
    if county_meta:
        boundary = load_county_boundary(county_meta["fips"])

    # Build building table for indexing
    buildings_df = buildings.drop(columns=["geometry"]).copy()
    if "DistToSFHA_m" not in buildings_df.columns:
        buildings_df["DistToSFHA_m"] = np.nan
    if "SlopeDeg" not in buildings_df.columns:
        buildings_df["SlopeDeg"] = np.nan

    inundated_ids = get_inundated_ids(buildings, inund) if inund is not None else None

    fz_available = buildings_df["FloodZone"].notna().any()

    # Compute distance to SFHA only for ZIPs with Zone X claims if missing
    if fz_available and buildings_df["DistToSFHA_m"].isna().any():
        zone_x = claims[claims["FloodZone"].astype(str).str.startswith("X", na=False)]
        zip_subset = zone_x["ZIP"].dropna().unique().tolist()
        if zip_subset:
            nfhl_path = NFHL_PATHS.get(county_key)
            if nfhl_path and Path(nfhl_path).exists():
                bounds = tuple(buildings.total_bounds)
                nfhl = gpd.read_file(nfhl_path, bbox=bounds)
                sfha = nfhl[nfhl["SFHA_TF"] == "T"].copy()
                if not sfha.empty:
                    target_crs = "EPSG:26914"
                    sfha_proj = sfha.to_crs(target_crs)
                    sfha_union = sfha_proj.geometry.union_all()
                    subset = buildings[buildings["ZIP"].isin(zip_subset)].copy()
                    points = subset.geometry.to_crs(target_crs).centroid
                    distances = points.distance(sfha_union).astype(float)
                    buildings_df.loc[buildings_df["BldgID"].isin(subset["BldgID"]),
                                     "DistToSFHA_m"] = distances.values

    if fz_available:
        variants = [
            {"name": "baseline_zip_fz", "key_field": "ZIP", "use_flood_zone": True},
            {"name": "zcta_fz", "key_field": "ZCTA", "use_flood_zone": True},
            {"name": "cbg_fz", "key_field": "CBG", "use_flood_zone": True},
            {"name": "latlon_strict", "key_field": "ZIP", "use_flood_zone": True, "latlon_mode": "strict"},
            {"name": "latlon_weighted", "key_field": "ZIP", "use_flood_zone": True, "latlon_mode": "weighted", "sigma_km": 10.0},
            {"name": "year_tol_0", "key_field": "ZIP", "use_flood_zone": True, "year_tolerance": 0},
            {"name": "year_tol_5", "key_field": "ZIP", "use_flood_zone": True, "year_tolerance": 5},
            {"name": "slope_le_2deg", "key_field": "ZIP", "use_flood_zone": True, "slope_max": 2.0},
            {"name": "dist_sfha_500m", "key_field": "ZIP", "use_flood_zone": True, "dist_sfha_max": 500.0},
        ]
    else:
        variants = [
            {"name": "baseline_zip", "key_field": "ZIP", "use_flood_zone": False},
            {"name": "zcta", "key_field": "ZCTA", "use_flood_zone": False},
            {"name": "cbg", "key_field": "CBG", "use_flood_zone": False},
            {"name": "latlon_strict", "key_field": "ZIP", "use_flood_zone": False, "latlon_mode": "strict"},
            {"name": "latlon_weighted", "key_field": "ZIP", "use_flood_zone": False, "latlon_mode": "weighted", "sigma_km": 10.0},
            {"name": "year_tol_0", "key_field": "ZIP", "use_flood_zone": False, "year_tolerance": 0},
            {"name": "year_tol_5", "key_field": "ZIP", "use_flood_zone": False, "year_tolerance": 5},
            {"name": "slope_le_2deg", "key_field": "ZIP", "use_flood_zone": False, "slope_max": 2.0},
        ]

    metrics_rows = []
    for variant in variants:
        config = {
            "key_field": variant["key_field"],
            "use_flood_zone": variant["use_flood_zone"],
            "elev_tolerance": 0.5,
            "val_tolerance": None,
            "year_tolerance": variant.get("year_tolerance"),
            "n_iterations": 1000,
            "seed": 42,
            "latlon_mode": variant.get("latlon_mode"),
            "sigma_km": variant.get("sigma_km"),
            "slope_max": variant.get("slope_max"),
            "dist_sfha_max": variant.get("dist_sfha_max"),
        }
        idx = build_indices(buildings_df, config["key_field"], config["use_flood_zone"])
        results, stats = run_bootstrap(claims, buildings_df, idx, config, boundary=boundary)
        metrics = calculate_metrics(results, inundated_ids) if inundated_ids is not None else {}
        metrics_rows.append({
            "county": county_key,
            "variant": variant["name"],
            **stats,
            **metrics,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(base_dir / "metrics_claims_variants.csv", index=False)

    # Policy vs claim distribution comparison (baseline ZIP+FZ)
    if policies is not None:
        base_config = {
            "key_field": "ZIP",
            "use_flood_zone": True,
            "elev_tolerance": 0.5,
            "val_tolerance": None,
            "year_tolerance": None,
            "n_iterations": 1000,
            "seed": 42,
        }
        idx = build_indices(buildings_df, "ZIP", True)
        claim_results, _ = run_bootstrap(claims, buildings_df, idx, base_config, boundary=boundary)
        pol_config = {**base_config, "use_policy_count": True}
        policy_results, _ = run_bootstrap(policies, buildings_df, idx, pol_config, boundary=boundary)

        dist_rows = [
            {"county": county_key, "group": "claims", **summarize_distribution(claim_results)},
            {"county": county_key, "group": "policies", **summarize_distribution(policy_results)},
        ]
        pd.DataFrame(dist_rows).to_csv(base_dir / "policy_claim_distribution.csv", index=False)

        # KS test + correlation on overlapping buildings
        ks = ks_2samp(claim_results["probability"], policy_results["probability"])
        merged = claim_results.merge(policy_results, on="BldgID", suffixes=("_claim", "_policy"))
        corr = merged["probability_claim"].corr(merged["probability_policy"])
        comp = pd.DataFrame([{
            "county": county_key,
            "ks_stat": ks.statistic,
            "ks_pvalue": ks.pvalue,
            "pearson_corr": corr,
            "n_overlap": len(merged),
        }])
        comp.to_csv(base_dir / "policy_claim_comparison.csv", index=False)

        # Save raw distributions for plots
        claim_results.to_csv(base_dir / "claim_probabilities.csv", index=False)
        policy_results.to_csv(base_dir / "policy_probabilities.csv", index=False)
    else:
        idx = build_indices(buildings_df, "ZIP", fz_available)
        claim_results, _ = run_bootstrap(claims, buildings_df, idx, {
            "key_field": "ZIP",
            "use_flood_zone": fz_available,
            "elev_tolerance": 0.5,
            "val_tolerance": None,
            "year_tolerance": None,
            "n_iterations": 1000,
            "seed": 42,
        }, boundary=boundary)
        claim_results.to_csv(base_dir / "claim_probabilities.csv", index=False)

    run_payment_strata(claims, buildings_df, boundary, inundated_ids, base_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revision experiments for NFIP disaggregation.")
    parser.add_argument("--counties", default="dodge,douglas",
                        help="Comma-separated county keys to run.")
    parser.add_argument("--base-dir", default=str(DATA_WORK / "revision"),
                        help="Base directory containing county subfolders.")
    parser.add_argument("--skip-policies", action="store_true",
                        help="Skip policy comparison outputs.")
    parser.add_argument("--skip-inundation", action="store_true",
                        help="Skip inundation-based metrics.")
    args = parser.parse_args()

    base_dir_root = Path(args.base_dir)
    for key in [c.strip() for c in args.counties.split(",") if c.strip()]:
        run_for_county(key, base_dir_root, args.skip_policies, args.skip_inundation)


if __name__ == "__main__":
    main()
