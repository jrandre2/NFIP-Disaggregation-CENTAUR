#!/usr/bin/env python3
"""
s02d_ia_validation.py - FEMA Housing Assistance validation
==========================================================

Generates ZIP-level validation metrics by comparing disaggregated NFIP
claim likelihoods to FEMA Housing Assistance Owners totals for matching
events and counties.

Outputs:
- data_work/ia_validation/ia_zip_detail.csv
- data_work/ia_validation/ia_zip_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"
DATA_EXTERNAL = PROJECT_ROOT / "data_external"

IA_PARQUET = DATA_EXTERNAL / "fema_ia" / "HousingAssistanceOwners.parquet"
OUT_DIR = DATA_WORK / "ia_validation"
MIN_CORR_N = 4
CLAIM_DRAWS_PER_CLAIM = 1000

COUNTIES = {
    "dodge": {"name": "Dodge", "fips": "053"},
    "douglas": {"name": "Douglas", "fips": "055"},
    "cass": {"name": "Cass", "fips": "025"},
    "dakota": {"name": "Dakota", "fips": "043"},
    "knox": {"name": "Knox", "fips": "107"},
}

EVENTS = [
    {
        "label": "2019_midwest_flood",
        "disaster_number": 4420,
        "base_dir": DATA_WORK / "revision",
        "counties": ["dodge", "douglas"],
    },
]


def load_ia_data(path: Path) -> pd.DataFrame:
    """Load FEMA Housing Assistance Owners data with a stable pandas path."""
    table = pq.read_table(path)
    df = pd.DataFrame(table.to_pydict())
    df["zipCode"] = df["zipCode"].astype(str).str.zfill(5)
    df["county"] = df["county"].astype(str)
    return df


def load_claim_zip_probs(base_dir: Path, county_key: str) -> pd.DataFrame | None:
    """Aggregate building-level claim probabilities to ZIPs (fallback to ZCTA)."""
    claims_path = base_dir / county_key / "claim_probabilities.csv"
    buildings_path = base_dir / county_key / "buildings_prepared.csv"
    if not claims_path.exists() or not buildings_path.exists():
        return None

    probs = pd.read_csv(claims_path)
    if probs.empty:
        return None
    if "draw_count" not in probs.columns:
        return None

    total_draws = probs["draw_count"].sum()
    if total_draws <= 0:
        return None
    matched_claims = total_draws / CLAIM_DRAWS_PER_CLAIM

    buildings = pd.read_csv(buildings_path, dtype={"ZIP": str, "ZCTA": str})
    merged = probs.merge(buildings[["BldgID", "ZIP", "ZCTA"]], on="BldgID", how="left")
    merged["ZIP"] = merged["ZIP"].replace("", pd.NA)
    merged["ZIP"] = merged["ZIP"].fillna(merged["ZCTA"])
    merged = merged.dropna(subset=["ZIP"])
    merged["ZIP"] = merged["ZIP"].astype(str).str.zfill(5)
    agg = merged.groupby("ZIP", as_index=False)["probability"].sum()
    agg = agg.rename(columns={"probability": "claim_prob_sum"})
    agg["matched_claims"] = matched_claims
    return agg


def summarize_event(ia_df: pd.DataFrame, event: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create ZIP-level detail and summary correlation metrics for an event."""
    detail_rows = []
    summary_rows = []

    def safe_corr(x: pd.Series, y: pd.Series, method: str | None = None) -> float:
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            return float("nan")
        return x.corr(y, method=method) if method else x.corr(y)

    event_df = ia_df[
        (ia_df["state"] == "NE") &
        (ia_df["disasterNumber"] == event["disaster_number"])
    ].copy()

    aggregate = event.get("aggregate", False)

    for county_key in event["counties"]:
        county_meta = COUNTIES[county_key]
        county_label = f"{county_meta['name']} (County)"
        ia_county = event_df[event_df["county"] == county_label].copy()
        if ia_county.empty:
            continue

        ia_zip = ia_county.groupby("zipCode", as_index=False).agg({
            "validRegistrations": "sum",
            "approvedForFemaAssistance": "sum",
            "totalApprovedIhpAmount": "sum",
            "repairReplaceAmount": "sum",
            "rentalAmount": "sum",
            "otherNeedsAmount": "sum",
        })
        ia_zip = ia_zip.rename(columns={"zipCode": "ZIP"})

        claim_zip = load_claim_zip_probs(event["base_dir"], county_key)
        if claim_zip is None:
            continue

        merged = claim_zip.merge(ia_zip, on="ZIP", how="inner")
        if merged.empty:
            continue
        merged["event"] = event["label"]
        merged["county"] = county_meta["name"]
        merged["claim_prob_weighted"] = merged["claim_prob_sum"] * merged["matched_claims"]
        detail_rows.append(merged)

        if aggregate:
            continue

        n_zip = len(merged)
        if n_zip < MIN_CORR_N:
            pearson_ihp_total = float("nan")
            spearman_ihp_total = float("nan")
            pearson_approved_count = float("nan")
            spearman_approved_count = float("nan")
        else:
            pearson_ihp_total = safe_corr(
                merged["claim_prob_sum"], merged["totalApprovedIhpAmount"]
            )
            spearman_ihp_total = safe_corr(
                merged["claim_prob_sum"], merged["totalApprovedIhpAmount"], method="spearman"
            )
            pearson_approved_count = safe_corr(
                merged["claim_prob_sum"], merged["approvedForFemaAssistance"]
            )
            spearman_approved_count = safe_corr(
                merged["claim_prob_sum"], merged["approvedForFemaAssistance"], method="spearman"
            )

        summary_rows.append({
            "event": event["label"],
            "county": county_meta["name"],
            "n_zip": n_zip,
            "pearson_ihp_total": pearson_ihp_total,
            "spearman_ihp_total": spearman_ihp_total,
            "pearson_approved_count": pearson_approved_count,
            "spearman_approved_count": spearman_approved_count,
        })

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    if aggregate and not detail_df.empty:
        claim_col = "claim_prob_weighted" if "claim_prob_weighted" in detail_df.columns else "claim_prob_sum"
        agg = detail_df.groupby("ZIP", as_index=False).agg({
            claim_col: "sum",
            "validRegistrations": "sum",
            "approvedForFemaAssistance": "sum",
            "totalApprovedIhpAmount": "sum",
            "repairReplaceAmount": "sum",
            "rentalAmount": "sum",
            "otherNeedsAmount": "sum",
        })
        n_zip = len(agg)
        if n_zip < MIN_CORR_N:
            pearson_ihp_total = float("nan")
            spearman_ihp_total = float("nan")
            pearson_approved_count = float("nan")
            spearman_approved_count = float("nan")
        else:
            pearson_ihp_total = safe_corr(
                agg[claim_col], agg["totalApprovedIhpAmount"]
            )
            spearman_ihp_total = safe_corr(
                agg[claim_col], agg["totalApprovedIhpAmount"], method="spearman"
            )
            pearson_approved_count = safe_corr(
                agg[claim_col], agg["approvedForFemaAssistance"]
            )
            spearman_approved_count = safe_corr(
                agg[claim_col], agg["approvedForFemaAssistance"], method="spearman"
            )
        summary_rows.append({
            "event": event["label"],
            "county": event.get("aggregate_label", "Pooled"),
            "n_zip": n_zip,
            "pearson_ihp_total": pearson_ihp_total,
            "spearman_ihp_total": spearman_ihp_total,
            "pearson_approved_count": pearson_approved_count,
            "spearman_approved_count": spearman_approved_count,
        })

    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def main() -> None:
    print("=" * 60)
    print("FEMA Housing Assistance Validation")
    print("=" * 60)

    if not IA_PARQUET.exists():
        raise FileNotFoundError(f"IA data not found: {IA_PARQUET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ia_df = load_ia_data(IA_PARQUET)

    all_detail = []
    all_summary = []
    for event in EVENTS:
        detail_df, summary_df = summarize_event(ia_df, event)
        if not detail_df.empty:
            all_detail.append(detail_df)
        if not summary_df.empty:
            all_summary.append(summary_df)

    if all_detail:
        detail = pd.concat(all_detail, ignore_index=True)
        detail.to_csv(OUT_DIR / "ia_zip_detail.csv", index=False)

    if all_summary:
        summary = pd.concat(all_summary, ignore_index=True)
        summary.to_csv(OUT_DIR / "ia_zip_summary.csv", index=False)

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
