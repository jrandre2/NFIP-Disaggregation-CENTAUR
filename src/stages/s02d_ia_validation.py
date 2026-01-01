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

COUNTIES = {
    "dodge": {"name": "Dodge", "fips": "053"},
    "douglas": {"name": "Douglas", "fips": "055"},
    "cass": {"name": "Cass", "fips": "025"},
    "dakota": {"name": "Dakota", "fips": "043"},
}

EVENTS = [
    {
        "label": "2019_midwest_flood",
        "disaster_number": 4420,
        "base_dir": DATA_WORK / "revision",
        "counties": ["dodge", "douglas"],
    },
    {
        "label": "2011_late_spring_storms",
        "disaster_number": 4013,
        "base_dir": DATA_WORK / "events_2011",
        "counties": ["cass", "dakota"],
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
    """Aggregate building-level claim probabilities to ZIPs."""
    claims_path = base_dir / county_key / "claim_probabilities.csv"
    buildings_path = base_dir / county_key / "buildings_prepared.csv"
    if not claims_path.exists() or not buildings_path.exists():
        return None

    probs = pd.read_csv(claims_path)
    buildings = pd.read_csv(buildings_path, dtype={"ZIP": str})
    merged = probs.merge(buildings[["BldgID", "ZIP"]], on="BldgID", how="left")
    merged = merged.dropna(subset=["ZIP"])
    agg = merged.groupby("ZIP", as_index=False)["probability"].sum()
    agg = agg.rename(columns={"probability": "claim_prob_sum"})
    return agg


def summarize_event(ia_df: pd.DataFrame, event: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create ZIP-level detail and summary correlation metrics for an event."""
    detail_rows = []
    summary_rows = []

    event_df = ia_df[
        (ia_df["state"] == "NE") &
        (ia_df["disasterNumber"] == event["disaster_number"])
    ].copy()

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
        merged["event"] = event["label"]
        merged["county"] = county_meta["name"]
        detail_rows.append(merged)

        if len(merged) < 2:
            continue

        summary_rows.append({
            "event": event["label"],
            "county": county_meta["name"],
            "n_zip": len(merged),
            "pearson_ihp_total": merged["claim_prob_sum"].corr(merged["totalApprovedIhpAmount"]),
            "spearman_ihp_total": merged["claim_prob_sum"].corr(merged["totalApprovedIhpAmount"], method="spearman"),
            "pearson_approved_count": merged["claim_prob_sum"].corr(merged["approvedForFemaAssistance"]),
            "spearman_approved_count": merged["claim_prob_sum"].corr(merged["approvedForFemaAssistance"], method="spearman"),
        })

    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
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
