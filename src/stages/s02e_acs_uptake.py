#!/usr/bin/env python3
"""
s02e_acs_uptake.py - ACS-based policy uptake context
====================================================

Fetches ACS 5-year ZCTA-level housing and income data and combines it
with NFIP policy counts to estimate policy uptake rates.

Outputs:
- data_work/acs_policy_uptake.csv
- data_work/acs_policy_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"

ACS_API = "https://api.census.gov/data/2019/acs/acs5"
ACS_VARS = {
    "B25001_001E": "housing_units",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    "B19013_001E": "median_income",
}

COUNTY_DIRS = {
    "dodge": DATA_WORK / "revision" / "dodge",
    "douglas": DATA_WORK / "revision" / "douglas",
}


def fetch_acs_zcta(zcta: str, state_fips: str) -> dict | None:
    params = {
        "get": "NAME," + ",".join(ACS_VARS.keys()),
        "for": f"zip code tabulation area:{zcta}",
        "in": f"state:{state_fips}",
    }
    resp = requests.get(ACS_API, params=params, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if len(data) < 2:
        return None
    row = data[1]
    out = {"ZCTA": zcta}
    for idx, key in enumerate(data[0]):
        if key in ACS_VARS:
            out[ACS_VARS[key]] = float(row[idx]) if row[idx] not in (None, "") else None
    return out


def main() -> None:
    print("=" * 60)
    print("ACS Policy Uptake Context")
    print("=" * 60)

    uptake_rows = []
    summary_rows = []
    all_zcta = set()

    # Collect ZCTA codes from policy files
    for county, base_dir in COUNTY_DIRS.items():
        policies_path = base_dir / "policies_prepared.csv"
        if not policies_path.exists():
            continue
        policies = pd.read_csv(policies_path, dtype={"ZIP": str, "ZCTA": str})
        zctas = policies["ZCTA"].dropna().astype(str).unique().tolist()
        all_zcta.update(zctas)

    # Fetch ACS data per ZCTA
    acs_rows = []
    for zcta in sorted(all_zcta):
        row = fetch_acs_zcta(zcta, "31")
        if row:
            acs_rows.append(row)

    acs_df = pd.DataFrame(acs_rows)
    if acs_df.empty:
        print("No ACS data returned.")
        return

    # Merge policy counts with ACS
    for county, base_dir in COUNTY_DIRS.items():
        policies_path = base_dir / "policies_prepared.csv"
        if not policies_path.exists():
            continue
        policies = pd.read_csv(policies_path, dtype={"ZIP": str, "ZCTA": str})
        pol_counts = policies.groupby("ZCTA", as_index=False).size()
        pol_counts = pol_counts.rename(columns={"size": "policy_count"})
        merged = pol_counts.merge(acs_df, on="ZCTA", how="left")
        merged["county"] = county
        merged["policy_uptake"] = merged["policy_count"] / merged["housing_units"]
        uptake_rows.append(merged)

        if merged["median_income"].notna().sum() > 1:
            corr = merged["policy_uptake"].corr(merged["median_income"])
        else:
            corr = None

        summary_rows.append({
            "county": county,
            "n_zcta": len(merged),
            "mean_uptake": merged["policy_uptake"].mean(),
            "median_uptake": merged["policy_uptake"].median(),
            "corr_uptake_income": corr,
        })

    uptake_df = pd.concat(uptake_rows, ignore_index=True) if uptake_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    if not uptake_df.empty:
        uptake_df.to_csv(DATA_WORK / "acs_policy_uptake.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(DATA_WORK / "acs_policy_summary.csv", index=False)

    print("Saved ACS uptake outputs to data_work/")


if __name__ == "__main__":
    main()
