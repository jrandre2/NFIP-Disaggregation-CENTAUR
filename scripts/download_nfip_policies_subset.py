#!/usr/bin/env python3
"""
Download a filtered subset of NFIP policies from OpenFEMA.

Filters to policies in-force during a target window for specified counties.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import requests


API_BASE = "https://www.fema.gov/api/open/v2/FimaNfipPolicies"

SELECT_FIELDS = [
    "id",
    "reportedZipCode",
    "floodZoneCurrent",
    "baseFloodElevation",
    "buildingReplacementCost",
    "originalConstructionDate",
    "occupancyType",
    "policyEffectiveDate",
    "policyTerminationDate",
    "policyCount",
    "latitude",
    "longitude",
    "countyCode",
    "censusTract",
    "censusBlockGroupFips",
    "propertyState",
]


def build_filter(counties: Iterable[int], start_date: str, end_date: str,
                 occupancy_types: Iterable[int], state: str) -> str:
    counties_str = " or ".join([f"countyCode eq '{c}'" for c in counties])
    occ_str = " or ".join([f"occupancyType eq {o}" for o in occupancy_types])
    # In-force during window: effective <= end AND (termination >= start OR termination is null)
    return (
        f"({counties_str}) and "
        f"propertyState eq '{state}' and "
        f"policyEffectiveDate le {end_date} and "
        f"(policyTerminationDate ge {start_date} or policyTerminationDate eq null) and "
        f"({occ_str})"
    )


def fetch_policies(filter_expr: str, output_path: Path, batch_size: int = 1000) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skip = 0
    total = 0
    params = {
        "$select": ",".join(SELECT_FIELDS),
        "$top": batch_size,
        "$skip": skip,
        "$filter": filter_expr,
    }

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SELECT_FIELDS)
        writer.writeheader()

        while True:
            params["$skip"] = skip
            resp = requests.get(API_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("FimaNfipPolicies", [])

            if not data:
                break

            for row in data:
                writer.writerow({k: row.get(k) for k in SELECT_FIELDS})

            n = len(data)
            total += n
            skip += n
            print(f"Fetched {n} records (total {total})")

            if n < batch_size:
                break

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NFIP policies subset from OpenFEMA.")
    parser.add_argument("--counties", default="31053,31055",
                        help="Comma-separated county FIPS codes (default: 31053,31055).")
    parser.add_argument("--start-date", default="2019-03-01T00:00:00.000Z",
                        help="Window start ISO date (default: 2019-03-01T00:00:00.000Z).")
    parser.add_argument("--end-date", default="2019-03-31T23:59:59.000Z",
                        help="Window end ISO date (default: 2019-03-31T23:59:59.000Z).")
    parser.add_argument("--occupancy", default="1",
                        help="Comma-separated occupancyType codes (default: 1).")
    parser.add_argument("--state", default="NE", help="2-letter state code (default: NE).")
    parser.add_argument("--output", default="data_raw/nfip_policies_subset.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    counties = [int(c.strip()) for c in args.counties.split(",") if c.strip()]
    occupancy = [int(o.strip()) for o in args.occupancy.split(",") if o.strip()]

    filter_expr = build_filter(
        counties=counties,
        start_date=f"'{args.start_date}'",
        end_date=f"'{args.end_date}'",
        occupancy_types=occupancy,
        state=args.state,
    )

    output_path = Path(args.output)
    total = fetch_policies(filter_expr, output_path)
    print(f"Saved {total} policy records to {output_path}")


if __name__ == "__main__":
    main()
