#!/usr/bin/env python3
"""
Generate the naive spatial-aggregation baselines reported in the manuscript
(Table @tbl-naive, "Naive Model Results: Dodge County, March 2019").

Each structure in a coarse spatial unit receives the same claim probability
P_b = C_g / N_g, where C_g is the number of claims in unit g and N_g is the
number of buildings in unit g. Two unit definitions (ZCTA, and ZCTA intersected
with FEMA flood zone) are each applied to (a) all mapped buildings and (b) only
the largest building per parcel. Discrimination/calibration are scored against
the event inundation footprint.

The naive model is fully deterministic (no bootstrap, no seed). Inputs come from
the prepared revision datasets produced by prepare_revision_data.py:
    data_work/revision/dodge/{buildings_all.gpkg, buildings_prepared.gpkg,
                              claims_prepared.csv, inundation.gpkg}

Usage:
    python scripts/run_naive_baselines.py            # dodge (manuscript table)
    python scripts/run_naive_baselines.py --county douglas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss,
)

# Reuse the exact helpers used to build the bootstrap tables, so the naive
# baselines are scored on a consistent flood-zone normalization and ground truth.
from run_revision_experiments import normalize_flood_zone, get_inundated_ids

PROJECT_ROOT = Path(__file__).parent.parent
DATA_WORK = PROJECT_ROOT / "data_work"

# Manuscript Table @tbl-naive values (Dodge), for the printed comparison.
MANUSCRIPT_DODGE = {
    ("ZCTA", "All Buildings"): (0.812, 0.491, 0.096, 0.627),
    ("ZCTA", "Largest Building"): (0.829, 0.501, 0.056, 0.454),
    ("ZCTA + FZ", "All Buildings"): (0.857, 0.733, 0.090, 0.994),
    ("ZCTA + FZ", "Largest Building"): (0.868, 0.722, 0.049, 0.542),
}


def unit_key(row: pd.Series, use_flood_zone: bool):
    """Spatial-unit key for a building or claim row."""
    zcta = row.get("ZCTA")
    if pd.isna(zcta):
        return None
    zcta = str(zcta)
    if not use_flood_zone:
        return zcta
    fz = normalize_flood_zone(row.get("FloodZone"))
    if fz is None or (isinstance(fz, float) and pd.isna(fz)):
        # Buildings/claims with no mapped NFHL polygon are outside the SFHA and
        # are treated as Zone X (minimal hazard), rather than dropped. Dropping
        # them would shrink the X units, inflate P_b = C_g/N_g, and overstate the
        # naive Brier score.
        fz = "X"
    return (zcta, fz)


def naive_scores(buildings: pd.DataFrame, claims: pd.DataFrame,
                 use_flood_zone: bool) -> pd.DataFrame:
    """Assign P_b = C_g / N_g to every building in the universe."""
    b_keys = buildings.apply(lambda r: unit_key(r, use_flood_zone), axis=1)
    c_keys = claims.apply(lambda r: unit_key(r, use_flood_zone), axis=1)

    claims_per_unit = c_keys.value_counts().to_dict()           # C_g
    bldgs_per_unit = b_keys.value_counts().to_dict()            # N_g

    out = buildings[["BldgID"]].copy()
    out["unit"] = b_keys
    out["probability"] = [
        (claims_per_unit.get(k, 0) / bldgs_per_unit[k]) if (k is not None and bldgs_per_unit.get(k)) else 0.0
        for k in b_keys
    ]
    # Buildings with no unit key (e.g. no flood zone under the +FZ definition)
    # are not part of that universe.
    out = out[b_keys.notna().values].reset_index(drop=True)
    return out


def metrics(scored: pd.DataFrame, inundated_ids: set) -> dict:
    y_true = scored["BldgID"].isin(inundated_ids).astype(int).to_numpy()
    y_score = scored["probability"].to_numpy(dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {}
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    y_clip = np.clip(y_score, 0.0, 1.0)  # P_b is a rate; clip for proper-score metrics
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": auc(recall, precision),
        "brier_score": brier_score_loss(y_true, y_clip),
        "log_loss": log_loss(y_true, np.clip(y_clip, 1e-15, 1 - 1e-15)),
        "n_buildings": int(len(y_true)),
        "n_inundated": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }


def run(county: str) -> pd.DataFrame:
    base = DATA_WORK / "revision" / county
    claims = pd.read_csv(base / "claims_prepared.csv", dtype={"ZIP": str, "ZCTA": str, "CBG": str})
    all_gdf = gpd.read_file(base / "buildings_all.gpkg")
    largest_gdf = gpd.read_file(base / "buildings_prepared.gpkg")
    inund = gpd.read_file(base / "inundation.gpkg")

    universes = {"All Buildings": all_gdf, "Largest Building": largest_gdf}
    constraints = {"ZCTA": False, "ZCTA + FZ": True}

    rows = []
    for sel_label, gdf in universes.items():
        inundated_ids = get_inundated_ids(gdf, inund)
        bdf = gdf.drop(columns=["geometry"])
        for con_label, use_fz in constraints.items():
            scored = naive_scores(bdf, claims, use_fz)
            m = metrics(scored, inundated_ids)
            rows.append({"spatial_constraints": con_label,
                         "building_selection": sel_label, **m})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate naive spatial-aggregation baselines.")
    ap.add_argument("--county", default="dodge")
    args = ap.parse_args()

    df = run(args.county)
    out_path = DATA_WORK / "revision" / args.county / "metrics_naive.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}\n")

    order = ["roc_auc", "pr_auc", "brier_score", "log_loss"]
    print(f"{'Constraint':<12}{'Selection':<18}{'ROC':>7}{'PR':>7}{'Brier':>8}{'LogLoss':>9}   vs manuscript")
    for _, r in df.iterrows():
        key = (r["spatial_constraints"], r["building_selection"])
        got = tuple(round(r[k], 3) for k in order)
        ref = MANUSCRIPT_DODGE.get(key)
        flag = ""
        if args.county == "dodge" and ref:
            # ROC/PR/Brier are the discrimination/calibration metrics; Log-Loss
            # depends on the near-zero clipping convention, so report it separately.
            core = max(abs(g - e) for g, e in zip(got[:3], ref[:3]))
            ll_delta = abs(got[3] - ref[3])
            flag = "  ROC/PR/Brier MATCH" if core <= 0.005 else f"  DIFF ref={ref[:3]}"
            flag += f" (LogLoss {got[3]:.3f} vs {ref[3]:.3f})"
        print(f"{r['spatial_constraints']:<12}{r['building_selection']:<18}"
              f"{got[0]:>7.3f}{got[1]:>7.3f}{got[2]:>8.3f}{got[3]:>9.3f}{flag}")


if __name__ == "__main__":
    main()
