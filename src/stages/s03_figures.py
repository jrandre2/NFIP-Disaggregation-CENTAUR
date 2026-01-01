#!/usr/bin/env python3
"""
s03_figures.py - Generate Publication Figures (Revision)
========================================================

Generates updated figures for the NFIP Claims Disaggregation manuscript:
1. ROC curves for spatial unit variants (ZIP, ZCTA, CBG)
2. Precision-Recall curves for spatial unit variants
3. Calibration plots for the ZCTA+FZ baseline
4. Metric comparison bars (ROC/PR/Brier) by county
5. Maps of building-level likelihoods with inundation overlay

Output: manuscript_quarto/figures/
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import os

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_WORK = PROJECT_ROOT / "data_work"
REVISION_DIR = DATA_WORK / "revision"
FIGURES_DIR = PROJECT_ROOT / "manuscript_quarto" / "figures"

_PROJ_CANDIDATES = [
    Path("/opt/anaconda3/pkgs/proj-9.4.1-hfb94cee_0/share/proj"),
    Path("/opt/anaconda3/pkgs/proj-9.4.0-h52fb9d0_0/share/proj"),
    Path("/opt/anaconda3/pkgs/proj-9.3.1-h805f6d4_0/share/proj"),
    Path("/opt/homebrew/Cellar/proj/9.7.1/share/proj"),
]
_PROJ_DATA_DIR = None
for _candidate in _PROJ_CANDIDATES:
    if _candidate.exists():
        _PROJ_DATA_DIR = _candidate
        os.environ["PROJ_DATA"] = str(_candidate)
        os.environ["PROJ_LIB"] = str(_candidate)
        break

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import pyproj
from pyproj import datadir as _pyproj_datadir

if _PROJ_DATA_DIR is not None:
    _pyproj_datadir.set_data_dir(str(_PROJ_DATA_DIR))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from shapely.geometry import box
from shapely.geometry import Polygon

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

# Counties
COUNTIES = {
    "dodge": {
        "label": "Dodge County, NE",
        "fips": "053",
    },
    "douglas": {
        "label": "Douglas County, NE",
        "fips": "055",
    },
}

# Configurations
CONFIGS = [
    {
        "name": "zip_fz",
        "label": "ZIP + FZ",
        "key_field": "ZIP",
        "use_flood_zone": True,
        "color": "#1f77b4",
        "linestyle": "--",
    },
    {
        "name": "zcta_fz",
        "label": "ZCTA + FZ",
        "key_field": "ZCTA",
        "use_flood_zone": True,
        "color": "#d62728",
        "linestyle": "-",
    },
    {
        "name": "cbg_fz",
        "label": "CBG + FZ",
        "key_field": "CBG",
        "use_flood_zone": True,
        "color": "#2ca02c",
        "linestyle": "-.",
    },
]

BASELINE_NAME = "zcta_fz"
N_ITERATIONS = 1000
SEED = 42

HEX_RADIUS_M = 1000
ZOOM_BUFFER_M = 3000

COUNTY_BOUNDARIES = DATA_RAW / "cbg" / "tl_2020_31_bg.shp"


def normalize_flood_zone(fz: str | None) -> str | None:
    if fz is None or pd.isna(fz):
        return None
    fz = str(fz).upper()
    if fz.startswith("A"):
        return fz.rstrip("0123456789")
    if fz == "B":
        return "X"
    return fz


def load_county_data(county_key: str) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load claims, buildings, and inundation for a county."""
    base_dir = REVISION_DIR / county_key
    claims = pd.read_csv(
        base_dir / "claims_prepared.csv",
        dtype={"ZIP": str, "ZCTA": str, "CBG": str},
    )
    buildings = gpd.read_file(base_dir / "buildings_prepared.gpkg")
    inundation = gpd.read_file(base_dir / "inundation.gpkg")
    return claims, buildings, inundation


def build_indices(buildings: gpd.GeoDataFrame, key_field: str, use_flood_zone: bool) -> Dict:
    idx = {}
    for _, row in buildings.iterrows():
        key = row.get(key_field)
        if pd.isna(key):
            continue
        key = str(key)
        if use_flood_zone:
            fz = normalize_flood_zone(row.get("FloodZone"))
            if fz is None:
                continue
            idx.setdefault((key, fz), []).append(int(row["BldgID"]))
        else:
            idx.setdefault(key, []).append(int(row["BldgID"]))
    return idx


def get_candidates(record: pd.Series, idx: Dict, key_field: str, use_flood_zone: bool) -> List[int]:
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


def run_bootstrap(claims: pd.DataFrame,
                  buildings: gpd.GeoDataFrame,
                  config: Dict) -> Tuple[pd.DataFrame, int]:
    """Run bootstrap and return building probabilities and draw counts."""
    idx = build_indices(buildings, config["key_field"], config["use_flood_zone"])
    rng = np.random.default_rng(seed=SEED)

    building_counts: Dict[int, int] = defaultdict(int)
    matched = 0

    for _, claim in claims.iterrows():
        candidates = get_candidates(claim, idx, config["key_field"], config["use_flood_zone"])
        if not candidates:
            continue
        matched += 1
        draws = rng.choice(candidates, N_ITERATIONS, replace=True)
        for bid in draws:
            building_counts[bid] += 1

    total_draws = matched * N_ITERATIONS
    rows = []
    for bid, count in building_counts.items():
        rows.append(
            {
                "BldgID": bid,
                "count": int(count),
                "probability": count / total_draws if total_draws else 0.0,
            }
        )
    return pd.DataFrame(rows), matched


def get_ground_truth(buildings: gpd.GeoDataFrame, inundation: gpd.GeoDataFrame) -> set:
    if buildings.crs != inundation.crs:
        buildings = buildings.to_crs(inundation.crs)
    buildings = buildings.copy()
    buildings["rep_point"] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[["BldgID"]],
        geometry=buildings["rep_point"],
        crs=buildings.crs,
    )
    inundated = gpd.sjoin(points, inundation[["geometry"]], how="inner", predicate="within")
    return set(inundated["BldgID"].unique())


def build_score_arrays(results_df: pd.DataFrame, inundated_ids: set) -> Tuple[np.ndarray, np.ndarray]:
    y_true = results_df["BldgID"].apply(lambda x: 1 if x in inundated_ids else 0).values
    y_scores = results_df["probability"].values
    return y_true, y_scores


def compute_results() -> Tuple[Dict, Dict, Dict]:
    """Run bootstrap for all counties and configs."""
    results_by_county: Dict[str, Dict[str, pd.DataFrame]] = {}
    gt_by_county: Dict[str, set] = {}
    buildings_by_county: Dict[str, gpd.GeoDataFrame] = {}
    inund_by_county: Dict[str, gpd.GeoDataFrame] = {}

    for county_key in COUNTIES:
        claims, buildings, inund = load_county_data(county_key)
        results_by_county[county_key] = {}
        gt_by_county[county_key] = get_ground_truth(buildings, inund)
        buildings_by_county[county_key] = buildings
        inund_by_county[county_key] = inund

        for config in CONFIGS:
            results_df, matched = run_bootstrap(claims, buildings, config)
            results_by_county[county_key][config["name"]] = results_df
            print(
                f"{county_key} {config['label']}: matched {matched} | "
                f"buildings scored {len(results_df)}"
            )

    return results_by_county, gt_by_county, buildings_by_county, inund_by_county


def plot_roc_curves(results_by_county: Dict, gt_by_county: Dict) -> None:
    print("Generating ROC curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, (county_key, county_info) in zip(axes, COUNTIES.items()):
        for config in CONFIGS:
            results_df = results_by_county[county_key].get(config["name"])
            if results_df is None or results_df.empty:
                continue
            y_true, y_scores = build_score_arrays(results_df, gt_by_county[county_key])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            ax.plot(
                fpr,
                tpr,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=f"{config['label']} (AUC = {roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_title(county_info["label"], fontsize=12)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("ROC Curves by Spatial Unit", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_roc_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_roc_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(results_by_county: Dict, gt_by_county: Dict) -> None:
    print("Generating PR curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, (county_key, county_info) in zip(axes, COUNTIES.items()):
        for config in CONFIGS:
            results_df = results_by_county[county_key].get(config["name"])
            if results_df is None or results_df.empty:
                continue
            y_true, y_scores = build_score_arrays(results_df, gt_by_county[county_key])
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            ax.plot(
                recall,
                precision,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=f"{config['label']} (AUC = {pr_auc:.3f})",
            )

        baseline_df = results_by_county[county_key].get(BASELINE_NAME)
        if baseline_df is not None and not baseline_df.empty:
            y_true, _ = build_score_arrays(baseline_df, gt_by_county[county_key])
            prevalence = y_true.mean()
            ax.axhline(
                y=prevalence,
                color="gray",
                linestyle=":",
                linewidth=1,
                label=f"Prevalence = {prevalence:.3f}",
            )

        ax.set_title(county_info["label"], fontsize=12)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Precision-Recall Curves by Spatial Unit", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_pr_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_pr_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results_by_county: Dict, gt_by_county: Dict) -> None:
    print("Generating calibration plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Determine global x-axis limits for log scale
    min_scores = []
    max_scores = []
    for county_key in COUNTIES:
        results_df = results_by_county[county_key].get(BASELINE_NAME)
        if results_df is None or results_df.empty:
            continue
        scores = results_df["probability"].values
        scores = scores[scores > 0]
        if scores.size == 0:
            continue
        min_scores.append(scores.min())
        max_scores.append(scores.max())

    if not min_scores or not max_scores:
        print("  No scores available for calibration plot.")
        plt.close(fig)
        return

    x_min = max(min(min_scores), 1e-6) * 0.8
    x_max = 1.0

    for ax, (county_key, county_info) in zip(axes, COUNTIES.items()):
        results_df = results_by_county[county_key].get(BASELINE_NAME)
        if results_df is None or results_df.empty:
            continue
        y_true, y_scores = build_score_arrays(results_df, gt_by_county[county_key])
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="quantile")
        ax.plot(
            prob_pred,
            prob_true,
            "s-",
            color="#d62728",
            linewidth=2,
            markersize=6,
            label="ZCTA + FZ baseline",
        )

        xline = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        ax.plot(xline, xline, "k--", linewidth=1.2, label="Perfect calibration", zorder=1)
        ax.set_title(county_info["label"], fontsize=12)
        ax.set_xlabel("Mean Predicted Probability (log scale)")
        ax.set_ylabel("Observed Frequency")
        ax.set_xscale("log")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0, 1])
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Calibration Plots (ZCTA + FZ baseline)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison() -> None:
    print("Generating metric comparison...")
    rows = []
    for county_key in COUNTIES:
        metrics_path = REVISION_DIR / county_key / "metrics_claims_variants.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df = df[df["variant"].isin(["baseline_zip_fz", "zcta_fz", "cbg_fz"])].copy()
        df["county"] = county_key
        rows.append(df)

    if not rows:
        print("  Metrics files not found, skipping metric comparison.")
        return

    df = pd.concat(rows, ignore_index=True)
    variant_map = {
        "baseline_zip_fz": "ZIP + FZ",
        "zcta_fz": "ZCTA + FZ",
        "cbg_fz": "CBG + FZ",
    }
    df["variant_label"] = df["variant"].map(variant_map)

    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier_score", "Brier"),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)

    for row_idx, (county_key, county_info) in enumerate(COUNTIES.items()):
        sub = df[df["county"] == county_key]
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            vals = sub.set_index("variant_label")[metric]
            ax.bar(range(len(vals)), vals.values, color=colors)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(vals.index, fontsize=8)
            ax.set_ylim([0, 1])
            if col_idx == 0:
                ax.set_ylabel(county_info["label"], fontsize=10)
            if row_idx == 0:
                ax.set_title(title, fontsize=10)

    fig.suptitle("Validation Metrics by Spatial Unit", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_metric_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_metric_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _log_tick_values(vmin: float, vmax: float) -> List[float]:
    ticks = []
    exp_min = int(np.floor(np.log10(vmin)))
    exp_max = int(np.ceil(np.log10(vmax)))
    for exp in range(exp_min, exp_max + 1):
        ticks.append(10 ** exp)
    ticks = [t for t in ticks if vmin <= t <= vmax]
    if ticks and ticks[0] != vmin:
        ticks = [vmin] + ticks
    if ticks and ticks[-1] != vmax:
        ticks.append(vmax)
    return ticks


def load_county_outlines() -> Dict[str, gpd.GeoDataFrame]:
    if not COUNTY_BOUNDARIES.exists():
        raise FileNotFoundError(f"County boundaries not found: {COUNTY_BOUNDARIES}")
    gdf = gpd.read_file(COUNTY_BOUNDARIES)
    gdf = gdf[gdf["STATEFP"] == "31"].copy()
    outlines = {}
    for county_key, info in COUNTIES.items():
        county = gdf[gdf["COUNTYFP"] == info["fips"]].dissolve(by="COUNTYFP")
        outlines[county_key] = county
    return outlines


def build_hex_grid(bounds: Tuple[float, float, float, float],
                   hex_radius: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    width = np.sqrt(3) * hex_radius
    height = 2 * hex_radius
    dx = width
    dy = 0.75 * height

    rows = np.arange(miny - height, maxy + height, dy)
    cols = np.arange(minx - width, maxx + width, dx)

    hexes = []
    hex_id = 0
    angles = np.deg2rad(np.arange(0, 360, 60) + 30)
    for row_idx, y in enumerate(rows):
        x_offset = width / 2 if row_idx % 2 else 0
        for x in cols + x_offset:
            coords = [(x + hex_radius * np.cos(a), y + hex_radius * np.sin(a)) for a in angles]
            hexes.append({"hex_id": hex_id, "geometry": Polygon(coords)})
            hex_id += 1

    return gpd.GeoDataFrame(hexes, crs="EPSG:3857")


def aggregate_counts_to_hex(buildings_gdf: gpd.GeoDataFrame,
                            hex_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = buildings_gdf.copy()
    points["geometry"] = points.geometry.representative_point()
    joined = gpd.sjoin(points, hex_grid, how="inner", predicate="within")
    agg = joined.groupby("hex_id")["count"].sum()
    hex_grid = hex_grid.join(agg, on="hex_id")
    hex_grid["count"] = hex_grid["count"].fillna(0)
    return hex_grid


def _add_basemap(ax: plt.Axes, crs: str) -> None:
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            crs=crs,
            attribution=False,
            zorder=0,
        )
    except Exception as exc:
        print(f"  Basemap load failed: {exc}")


def _plot_inundation(ax: plt.Axes, inund: gpd.GeoDataFrame) -> None:
    if inund.empty:
        return
    inund.plot(ax=ax, color="#9ecae1", alpha=0.2, edgecolor="none", zorder=1)


def _plot_county_outline(ax: plt.Axes, outline: gpd.GeoDataFrame) -> None:
    if outline.empty:
        return
    outline.boundary.plot(ax=ax, color="#1f1f1f", linewidth=1, zorder=4)


def _plot_colorbar(fig: plt.Figure,
                   ax: plt.Axes,
                   norm,
                   label: str,
                   ticks: List[float],
                   tick_labels: List[str] | None = None) -> None:
    sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
    if tick_labels is not None:
        cbar.ax.set_yticklabels(tick_labels)
    else:
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{int(round(x)):,}")
        )
    cbar.set_label(label)


def plot_probability_maps(results_by_county: Dict,
                           buildings_by_county: Dict,
                           inund_by_county: Dict) -> None:
    print("Generating maps...")

    outlines = load_county_outlines()

    all_counts = []
    merged_by_county = {}
    for county_key in COUNTIES:
        buildings = buildings_by_county[county_key]
        results_df = results_by_county[county_key].get(BASELINE_NAME)
        if results_df is None or results_df.empty:
            continue
        merged = buildings.merge(results_df, on="BldgID", how="inner")
        merged_by_county[county_key] = merged
        all_counts.extend(merged["count"].values.tolist())

    if not all_counts:
        print("  No counts available for maps.")
        return

    vmin = max(min(all_counts), 1)
    vmax = max(all_counts)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ticks = _log_tick_values(vmin, vmax)
    hex_norm = Normalize(vmin=0, vmax=np.log1p(vmax))
    tick_counts = [0]
    if vmax > 0:
        tick_counts.extend(
            sorted({int(round(v)) for v in np.geomspace(1, vmax, num=4)})
        )
    if tick_counts[-1] != int(vmax):
        tick_counts.append(int(vmax))
    tick_counts = sorted(set(tick_counts))
    hex_ticks = [np.log1p(v) for v in tick_counts]
    hex_tick_labels = [f"{v:,}" for v in tick_counts]

    for county_key, county_info in COUNTIES.items():
        merged = merged_by_county.get(county_key)
        if merged is None or merged.empty:
            continue

        inund = inund_by_county[county_key]
        outline = outlines[county_key]

        merged = merged.to_crs("EPSG:3857")
        inund = inund.to_crs("EPSG:3857")
        outline = outline.to_crs("EPSG:3857")

        minx, miny, maxx, maxy = outline.total_bounds

        # Build hex grid and aggregate counts
        hex_grid = build_hex_grid((minx, miny, maxx, maxy), HEX_RADIUS_M)
        hex_grid = gpd.clip(hex_grid, outline)
        hex_grid = aggregate_counts_to_hex(merged, hex_grid)
        hex_grid["count_plot"] = np.log1p(hex_grid["count"])

        # County-level hex map
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        _add_basemap(ax, "EPSG:3857")
        _plot_inundation(ax, inund)
        if not hex_grid.empty:
            hex_grid.boundary.plot(
                ax=ax,
                color="#f0f0f0",
                linewidth=0.2,
                zorder=1.5,
            )
            hex_nonzero = hex_grid[hex_grid["count"] > 0].copy()
            hex_nonzero.plot(
                ax=ax,
                column="count_plot",
                cmap="magma",
                norm=hex_norm,
                linewidth=0,
                edgecolor="none",
                zorder=2,
            )
        _plot_county_outline(ax, outline)
        ax.set_title(f"{county_info['label']} (hex aggregation)", fontsize=12)
        ax.set_axis_off()
        ax.set_aspect("equal")
        _plot_colorbar(
            fig,
            ax,
            hex_norm,
            "Bootstrap draw count (hex sum)",
            hex_ticks,
            tick_labels=hex_tick_labels,
        )
        fig.suptitle(
            "County-Level Hex Bins of Claim Frequency (ZCTA + FZ baseline)", fontsize=13
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"fig_hex_{county_key}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            FIGURES_DIR / f"fig_hex_{county_key}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Zoomed-in building footprint map using highest-intensity hex
        if hex_grid.empty:
            continue
        top_hex = hex_grid.loc[hex_grid["count"].idxmax()]
        center = top_hex.geometry.centroid
        zoom_bounds = (
            center.x - ZOOM_BUFFER_M,
            center.y - ZOOM_BUFFER_M,
            center.x + ZOOM_BUFFER_M,
            center.y + ZOOM_BUFFER_M,
        )
        zoom_box = box(*zoom_bounds)

        buildings_zoom = merged[merged.intersects(zoom_box)].copy()
        inund_zoom = inund[inund.intersects(zoom_box)].copy()
        outline_zoom = outline[outline.intersects(zoom_box)].copy()

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        ax.set_xlim(zoom_bounds[0], zoom_bounds[2])
        ax.set_ylim(zoom_bounds[1], zoom_bounds[3])
        _add_basemap(ax, "EPSG:3857")
        _plot_inundation(ax, inund_zoom)
        if not buildings_zoom.empty:
            buildings_zoom.plot(
                ax=ax,
                column="count",
                cmap="magma",
                norm=norm,
                linewidth=0,
                edgecolor="none",
                zorder=2,
            )
        _plot_county_outline(ax, outline_zoom)
        ax.set_title(f"{county_info['label']} (zoomed)", fontsize=12)
        ax.set_axis_off()
        ax.set_aspect("equal")
        _plot_colorbar(fig, ax, norm, "Bootstrap draw count", ticks)
        fig.suptitle(
            "Zoomed Building Footprints in a High-Frequency Area", fontsize=13
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"fig_zoom_{county_key}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            FIGURES_DIR / f"fig_zoom_{county_key}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("NFIP Disaggregation - Figure Generation (Revision)")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_by_county, gt_by_county, buildings_by_county, inund_by_county = compute_results()

    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    plot_roc_curves(results_by_county, gt_by_county)
    plot_pr_curves(results_by_county, gt_by_county)
    plot_calibration(results_by_county, gt_by_county)
    plot_metric_comparison()
    plot_probability_maps(results_by_county, buildings_by_county, inund_by_county)

    print("\n" + "=" * 60)
    print("Figure generation complete")
    print("=" * 60)
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
