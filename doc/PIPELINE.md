# Pipeline Documentation

**Related**: [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | [METHODOLOGY.md](METHODOLOGY.md)
**Status**: Active
**Last Updated**: 2025-12-31

---

> **Note:** For quick command syntax reference, see [skills.md](skills.md).

---

# NFIP Claims Disaggregation Pipeline

This project uses a custom pipeline for NFIP claims disaggregation analysis. The stages are located in `src/stages/`.

## Quick Start (NFIP Analysis)

```bash
source .venv/bin/activate

# Data Preparation
python src/stages/s00_prepare_nfip.py      # Prepare claims and buildings
python src/stages/s00b_download_nfhl.py    # Download NFHL flood zones (optional)

# Core Analysis
python src/stages/s01_bootstrap_disagg.py   # Bootstrap disaggregation (baseline)
python src/stages/s02_parameter_sweep.py    # Table 1 recreation (4 configs)

# Extended Analysis
python src/stages/s02b_sensitivity_analysis.py  # Sensitivity tests
python src/stages/s02c_claim_diagnostics.py     # Per-claim filter diagnostics

# Outputs
python src/stages/s03_figures.py            # Generate publication figures
python src/stages/s04_robustness.py         # Robustness checks

# Manuscript
cd manuscript_quarto && ./render_all.sh --profile risa
```

---

## NFIP Pipeline Overview

```
data_raw/                    ──► s00_prepare_nfip.py ──► data_work/claims_prepared.csv
  claims_raw.csv                                        data_work/buildings_prepared.gpkg
  buildings_raw.gpkg                                    data_work/inundation.gpkg
  inundation_raw.gpkg
                                      │
                                      ▼
                             s01_bootstrap_disagg.py ──► data_work/bootstrap_results.csv
                                      │                  data_work/diagnostics/validation_metrics.csv
                                      ▼
                             s02_parameter_sweep.py ──► data_work/parameter_sweep_results.csv
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
        s02b_sensitivity.py   s02c_diagnostics.py   s04_robustness.py
                    │                 │                 │
                    ▼                 ▼                 ▼
        data_work/sensitivity/   data_work/diagnostics/   data_work/robustness/
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                               s03_figures.py ──► manuscript_quarto/figures/*.png
```

---

## NFIP Stage Details

### Stage s00: Data Preparation

**Script:** `src/stages/s00_prepare_nfip.py`

**Purpose:** Prepare claims, buildings, and inundation data for disaggregation analysis.

**Operations:**
1. Load and filter NFIP claims (Dodge County, March 2019 flood)
2. Load Microsoft building footprints
3. Join buildings with parcel data (assessed values, ZIP codes)
4. Join buildings with NFHL flood zones
5. Extract ground elevations from USGS 3DEP 10m DEM
6. Prepare inundation extent polygon

**Inputs:**
| File | Description |
|------|-------------|
| `data_raw/nfip_claims.csv` | Raw NFIP claims from OpenFEMA |
| Nebraska building footprints | Microsoft Building Footprints GeoJSON |
| NE statewide parcels GDB | Parcel boundaries with assessed values |
| USGS 3DEP 10m DEM | Digital elevation model |

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/claims_prepared.csv` | 261 filtered claims |
| `data_work/buildings_prepared.gpkg` | 34,350 buildings with attributes |
| `data_work/buildings_all.gpkg` | All buildings (before largest filter) |
| `data_work/inundation.gpkg` | Flood inundation extent |

---

### Stage s00b: NFHL Download (Optional)

**Script:** `src/stages/s00b_download_nfhl.py`

**Purpose:** Download NFHL flood zone data from FEMA's ArcGIS REST API.

**API Endpoint:** `https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer`

**Outputs:**
| File | Description |
|------|-------------|
| `data_raw/nfhl/S_FLD_HAZ_AR.shp` | Flood hazard areas shapefile |

---

### Stage s01: Bootstrap Disaggregation

**Script:** `src/stages/s01_bootstrap_disagg.py`

**Purpose:** Run core bootstrap disaggregation algorithm with baseline configuration.

**Configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_ITERATIONS` | 1,000 | Bootstrap iterations |
| `USE_FLOOD_ZONE` | True | Match on flood zone |
| `LARGEST_ONLY` | True | Keep largest building per parcel |
| `SEED` | 42 | Random seed for reproducibility |

**Algorithm:**
1. For each claim, find candidate buildings (ZIP + flood zone match)
2. Draw N random buildings from candidate set (with replacement)
3. Aggregate draws across all claims to compute probabilities
4. Validate against inundation ground truth

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/bootstrap_results.csv` | Building probabilities |
| `data_work/diagnostics/validation_metrics.csv` | ROC-AUC, PR-AUC, Brier |

**Key Metrics (Baseline):**
- ROC-AUC: 0.969
- PR-AUC: 0.874
- Brier Score: 0.221

---

### Stage s02: Parameter Sweep

**Script:** `src/stages/s02_parameter_sweep.py`

**Purpose:** Recreate Table 1 by testing all 4 parameter configurations.

**Configurations:**
| Config | Spatial Match | Building Filter |
|--------|---------------|-----------------|
| 1 | ZIP only | All buildings |
| 2 | ZIP only | Largest per parcel |
| 3 | ZIP + FZ | All buildings |
| 4 | ZIP + FZ | Largest per parcel (baseline) |

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/parameter_sweep_results.csv` | Metrics for all 4 configs |

---

### Stage s02b: Sensitivity Analysis

**Script:** `src/stages/s02b_sensitivity_analysis.py`

**Purpose:** Test sensitivity to key parameters.

**Tests Performed:**
| Test | Parameter Range | Purpose |
|------|-----------------|---------|
| Iteration convergence | N = [100, 500, 1000, 5000, 10000] | Verify N=1000 sufficient |
| Elevation tolerance | [0.25, 0.50, 1.00, 2.00] ft | Test elevation filter |
| Inundation buffer | [0, 50, 100, 250] ft | Test ground truth definition |

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/sensitivity/iteration_convergence.csv` | N sensitivity results |
| `data_work/sensitivity/elevation_tolerance.csv` | Elevation filter results |
| `data_work/sensitivity/buffer_sensitivity.csv` | Buffer test results |

---

### Stage s02c: Claim Diagnostics

**Script:** `src/stages/s02c_claim_diagnostics.py`

**Purpose:** Generate per-claim filter chain diagnostics.

**Metrics per Claim:**
| Metric | Description |
|--------|-------------|
| `n_initial` | Candidates after ZIP/FZ match |
| `n_after_elev` | Candidates after elevation filter |
| `n_final` | Final candidate pool size |
| `n_unique_sel` | Unique buildings selected in bootstrap |
| `var_sel` | Variance in selection counts |
| `top_share` | Share captured by most-selected building |

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/diagnostics/filter_chain.csv` | Per-claim diagnostics |
| `data_work/diagnostics/summary_by_fz.csv` | Summary by flood zone |
| `data_work/diagnostics/summary_by_zip.csv` | Summary by ZIP code |

---

### Stage s03: Figure Generation

**Script:** `src/stages/s03_figures.py`

**Purpose:** Generate publication-quality figures for manuscript.

**Figures Generated:**
| Figure | Description |
|--------|-------------|
| `fig_roc_curves.png` | ROC curves for all 4 configurations |
| `fig_pr_curves.png` | Precision-Recall curves |
| `fig_calibration.png` | Calibration plot (baseline model) |
| `fig_metric_comparison.png` | Bar chart comparing metrics |

**Outputs:**
| Directory | Description |
|-----------|-------------|
| `manuscript_quarto/figures/` | PNG and PDF figures |

---

### Stage s04: Robustness Checks

**Script:** `src/stages/s04_robustness.py`

**Purpose:** Comprehensive robustness testing.

**Tests Performed:**
| Test | Description |
|------|-------------|
| Seed stability | Metrics across 10 random seeds |
| Jackknife | Leave-one-claim-out influence analysis |
| Spatial CV | Leave-ZIP-out cross-validation |
| Bootstrap CI | 95% confidence intervals from resampling |

**Outputs:**
| File | Description |
|------|-------------|
| `data_work/robustness/seed_stability.csv` | Seed variation results |
| `data_work/robustness/jackknife_results.csv` | Claim influence scores |
| `data_work/robustness/spatial_cv_results.csv` | Leave-ZIP-out results |
| `data_work/robustness/bootstrap_ci.csv` | Confidence intervals |

---

## Review Management Commands

```bash
# Check revision status
python src/pipeline.py review_status

# Verify revision completeness
python src/pipeline.py review_verify

# Generate response letter
python src/pipeline.py review_response

# Archive completed review cycle
python src/pipeline.py review_archive
```

---

## Manuscript Rendering (Risk Analysis)

```bash
cd manuscript_quarto

# Render with Risk Analysis profile
./render_all.sh --profile risa

# Preview
../tools/bin/quarto preview
```

---

## Pipeline Overview

```
data_raw/ ──► ingest_data ──► data_work/data_raw.parquet
                           │
                           ▼
                link_records ──► data_work/data_linked.parquet
                           │           └─► data_work/diagnostics/linkage_summary.csv
                           ▼
                build_panel ──► data_work/panel.parquet
                           │           └─► data_work/diagnostics/panel_summary.csv
                           ▼
             run_estimation ──► data_work/diagnostics/estimation_results.csv
                           │           └─► data_work/diagnostics/coefficients.csv
                           ▼
          estimate_robustness ──► data_work/diagnostics/robustness_results.csv
                           │           └─► data_work/diagnostics/placebo_results.csv
                           ▼
              make_figures ──► manuscript_quarto/figures/*.png ──► render_all.sh ──► manuscript_quarto/_output/
```

---

## Stage Details

### Stage 00: Data Ingestion

**Command:** `python src/pipeline.py ingest_data`

**Purpose:** Load and preprocess raw data files. Use `--demo` flag to generate synthetic data if `data_raw/` is empty.

**Input:** `data_raw/`
**Output:** `data_work/data_raw.parquet`

**Implementation:** `src/stages/s00_ingest.py`

---

### Stage 01: Record Linkage

**Command:** `python src/pipeline.py link_records`

**Purpose:** Link records across multiple data sources.

**Input:** `data_work/data_raw.parquet`
**Output:** `data_work/data_linked.parquet`
**Diagnostics:** `data_work/diagnostics/linkage_summary.csv`

**Implementation:** `src/stages/s01_link.py`

---

### Stage 02: Panel Construction

**Command:** `python src/pipeline.py build_panel`

**Purpose:** Create the analysis panel from linked data.

**Input:** `data_work/data_linked.parquet`
**Output:** `data_work/panel.parquet`
**Diagnostics:** `data_work/diagnostics/panel_summary.csv`

**Implementation:** `src/stages/s02_panel.py`

---

### Stage 03: Primary Estimation

**Command:** `python src/pipeline.py run_estimation [options]`

**Options:**
- `--specification, -s`: Specification name (default: baseline)
- `--sample`: Sample restriction (default: full)

**Purpose:** Run primary estimation specifications.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/estimation_results.csv`, `data_work/diagnostics/coefficients.csv`

**Implementation:** `src/stages/s03_estimation.py`

---

### Stage 04: Robustness Checks

**Command:** `python src/pipeline.py estimate_robustness`

**Purpose:** Run robustness specifications and sensitivity analyses.

**Input:** `data_work/panel.parquet`
**Output:** `data_work/diagnostics/robustness_results.csv`, `data_work/diagnostics/placebo_results.csv`

**Implementation:** `src/stages/s04_robustness.py`

**Extended Robustness Tests (when applicable):**

The robustness stage includes additional tests for geographic and ML-based analyses:

| Test | Description | When to Use |
|------|-------------|-------------|
| Spatial vs Random CV | Compare spatial and random cross-validation to quantify geographic data leakage | Data with latitude/longitude coordinates |
| Feature Ablation | Test model performance with feature subsets | Multiple feature groups |
| Tuned Models | Nested CV with hyperparameter tuning for Ridge, ElasticNet, RF, GB | ML prediction tasks |
| Encoding Comparisons | Compare categorical vs ordinal treatment encoding | Categorical treatment variables |

**Usage with spatial CV:**

```python
from stages.s04_robustness import run_spatial_cv_comparison, run_feature_ablation

# Compare spatial vs random CV
results = run_spatial_cv_comparison(
    df, feature_cols=['feature_1', 'feature_2'],
    lat_col='latitude', lon_col='longitude'
)

# Feature ablation study
results = run_feature_ablation(df, feature_cols)
```

See [Spatial Cross-Validation](#spatial-cross-validation) section below for methodology details.

---

### Stage 05: Figure Generation

**Command:** `python src/pipeline.py make_figures`

**Purpose:** Generate publication-quality figures.

**Input:** `data_work/panel.parquet`, `data_work/diagnostics/*.csv`
**Output:** `manuscript_quarto/figures/*.png`

**Implementation:** `src/stages/s05_figures.py`

If you need a top-level export, copy from `manuscript_quarto/figures/` to `figures/`.

---

### Stage 06: Manuscript Validation

**Command:** `python src/pipeline.py validate_submission [options]`

**Options:**
- `--journal, -j`: Target journal (default: jeem)
- `--report`: Generate markdown report

**Purpose:** Validate manuscript against journal requirements.

**Output (with `--report`):** `data_work/diagnostics/submission_validation.md`

**Implementation:** `src/stages/s06_manuscript.py`

---

### Stage 07: Review Management

**Commands:**

- `python src/pipeline.py review_status [-m manuscript]`
- `python src/pipeline.py review_new [-m manuscript] [-f focus]`
- `python src/pipeline.py review_new --actual [-j journal] [-r round]`
- `python src/pipeline.py review_verify [-m manuscript]`
- `python src/pipeline.py review_archive [-m manuscript] [--no-tag]`
- `python src/pipeline.py review_diff [-m manuscript] [--from N] [--to M]`
- `python src/pipeline.py review_response [-m manuscript]`
- `python src/pipeline.py review_report`

**Options:**

- `--manuscript, -m`: Target manuscript (default: main)
- `--focus, -f`: Review focus area (economics, engineering, social_sciences, general, methods, policy, clarity)
- `--actual`: Mark as actual journal review (vs synthetic)
- `--journal, -j`: Journal name for actual reviews
- `--round, -r`: Submission round (initial, R&R1, R&R2)
- `--decision`: Decision received (major_revision, minor_revision, reject, accept)
- `--reviewers`: Reviewer IDs (R1 R2 R3)
- `--no-tag`: Skip git tagging on archive
- `--tag`: Custom git tag name

**Purpose:** Manage both synthetic (AI-generated) and actual (journal) peer review cycles. Supports multi-manuscript projects with git integration for change tracking.

**Features:**

- **Synthetic reviews**: AI-generated reviews for pre-submission stress-testing
- **Actual reviews**: Track journal reviews with metadata (journal, round, decision)
- **Git integration**: Automatic commit tracking and tagging on archive
- **Visual diffs**: Generate diffs between review cycles
- **Response letters**: Auto-generate "Response to Reviewers" documents

**Outputs:**

- `manuscript_quarto/REVISION_TRACKER.md`
- `doc/reviews/archive/` (archived reviews)
- Git tags: `review-{manuscript}-{cycle:02d}-{status}`

**Implementation:** `src/stages/s07_reviews.py`, `src/stages/_review_models.py`

---

### Stage 08: Journal Configuration Tools

**Commands:**
- `python src/pipeline.py journal_list`
- `python src/pipeline.py journal_validate --config natural_hazards`
- `python src/pipeline.py journal_compare --journal natural_hazards`
- `python src/pipeline.py journal_parse --input guidelines.txt --output new_journal.yml`
- `python src/pipeline.py journal_parse --url https://example.com/guidelines --journal "Journal Name" --output journal.yml --save-raw`
- `python src/pipeline.py journal_fetch --url https://example.com/guidelines --journal "Journal Name" --text`

**Purpose:** List, validate, compare, and parse journal requirements.

**Outputs:**
- `manuscript_quarto/journal_configs/<name>.yml` (for `journal_parse`)
- `doc/journal_guidelines/*` (when using `journal_fetch` or `journal_parse --save-raw`)

**Notes:**
- PDF guidelines must be converted to text or HTML before parsing.
- Parsing uses heuristic extraction; manual review is required.

**Implementation:** `src/stages/s08_journal_parser.py`

---

## Stage 09: AI-Assisted Writing

**Purpose:** Generate draft manuscript sections from pipeline outputs using LLMs.

**Module:** `src/stages/s09_writing.py`

### Commands

| Command          | Description                                  |
| ---------------- | -------------------------------------------- |
| `draft_results`  | Draft results section from estimation tables |
| `draft_captions` | Generate figure captions                     |
| `draft_abstract` | Synthesize abstract from manuscript          |

### Usage

```bash
# Draft results section from estimation table
python src/pipeline.py draft_results --table main_results
python src/pipeline.py draft_results --table main_results --section primary

# Preview prompt without API call
python src/pipeline.py draft_results --table main_results --dry-run

# Generate figure captions
python src/pipeline.py draft_captions --figure "fig_*.png"

# Synthesize abstract with word limit
python src/pipeline.py draft_abstract --max-words 200

# Use alternative provider
python src/pipeline.py draft_results --table main_results --provider openai
```

### Options

| Option             | Description                                   |
| ------------------ | --------------------------------------------- |
| `--table, -t`      | Diagnostic CSV name (without .csv extension)  |
| `--section, -s`    | Section name for output file (default: main)  |
| `--figure, -f`     | Figure glob pattern (e.g., "fig_*.png")       |
| `--manuscript, -m` | Target manuscript (default: main)             |
| `--max-words`      | Target word limit for abstract (default: 250) |
| `--dry-run`        | Show prompt without making API call           |
| `--provider, -p`   | LLM provider: anthropic or openai             |

### Configuration

LLM settings in `src/config.py`:

```python
LLM_PROVIDER = 'anthropic'  # or 'openai'
LLM_MODELS = {
    'anthropic': 'claude-sonnet-4-20250514',
    'openai': 'gpt-4-turbo-preview',
}
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4096
```

**Environment Variables:**

- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENAI_API_KEY`: Required for OpenAI provider

### Output

Drafts are saved to `manuscript_quarto/drafts/` with metadata headers:

```markdown
<!-- AI-Generated Draft
     Source: data_work/diagnostics/main_results.csv
     Provider: anthropic/claude-sonnet-4-20250514
     Generated: 2025-12-27 14:30:22
     Status: REQUIRES HUMAN REVIEW
-->
```

All drafts require human review before integration into the manuscript.

**Implementation:** `src/stages/s09_writing.py`, `src/llm/`

---

## Versioned Stages

Stages can evolve over time using version suffixes. This allows keeping alternative implementations while maintaining a clear evolution history.

**Naming Convention:** `s00_ingest` → `s00b_standardize` → `s00c_enhanced`

**Commands:**

```bash
# List all available stages
python src/pipeline.py list_stages

# List versions of a specific stage
python src/pipeline.py list_stages -p s00

# Run a specific stage version
python src/pipeline.py run_stage s00b_standardize
```

**Benefits:**

- Preserve alternative implementations for comparison
- Track methodological evolution
- Switch between versions for robustness checks

---

## Caching and Parallel Execution

The pipeline includes intelligent caching and parallel execution to dramatically improve performance during iterative development.

### Performance Impact

| Stage                      | Without Cache | With Cache | Improvement |
| -------------------------- | ------------- | ---------- | ----------- |
| s03_estimation (4 specs)   | ~3 sec        | <100ms     | ~30x        |
| s04_robustness (10 tests)  | ~8 sec        | <300ms     | ~25x        |
| s05_figures (5 plots)      | ~4 sec        | <500ms     | ~8x         |

### How Caching Works

The cache automatically tracks:

1. **Data hash** - MD5 hash of input DataFrames
2. **Configuration hash** - Hash of specification parameters
3. **File dependencies** - Content hash of any file inputs

When you re-run a stage, cached results are used if:

- Input data is unchanged
- Configuration is unchanged
- All file dependencies are unchanged

Cache files are stored in `data_work/.cache/<stage_name>/`.

### CLI Flags

```bash
# Disable caching (force recomputation)
python src/pipeline.py run_estimation --no-cache
python src/pipeline.py estimate_robustness --no-cache
python src/pipeline.py make_figures --no-cache

# Disable parallel execution
python src/pipeline.py run_estimation --sequential
python src/pipeline.py estimate_robustness --sequential
python src/pipeline.py make_figures --sequential

# Control parallel workers (default: CPU count)
python src/pipeline.py run_estimation --workers 4
python src/pipeline.py estimate_robustness -w 2
```

### Cache Management

```bash
# View cache statistics
python src/pipeline.py cache stats

# Clear all cached data
python src/pipeline.py cache clear

# Clear a specific stage's cache
python src/pipeline.py cache clear --stage s03_estimation
```

### When Cache Invalidates

Caches automatically invalidate when:

- **Data changes**: Any modification to input parquet files
- **Config changes**: Different specification parameters
- **Code changes**: Not automatically tracked (use `cache clear` after code changes)

### Parallel Execution

Parallel execution runs independent computations simultaneously:

- **s03_estimation**: Multiple specifications in parallel
- **s04_robustness**: Robustness tests in parallel by category
- **s05_figures**: Multiple figures in parallel

Uses `ProcessPoolExecutor` for CPU-bound work (estimation) and `ThreadPoolExecutor` for I/O-bound work (figures).

### Configuration

Global settings in `src/config.py`:

```python
CACHE_ENABLED = True           # Enable/disable caching globally
CACHE_MAX_AGE_HOURS = 168      # Cache TTL (1 week default)
PARALLEL_ENABLED = True        # Enable/disable parallel execution
PARALLEL_MAX_WORKERS = None    # None = CPU count
```

### Best Practices

1. **During development**: Leave caching enabled for fast iteration
2. **After code changes**: Clear cache with `python src/pipeline.py cache clear`
3. **For final runs**: Use `--no-cache` to ensure clean computation
4. **Low memory systems**: Use `--sequential` to limit memory usage

---

## QA Reports

Each pipeline stage automatically generates quality assurance reports.

**Output Location:** `data_work/quality/`

**File Pattern:** `{stage_name}_quality_{timestamp}.csv`

**Example:**

```text
data_work/quality/s00_ingest_quality_20251227_143022.csv
data_work/quality/s01_link_quality_20251227_143025.csv
data_work/quality/s02_panel_quality_20251227_143028.csv
```

**Metrics Tracked:**

- Row and column counts
- Missing value percentages
- Duplicate row counts
- Memory usage
- Stage-specific metrics (e.g., linkage rates, estimation sample sizes)

**Configuration:** QA reports are controlled by `ENABLE_QA_REPORTS` in `src/config.py`.

---

## Manuscript Rendering

### Render All Formats

```bash
cd manuscript_quarto
./render_all.sh
```

Output in `manuscript_quarto/_output/`:
- HTML files
- PDF
- DOCX

PDF/DOCX filenames follow the `book.title` in `manuscript_quarto/_quarto.yml`.

### Journal-Specific Rendering

```bash
./render_all.sh --profile jeem   # JEEM format
./render_all.sh --profile aer    # AER format
```

### Live Preview

```bash
cd manuscript_quarto
../tools/bin/quarto preview
```

---

## Data Audit

```bash
python src/pipeline.py audit_data
python src/pipeline.py audit_data --full
python src/pipeline.py audit_data --full --report
```

`audit_data` prints a summary to the console and optionally writes a markdown report to `data_work/diagnostics/`.

---

## Common Workflows

### Full Rebuild

```bash
source .venv/bin/activate
# Add --demo to ingest_data if using synthetic data
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures
cd manuscript_quarto && ./render_all.sh
```

### Update After Data Change

```bash
python src/pipeline.py ingest_data
python src/pipeline.py link_records
python src/pipeline.py build_panel
python src/pipeline.py run_estimation
python src/pipeline.py estimate_robustness
python src/pipeline.py make_figures
```

### Update Figures Only

```bash
python src/pipeline.py make_figures
cd manuscript_quarto && ./render_all.sh
```

---

## Spatial Cross-Validation

When working with geographic data, standard k-fold cross-validation can produce overly optimistic performance estimates due to spatial autocorrelation. The spatial CV module provides tools to address this.

### Why Spatial CV?

Geographic observations that are close to each other tend to be similar (spatial autocorrelation). Standard CV randomly assigns observations to folds, which can:

- Put nearby observations in both training and test sets
- Allow information to "leak" from training to test
- Produce inflated performance metrics

Spatial CV ensures geographic separation between training and test sets.

### Spatial CV Usage

```python
from src.utils.spatial_cv import SpatialCVManager, compare_spatial_vs_random_cv

# Create spatial groups
manager = SpatialCVManager(n_groups=5, method='kmeans')
groups = manager.create_groups_from_coordinates(df['latitude'], df['longitude'])

# Cross-validate with spatial groups
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
results = manager.cross_validate(model, X, y)
print(f"Spatial CV R2: {results['mean']:.3f} +/- {results['std']:.3f}")

# Quantify leakage
comparison = manager.compare_to_random_cv(model, X, y)
print(f"Random CV:  {comparison['random_cv']['mean']:.3f}")
print(f"Spatial CV: {comparison['spatial_cv']['mean']:.3f}")
print(f"Leakage:    {comparison['leakage']:.3f}")
```

### Grouping Methods

| Method | Description | Requirements |
| ------ | ----------- | ------------ |
| `kmeans` | K-means clustering on coordinates | lat/lon |
| `balanced_kmeans` | K-means with balanced group sizes | lat/lon |
| `geographic_bands` | Latitude-based horizontal bands | lat/lon |
| `longitude_bands` | Longitude-based vertical bands | lat/lon |
| `spatial_blocks` | Grid-based spatial blocks | lat/lon |
| `zip_digit` | ZIP code digit-based grouping | zip codes |
| `contiguity_queen` | Polygon contiguity (shared edges/vertices) | geopandas |
| `contiguity_rook` | Polygon contiguity (shared edges only) | geopandas |

### Spatial CV Configuration

Settings in `src/config.py`:

```python
SPATIAL_CV_N_GROUPS = 5          # Number of spatial folds
SPATIAL_GROUPING_METHOD = 'kmeans'  # Default method
SPATIAL_SENSITIVITY_METHODS = [   # Methods for sensitivity analysis
    'kmeans', 'balanced_kmeans', 'geographic_bands',
    'longitude_bands', 'spatial_blocks'
]
```

### Optional Dependencies

Contiguity-based methods require geopandas:

```bash
pip install -r requirements-spatial.txt
```

---

## Minimal Demo

See `demo/README.md` for a small sample dataset and expected outputs that exercise the full data → manuscript path.
