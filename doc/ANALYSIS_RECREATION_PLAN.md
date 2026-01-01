# Analysis Recreation Plan

## Overview

Convert the ArcPy-based analysis pipeline to open-source Python GIS tools and recreate the dataset from publicly available sources.

## Current State

### Available Data
- **FEMA Claims CSV** (`NE_FEMA_Claims.csv`) - 6,021 Nebraska claims from OpenFEMA
- **Nebraska Statewide Parcels** (`NE_2023_statewideparcels.gdb`) - May be readable with GDAL/Fiona

### Data to Obtain
1. **Microsoft Building Footprints** - Available via GitHub/Planetary Computer
2. **FEMA NFHL Flood Zones** - Available from FEMA Map Service Center
3. **DEM/Elevation Data** - USGS 3DEP or Nebraska LiDAR
4. **Inundation Extent** - Sentinel-2 imagery from March 2019

---

## Conversion: ArcPy to Open-Source

### Tool Mapping

| ArcPy Function | Open-Source Replacement |
|----------------|------------------------|
| `arcpy.da.SearchCursor` | `geopandas.read_file()` + DataFrame iteration |
| `arcpy.analysis.Buffer` | `geopandas.GeoDataFrame.buffer()` |
| `arcpy.management.SelectLayerByLocation` | `geopandas.sjoin()` or `.intersects()` |
| `arcpy.management.SpatialJoin` | `geopandas.sjoin()` |
| Feature class I/O | `geopandas.read_file()` / `.to_file()` |
| Geodatabase tables | Parquet, GeoPackage, or PostGIS |

### Python Dependencies
```
geopandas>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
shapely>=2.0.0
pyproj>=3.6.0
rasterio>=1.3.0
fiona>=1.9.0
scikit-learn>=1.3.0  # For bootstrap metrics
scipy>=1.11.0
```

---

## Data Processing Pipeline

### Stage 1: Data Acquisition

#### 1.1 FEMA Claims (Dodge County, March 2019)
```python
# Filter NE_FEMA_Claims.csv to:
# - countyCode == 31053 (Dodge County)
# - dateOfLoss between 2019-03-01 and 2019-03-31
# - occupancyType == 1 (residential)
```

**Fields needed:**
- `reportedZipCode`
- `floodZoneCurrent`
- `baseFloodElevation`
- `buildingReplacementCost`
- `originalConstructionDate`
- `latitude`, `longitude` (rounded)
- `censusTract`, `censusBlockGroupFips`

#### 1.2 Microsoft Building Footprints
- Source: https://github.com/microsoft/USBuildingFootprints
- Nebraska file: `Nebraska.geojson.zip`
- Filter to Dodge, Washington, Douglas counties

#### 1.3 FEMA Flood Zones (NFHL)
- Source: FEMA Map Service Center or state data portal
- Need: S_FLD_HAZ_AR (flood hazard areas)
- Fields: FLD_ZONE, ZONE_SUBTY, STATIC_BFE

#### 1.4 Parcel Data
- Nebraska statewide parcels available locally
- Alternative: County assessor data
- Need: Parcel boundaries, assessed values, year built

#### 1.5 Elevation Data
- USGS 3DEP 1/3 arc-second (~10m) national DEM
- Or: Nebraska LiDAR-derived DEM (higher resolution)
- Extract building centroid elevations using `rasterstats`

#### 1.6 Inundation Validation Layer
- Sentinel-2 imagery: March 16, 2019 (first cloud-free after event)
- Source: Copernicus Open Access Hub or Google Earth Engine
- Process: NDWI classification to extract flood extent

---

### Stage 2: Data Integration

#### 2.1 Building Preparation
```python
# 1. Load Microsoft building footprints
# 2. Spatial join with parcels to get:
#    - Assessed value (Total_Asse)
#    - Year built (BuildYear)
#    - Parcel ID
# 3. Spatial join with NFHL to get:
#    - Flood zone
#    - Base flood elevation (where available)
# 4. Extract ZIP code (spatial join with ZCTA boundaries)
# 5. Extract elevation at centroid from DEM
# 6. Filter to largest building per parcel
# 7. Filter to residential parcels only
```

#### 2.2 Claims Preparation
```python
# 1. Filter to Dodge County, March 2019
# 2. Filter to residential (occupancyType == 1)
# 3. Parse dates for construction year
# 4. Create grouping key: (ZIP, FloodZone)
```

---

### Stage 3: Bootstrap Disaggregation

Port the core algorithm from `Bootstrap_Parameter_Testing_Pipeline.py`:

```python
def find_matching_buildings(claim, buildings_df, config):
    """Find candidate buildings for a claim."""
    candidates = buildings_df.copy()

    # Filter 1: ZIP code match
    candidates = candidates[candidates['ZIP'] == claim['ZIP']]

    # Filter 2: Flood zone match (if enabled)
    if config['use_flood_zone']:
        candidates = candidates[candidates['FloodZone'] == claim['FZ']]

    # Filter 3: Elevation tolerance (if BFE available)
    if config['elev_tolerance'] and claim['BFE'] is not None:
        candidates = candidates[
            abs(candidates['ELEVATION'] - claim['BFE']) <= config['elev_tolerance']
        ]

    return candidates

def run_bootstrap(claims, buildings, n_iter=1000):
    """Run bootstrap assignment."""
    rng = np.random.default_rng(seed=42)
    building_counts = defaultdict(int)

    for claim in claims.itertuples():
        matches = find_matching_buildings(claim, buildings, config)
        if len(matches) > 0:
            draws = rng.choice(matches['BldgID'].values, n_iter, replace=True)
            for bldg_id in draws:
                building_counts[bldg_id] += 1

    return building_counts
```

---

### Stage 4: Validation Metrics

Calculate the same metrics as the original analysis:

```python
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss
)

def calculate_metrics(y_true, y_scores):
    """Calculate discrimination and calibration metrics."""
    return {
        'roc_auc': roc_auc_score(y_true, y_scores),
        'pr_auc': auc(*precision_recall_curve(y_true, y_scores)[:2][::-1]),
        'brier_score': brier_score_loss(y_true, y_scores),
        'log_loss': log_loss(y_true, y_scores, labels=[0, 1])
    }
```

---

## Directory Structure

```
data_raw/
├── claims/
│   └── NE_FEMA_Claims.csv
├── buildings/
│   └── Nebraska_buildings.geojson
├── parcels/
│   └── dodge_county_parcels.gpkg
├── flood_zones/
│   └── NFHL_Dodge.gpkg
├── elevation/
│   └── dodge_county_dem.tif
└── inundation/
    └── 2019_march_flood_extent.gpkg

data_work/
├── buildings_prepared.parquet
├── claims_prepared.parquet
├── bootstrap_results.parquet
└── validation_metrics.csv
```

---

## Execution Plan

1. **Week 1: Data Acquisition**
   - Download Microsoft Building Footprints for Nebraska
   - Download FEMA NFHL data
   - Obtain DEM data
   - Process Sentinel-2 imagery for inundation extent

2. **Week 2: Data Integration**
   - Create building dataset with all attributes
   - Prepare claims dataset
   - Validate data quality

3. **Week 3: Algorithm Implementation**
   - Port bootstrap algorithm to GeoPandas
   - Implement parameter sweep
   - Validate against paper's reported metrics

4. **Week 4: Documentation & Validation**
   - Compare results to original paper
   - Document any discrepancies
   - Prepare reproducibility package

---

## Notes

- The original analysis used 264 residential claims from March 2019 Dodge County flooding
- ROC-AUC target: ~0.95 for best configuration (ZIP+FZ, largest building)
- Brier Score target: ~0.07
