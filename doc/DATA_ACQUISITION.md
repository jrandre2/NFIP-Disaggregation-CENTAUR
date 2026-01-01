# Data Acquisition Guide

## Summary

This document outlines how to obtain the data needed to recreate the NFIP claims disaggregation analysis.

---

## Data Available Locally

| Dataset | Path | Notes |
|---------|------|-------|
| **FEMA Claims** | `scripts/NEFloodMitigation/Data/NE_FEMA_Claims.csv` | 6,020 Nebraska claims; filter to Dodge County March 2019 = 261 residential |
| **Nebraska Building Footprints** | `/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Building_Footprints/Nebraska.geojson` | Microsoft Building Footprints, 303 MB |
| **Nebraska Statewide Parcels** | `/Volumes/T9/Projects/Freeze and Flight/statewide parcel/NE_2023_statewideparcels.gdb` | 20,980 parcels in Dodge County (County_ID = '053') |
| **USGS Elevation** | `/Volumes/T9/Projects/Freeze and Flight/GIS_Data/Elevation/USGS_3DEP_10m/` | Two tiles covering Dodge County |

---

## Data to Download

### 1. FEMA NFHL (National Flood Hazard Layer)

**Source**: FEMA Map Service Center
**URL**: https://msc.fema.gov/portal/advanceSearch

**Steps**:
1. Go to FEMA MSC Advanced Search
2. Search for "Dodge County, Nebraska" (FIPS: 31053)
3. Download the NFHL data package (includes S_FLD_HAZ_AR - flood hazard areas)
4. Extract to `/Volumes/T9/Projects/NFIP-Disaggregation-CENTAUR/data_raw/nfhl/`

**Alternative**: Nebraska GIS Data Portal may have statewide NFHL

**Fields Needed**:
- `FLD_ZONE` - Flood zone designation (A, AE, AO, X, etc.)
- `STATIC_BFE` - Base Flood Elevation (where available)
- `ZONE_SUBTY` - Zone subtype

### 2. March 2019 Inundation Extent

**Option A: Nebraska DNR**
- Check Nebraska Department of Natural Resources for official flood extent
- May have post-flood mapping from the March 2019 event

**Option B: Sentinel-2 Processing**
If official extent unavailable:
1. Download Sentinel-2 imagery from Copernicus Open Access Hub
2. Date: March 16, 2019 (first cloud-free after event)
3. Tile: T15TUF or T14TQL (covering Dodge County)
4. Process using NDWI/MNDWI to classify water

**Option C: USGS Flood Event Viewer**
- URL: https://stn.wim.usgs.gov/FEV/
- Search for 2019 Nebraska flooding event
- May have high water marks and flood extent data

---

## Data Processing Pipeline

After acquiring all data:

```bash
# 1. Copy raw data to project
cp -r [source] data_raw/

# 2. Run data preparation
python src/stages/s00_prepare_data.py

# 3. Verify data quality
python src/pipeline.py audit_data --full
```

---

## Alternative: Simplified Analysis

If NFHL unavailable, can run simplified analysis:
- Use flood zone from claims records (already present in OpenFEMA data)
- Skip building-to-flood-zone spatial join
- May slightly reduce accuracy but allows faster iteration

---

## Notes

- Paper used 264 claims; current OpenFEMA filter yields 261 (minor version difference)
- Parcel data lacks `year_built` field - may need county assessor API
- Building footprints need spatial join with parcels for value/attributes
