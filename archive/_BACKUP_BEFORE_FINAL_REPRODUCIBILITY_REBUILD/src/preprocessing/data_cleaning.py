import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

raw_dir = Path("01_Raw_Datasets")
processed_dir = Path("02_Processed_Data")
reports_dir = Path("08_Reports")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Logs and Reports content
dq_records = []
unit_log = ["# Unit Conversion Log\n"]
spatial_log = ["# Spatial Alignment Report\n"]
temporal_log = ["# Temporal Alignment Report\n"]
cleaning_log = ["# Data Cleaning Report\n"]

def get_dq(df, name, stage="Before"):
    for col in df.columns:
        dq_records.append({
            "Dataset": name,
            "Stage": stage,
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Rows": len(df),
            "Missing": df[col].isnull().sum(),
            "Missing_Pct": df[col].isnull().sum() / len(df) * 100,
            "Unique": df[col].nunique(),
            "Min": df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
            "Max": df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
        })

# --- 1. Load and Clean Historical Climate Datasets (ERA5 & CHIRPS) ---
climate_files = {
    "rainfall": raw_dir / "01_Flood/India_CHIRPS_Daily_Rainfall_2000_2018.csv",
    "runoff": raw_dir / "01_Flood/India_ERA5_Runoff_2000_2018.csv",
    "soil_moisture": raw_dir / "01_Flood/India_ERA5_Soil_Moisture_2000_2018.csv",
    "humidity": raw_dir / "03_Common_Climate_Data/India_ERA5_Humidity_2000_2018.csv",
    "pressure": raw_dir / "03_Common_Climate_Data/India_ERA5_Surface_Pressure_2000_2018.csv",
    "temperature": raw_dir / "03_Common_Climate_Data/India_ERA5_Temperature_2000_2018.csv",
    "evaporation": raw_dir / "03_Common_Climate_Data/India_ERA5_Total_Evaporation_2000_2018.csv",
    "wind": raw_dir / "03_Common_Climate_Data/India_ERA5_Wind_2000_2018.csv"
}

dfs_climate = {}
for name, fpath in climate_files.items():
    df = pd.read_csv(fpath)
    get_dq(df, f"Historical_{name}", "Before")
    
    # Date Standardization
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Irrelevant columns
    cols_to_drop = []
    if '.geo' in df.columns:
        # Check if it's constant
        if df['.geo'].nunique() <= 1:
            cols_to_drop.append('.geo')
            cleaning_log.append(f"- Dropped `.geo` from {name} as it is constant/empty MultiPoint.\n")
    if 'system:index' in df.columns:
        cols_to_drop.append('system:index')
        cleaning_log.append(f"- Dropped `system:index` from {name} as it's redundant to date.\n")
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    dfs_climate[name] = df

# Temporal Alignment of Historical Data
date_ranges = {}
common_dates = None
for name, df in dfs_climate.items():
    dates = set(df['date'].dt.date)
    date_ranges[name] = dates
    if common_dates is None:
        common_dates = dates
    else:
        common_dates = common_dates.intersection(dates)

temporal_log.append(f"Historical common dates count: {len(common_dates)}\n")
expected_days = (pd.to_datetime("2018-12-31") - pd.to_datetime("2000-01-01")).days + 1
temporal_log.append(f"Expected daily records (2000-2018): {expected_days}\n")
for name, dates in date_ranges.items():
    temporal_log.append(f"- {name}: {len(dates)} dates (Missing: {expected_days - len(dates)})\n")

# Merge Historical Data
df_historical = None
for name, df in dfs_climate.items():
    if df_historical is None:
        df_historical = df
    else:
        df_historical = pd.merge(df_historical, df, on='date', how='outer')

# Unit conversions & Suspicious values
unit_log.append("Historical Data:\n")
# Check if rainfall needs conversion or is already mm. (CHIRPS is usually mm, but let's document it)
unit_log.append("- CHIRPS Rainfall: Assumed mm based on typical metadata. Checked min/max.\n")
# Check evaporation (sometimes negative in ERA5 to denote flux upwards)
if 'total_evaporation_m' in df_historical.columns:
    if df_historical['total_evaporation_m'].mean() < 0:
        df_historical['total_evaporation_m'] = df_historical['total_evaporation_m'].abs()
        unit_log.append("- Converted `total_evaporation_m` to absolute values (ERA5 uses negative for upward flux).\n")

# Spatial Alignment Assumption
spatial_log.append("Historical ERA5 & CHIRPS Data:\n")
spatial_log.append("- Historical datasets lack explicit Lat/Lon coordinates per row, implying they have been pre-aggregated (e.g., country-level average for India) via Earth Engine.\n")
spatial_log.append("- Will assume these are national aggregates for India unless specific grid metadata is provided.\n")

get_dq(df_historical, "Historical_Merged", "After")
df_historical.to_csv(processed_dir / "cleaned_climate_data.csv", index=False)


# --- 2. Load and Clean NASA POWER (Recent) ---
nasa_multi = pd.read_csv(raw_dir / "05_Recent_Data/NASA_POWER_India_Recent_2019_2025.csv")
get_dq(nasa_multi, "NASA_Multi", "Before")

# NASA Missing Values (-999)
nasa_multi.replace(-999, np.nan, inplace=True)
cleaning_log.append("- Replaced `-999` with `NaN` in NASA_POWER_India_Recent_2019_2025.csv.\n")

nasa_multi['date'] = pd.to_datetime(nasa_multi['date'])

# NASA Single Point
def load_nasa_point(filepath):
    skiprows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if "-END HEADER-" in line:
                skiprows = i + 1
                break
    df = pd.read_csv(filepath, skiprows=skiprows)
    return df

nasa_point = load_nasa_point(raw_dir / "05_Recent_Data/POWER_Point_Daily_20190101_20251231_016d99N_081d23E_UTC.csv")
get_dq(nasa_point, "NASA_Point", "Before")
nasa_point.replace(-999, np.nan, inplace=True)
cleaning_log.append("- Replaced `-999` with `NaN` in POWER_Point_Daily.\n")

# Date Parsing (DOY -> Date)
if 'YEAR' in nasa_point.columns and 'DOY' in nasa_point.columns:
    # Convert YEAR and DOY to datetime safely accounting for leap years
    nasa_point['date'] = pd.to_datetime(nasa_point['YEAR'].astype(str) + nasa_point['DOY'].astype(str), format='%Y%j')
    cleaning_log.append("- Constructed standard calendar `date` from `YEAR` and `DOY` in NASA Point dataset.\n")
    nasa_point.drop(columns=['YEAR', 'DOY'], inplace=True)

# Spatial alignment for NASA
spatial_log.append("\nNASA POWER 2019-2025:\n")
locs = nasa_multi[['state', 'city', 'latitude', 'longitude']].drop_duplicates()
spatial_log.append(f"- Multi-location dataset contains {len(locs)} distinct locations.\n")
for _, row in locs.iterrows():
    spatial_log.append(f"  * {row['state']} - {row['city']} ({row['latitude']}, {row['longitude']})\n")

# Merge recent data or keep separate? The prompt says "cleaned_recent_data.csv".
# Let's save the multi-location as the primary recent data.
get_dq(nasa_multi, "NASA_Multi", "After")
get_dq(nasa_point, "NASA_Point", "After")
nasa_multi.to_csv(processed_dir / "cleaned_recent_data.csv", index=False)


# --- 3. Flood Events (GFD) ---
flood_gfd = pd.read_csv(raw_dir / "01_Flood/India_Flood_Events_GFD_2000_2018.csv")
get_dq(flood_gfd, "Flood_Events_GFD", "Before")

# Filter for India
def is_india(row):
    try:
        if 'India' in str(row['primary_country']): return True
        if 'India' in str(row['country']): return True
        if 'INDIA' in str(row['gfd_country_name']): return True
        return False
    except:
        return False

mask = flood_gfd.apply(is_india, axis=1)
flood_filtered = flood_gfd[mask].copy()
cleaning_log.append(f"- Filtered GFD events for India (Retained {len(flood_filtered)} out of {len(flood_gfd)} events).\n")

# Extract Dates from system:index
# format: DFO_1641_From_20000918_to_20001021
import re
def extract_dates(idx):
    # Regex to find From_YYYYMMDD_to_YYYYMMDD
    match = re.search(r'From_(\d{8})_to_(\d{8})', str(idx))
    if match:
        start = pd.to_datetime(match.group(1), format='%Y%m%d')
        end = pd.to_datetime(match.group(2), format='%Y%m%d')
        return start, end
    return pd.NaT, pd.NaT

dates = flood_filtered['system:index'].apply(extract_dates)
flood_filtered['start_date'] = dates.apply(lambda x: x[0])
flood_filtered['end_date'] = dates.apply(lambda x: x[1])

cleaning_log.append("- Extracted `start_date` and `end_date` from `system:index` for GFD events.\n")

# Clean Event Table
event_cols = [
    'event_id', 'start_date', 'end_date', 'latitude', 'longitude', 
    'severity', 'deaths', 'displaced_people', 'main_cause', 
    'primary_country', 'validation_type'
]
available_cols = [c for c in event_cols if c in flood_filtered.columns]
flood_clean = flood_filtered[available_cols]

get_dq(flood_clean, "Flood_Events_Cleaned", "After")
flood_clean.to_csv(processed_dir / "cleaned_flood_events.csv", index=False)

# Save Reports
pd.DataFrame(dq_records).to_csv(processed_dir / "data_quality_report.csv", index=False)

with open(reports_dir / "DATA_CLEANING_REPORT.md", "w") as f:
    f.writelines(cleaning_log)
with open(reports_dir / "SPATIAL_ALIGNMENT_REPORT.md", "w") as f:
    f.writelines(spatial_log)
with open(reports_dir / "TEMPORAL_ALIGNMENT_REPORT.md", "w") as f:
    f.writelines(temporal_log)
with open(reports_dir / "UNIT_CONVERSION_LOG.md", "w") as f:
    f.writelines(unit_log)

print("Data Cleaning Phase Complete.")
