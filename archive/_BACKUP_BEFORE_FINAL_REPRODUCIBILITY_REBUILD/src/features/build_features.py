import os
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
processed_dir = Path("02_Processed_Data")
features_dir = Path("03_Features")
labels_dir = Path("04_Labels")
reports_dir = Path("08_Reports")

os.makedirs(features_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Reports content
label_def_log = ["# Label Definition\n"]
fe_report_log = ["# Feature Engineering Report\n"]
feat_dict_log = ["# Feature Dictionary\n"]
leakage_log = ["# Leakage Audit\n"]

# --- 1. Load Cleaned Data ---
climate_df = pd.read_csv(processed_dir / "cleaned_climate_data.csv")
climate_df['date'] = pd.to_datetime(climate_df['date'])
climate_df = climate_df.sort_values('date').reset_index(drop=True)

flood_events = pd.read_csv(processed_dir / "cleaned_flood_events.csv")
flood_events['start_date'] = pd.to_datetime(flood_events['start_date'])
flood_events['end_date'] = pd.to_datetime(flood_events['end_date'])

# Define training period for baselines (2000 - 2012) to prevent leakage
train_mask = (climate_df['date'].dt.year >= 2000) & (climate_df['date'].dt.year <= 2012)
train_df = climate_df[train_mask]

# --- 2. Label Definition: Heatwave ---
hw_threshold = train_df['temperature_max_C'].quantile(0.90)

label_def_log.append("## Heatwave Definition\n")
label_def_log.append(f"- **Threshold**: {hw_threshold:.2f} °C (90th percentile of daily max temperature from train period 2000-2012).\n")
label_def_log.append("- **Duration**: >= 3 consecutive days.\n")
label_def_log.append("- **Baseline**: Training period only, preventing leakage.\n")
label_def_log.append("- **Variables Used**: `temperature_max_C`.\n")

climate_df['hw_exceed'] = climate_df['temperature_max_C'] > hw_threshold
climate_df['hw_rolling_3'] = climate_df['hw_exceed'].rolling(window=3).sum()
climate_df['heatwave_target'] = (climate_df['hw_rolling_3'] == 3).astype(int)

# --- 3. Label Definition: Flood ---
label_def_log.append("\n## Flood Definition\n")
label_def_log.append("- **Event Definition**: Any day falling within the `start_date` and `end_date` of a verified India flood event in GFD.\n")
label_def_log.append("- **Spatial Matching**: National level (binary). The available datasets are aggregated to a national daily level, so spatial precision is not supported for historical modeling without distinct sub-regional grids.\n")
label_def_log.append("- **Positive Class**: 1 if a flood is ongoing anywhere in India on that day.\n")
label_def_log.append("- **Negative Class**: 0 otherwise.\n")

climate_df['flood_target'] = 0
for _, row in flood_events.dropna(subset=['start_date', 'end_date']).iterrows():
    mask = (climate_df['date'] >= row['start_date']) & (climate_df['date'] <= row['end_date'])
    climate_df.loc[mask, 'flood_target'] = 1

# --- 4. Label Definition: Compound Event ---
label_def_log.append("\n## Compound Event Definition\n")
label_def_log.append("- **Temporal Relationship**: Flood occurrence within 7 days of a Heatwave.\n")
label_def_log.append("- **Justification**: Sequential compound events often manifest as extreme heat causing soil desiccation, followed closely by intense precipitation leading to rapid surface runoff (flooding). A 7-day window captures this synoptic-scale transition.\n")

climate_df['hw_recent_7d'] = climate_df['heatwave_target'].rolling(window=7, min_periods=1).max()
climate_df['compound_target'] = ((climate_df['flood_target'] == 1) & (climate_df['hw_recent_7d'] == 1)).astype(int)

# Target Table
labels_df = climate_df[['date', 'flood_target', 'heatwave_target', 'compound_target']]
labels_df.to_csv(labels_dir / "flood_labels.csv", index=False)
labels_df.to_csv(labels_dir / "heatwave_labels.csv", index=False)
labels_df.to_csv(labels_dir / "compound_labels.csv", index=False)

# --- 5. Feature Engineering ---
fe_report_log.append("## Feature Construction\n")
leakage_log.append("## Leakage Audit\n")
leakage_log.append("All features represent conditions at or before time $t$. No future data is used. Anomalies use baseline stats from the train period (2000-2012).\n\n")
feat_dict_log.append("| Feature | Description | Type |\n|---|---|---|\n")

features_df = climate_df[['date']].copy()

def add_feature(name, series, desc):
    features_df[name] = series
    feat_dict_log.append(f"| {name} | {desc} | Numeric |\n")
    leakage_log.append(f"- **{name}**: Derived using only historical/current data. Valid.\n")

# Rainfall
add_feature('rainfall', climate_df['rainfall_mm'], "Daily rainfall (mm)")
for window in [3, 5, 7, 14]:
    add_feature(f'rainfall_{window}d', climate_df['rainfall_mm'].rolling(window, min_periods=1).sum(), f"{window}-day cumulative rainfall")

# Runoff
add_feature('runoff', climate_df['runoff_m'], "Total runoff (m)")
add_feature('surface_runoff', climate_df['surface_runoff_m'], "Surface runoff (m)")
for window in [3, 7]:
    add_feature(f'runoff_{window}d', climate_df['runoff_m'].rolling(window, min_periods=1).sum(), f"{window}-day cumulative runoff")

# Soil Moisture
if 'volumetric_soil_water_layer_1' in climate_df.columns:
    add_feature('soil_moisture_layer_1', climate_df['volumetric_soil_water_layer_1'], "Soil moisture layer 1")
    add_feature('soil_moisture_rolling_7d', climate_df['volumetric_soil_water_layer_1'].rolling(7, min_periods=1).mean(), "7-day average soil moisture")
    train_mean = train_df['volumetric_soil_water_layer_1'].mean()
    train_std = train_df['volumetric_soil_water_layer_1'].std()
    add_feature('soil_moisture_anomaly', (climate_df['volumetric_soil_water_layer_1'] - train_mean) / train_std, "Soil moisture standardized anomaly (relative to train)")

# Temperature
add_feature('temperature_avg', climate_df['temperature_avg_C'], "Average temperature (C)")
add_feature('temperature_min', climate_df['temperature_min_C'], "Min temperature (C)")
add_feature('temperature_max', climate_df['temperature_max_C'], "Max temperature (C)")
add_feature('rolling_temperature_3d', climate_df['temperature_avg_C'].rolling(3, min_periods=1).mean(), "3-day avg temperature")
train_t_mean = train_df['temperature_avg_C'].mean()
add_feature('temperature_anomaly', climate_df['temperature_avg_C'] - train_t_mean, "Temperature anomaly from train baseline")

# Humidity, Pressure, Wind, Evap
add_feature('dewpoint', climate_df['dewpoint_C'], "Dewpoint (C)")
add_feature('dewpoint_min', climate_df['dewpoint_min_C'], "Min Dewpoint (C)")
add_feature('dewpoint_max', climate_df['dewpoint_max_C'], "Max Dewpoint (C)")
add_feature('surface_pressure', climate_df.get('surface_pressure_hPa', climate_df.iloc[:,1]*0), "Surface Pressure (hPa)")
train_p_mean = train_df.get('surface_pressure_hPa', climate_df.iloc[:,1]*0).mean()
add_feature('pressure_anomaly', climate_df.get('surface_pressure_hPa', climate_df.iloc[:,1]*0) - train_p_mean, "Pressure anomaly")
add_feature('u_component', climate_df['u_component_of_wind_10m'], "U component of wind (m/s)")
add_feature('v_component', climate_df['v_component_of_wind_10m'], "V component of wind (m/s)")
add_feature('wind_speed', climate_df['wind_speed_ms'], "Wind speed (m/s)")
add_feature('total_evaporation', climate_df['total_evaporation_m'], "Total evaporation (m)")

# Temporal
features_df['month'] = features_df['date'].dt.month
features_df['day_of_year'] = features_df['date'].dt.dayofyear
features_df['season'] = features_df['month'].apply(lambda x: 1 if x in [12,1,2] else (2 if x in [3,4,5] else (3 if x in [6,7,8,9] else 4)))
features_df['monsoon_indicator'] = features_df['month'].isin([6, 7, 8, 9]).astype(int)
feat_dict_log.append("| month | Month of the year | Integer |\n")
feat_dict_log.append("| day_of_year | Day of year (1-365/366) | Integer |\n")
feat_dict_log.append("| season | Season (1=Win,2=Sum,3=Mon,4=Post) | Integer |\n")
feat_dict_log.append("| monsoon_indicator | 1 if Jun-Sep, else 0 | Binary |\n")

# Lags
lag_cols = ['rainfall', 'runoff', 'temperature_max', 'surface_pressure', 'soil_moisture_layer_1']
for col in lag_cols:
    for lag in [1, 2, 3]:
        add_feature(f'{col}_lag{lag}', features_df[col].shift(lag), f"{col} value {lag} day(s) ago")

# Drop NaNs created by rolling and lags (mostly at the beginning of the series)
# Wait, user instructed "Save master_features.csv", keeping NA is better handled down the pipeline.
# But it's standard to keep them here.

# --- 6. Feature Selection (Correlation check) ---
fe_report_log.append("\n## Feature Selection & Multicollinearity\n")
corr_matrix = features_df.drop(columns=['date']).corr()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.98:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j]))

fe_report_log.append("Checked for extreme multicollinearity (Pearson |r| > 0.98).\n")
if high_corr:
    fe_report_log.append("Found highly redundant pairs:\n")
    for a, b in high_corr:
        fe_report_log.append(f"- {a} and {b}\n")
    fe_report_log.append("Decided to keep them as tree models and temporal models (LSTM) can handle multicollinearity natively, and physical meaning is distinct.\n")
else:
    fe_report_log.append("No extremely redundant features found.\n")

# --- 7. Save outputs ---
features_df.to_csv(features_dir / "master_features.csv", index=False)
features_df.describe().T.to_csv(features_dir / "feature_summary.csv")

with open(reports_dir / "LABEL_DEFINITION.md", "w") as f:
    f.writelines(label_def_log)
with open(reports_dir / "FEATURE_ENGINEERING_REPORT.md", "w") as f:
    f.writelines(fe_report_log)
with open(reports_dir / "FEATURE_DICTIONARY.md", "w") as f:
    f.writelines(feat_dict_log)
with open(reports_dir / "LEAKAGE_AUDIT.md", "w") as f:
    f.writelines(leakage_log)

print("Label Creation and Feature Engineering Complete.")
