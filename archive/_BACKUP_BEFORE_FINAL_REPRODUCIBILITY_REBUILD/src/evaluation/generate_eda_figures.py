import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

figures_dir = Path("07_Figures")
os.makedirs(figures_dir, exist_ok=True)

print("Loading data for EDA figures...")
features = pd.read_csv("03_Features/master_features.csv")
features['date'] = pd.to_datetime(features['date'])
flood_lbl = pd.read_csv("04_Labels/flood_labels.csv")
flood_lbl['date'] = pd.to_datetime(flood_lbl['date'])
heatwave_lbl = pd.read_csv("04_Labels/heatwave_labels.csv")
heatwave_lbl['date'] = pd.to_datetime(heatwave_lbl['date'])
compound_lbl = pd.read_csv("04_Labels/compound_labels.csv")
compound_lbl['date'] = pd.to_datetime(compound_lbl['date'])

df = features.merge(flood_lbl[['date', 'flood_target']], on='date')
df = df.merge(heatwave_lbl[['date', 'heatwave_target']], on='date')
df = df.merge(compound_lbl[['date', 'compound_target']], on='date')
df = df.sort_values('date').reset_index(drop=True)

print("Generating Figure 1: Dataset overview...")
fig, ax = plt.subplots(figsize=(10, 6))
# A simple missing value map or feature distributions. Let's do a missing values heatmap as dataset overview
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax, yticklabels=False)
ax.set_title("Dataset Overview (Missing Values Mapping)")
plt.tight_layout()
plt.savefig(figures_dir / "01_Dataset_overview.png")
plt.close()

def plot_time_series(df, columns, title, filename, ylabel="Value"):
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in columns:
        if col in df.columns:
            ax.plot(df['date'], df[col], label=col, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / filename)
    plt.close()

print("Generating time series figures (2-5)...")
# 2. Rainfall time series
rainfall_cols = [c for c in df.columns if 'rainfall' in c.lower() and 'lag' not in c.lower()][:1] # Pick the first main one
if not rainfall_cols: rainfall_cols = [c for c in df.columns if 'precip' in c.lower()][:1]
if rainfall_cols:
    plot_time_series(df, rainfall_cols, "Rainfall Time Series", "02_Rainfall_time_series.png", "Rainfall (mm)")

# 3. Temperature time series
temp_cols = [c for c in df.columns if 'temperature' in c.lower() and 'lag' not in c.lower()][:2]
if temp_cols:
    plot_time_series(df, temp_cols, "Temperature Time Series", "03_Temperature_time_series.png", "Temperature (°C)")

# 4. Runoff time series
runoff_cols = [c for c in df.columns if 'runoff' in c.lower() and 'lag' not in c.lower()][:1]
if runoff_cols:
    plot_time_series(df, runoff_cols, "Runoff Time Series", "04_Runoff_time_series.png", "Runoff")

# 5. Soil moisture time series
soil_cols = [c for c in df.columns if 'soil_moisture' in c.lower() and 'lag' not in c.lower()][:2]
if soil_cols:
    plot_time_series(df, soil_cols, "Soil Moisture Time Series", "05_Soil_moisture_time_series.png", "Soil Moisture")

print("Generating Figure 6: Correlation matrix...")
fig, ax = plt.subplots(figsize=(12, 10))
# Select a subset of main features to keep correlation matrix readable
main_features = [c for c in df.columns if 'lag' not in c.lower() and c not in ['date', 'flood_target', 'heatwave_target', 'compound_target']]
corr = df[main_features].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, center=0, cbar_kws={"shrink": .7})
ax.set_title("Correlation Matrix of Core Features")
plt.tight_layout()
plt.savefig(figures_dir / "06_Correlation_matrix.png")
plt.close()

def plot_event_timeline(df, target_col, title, filename):
    fig, ax = plt.subplots(figsize=(14, 3))
    events = df[df[target_col] == 1]
    ax.vlines(events['date'], ymin=0, ymax=1, color='red', alpha=0.5, linewidth=2)
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(figures_dir / filename)
    plt.close()

print("Generating event timelines (7-9)...")
# 7. Flood event timeline
plot_event_timeline(df, 'flood_target', "Flood Event Timeline", "07_Flood_event_timeline.png")

# 8. Heatwave event timeline
plot_event_timeline(df, 'heatwave_target', "Heatwave Event Timeline", "08_Heatwave_event_timeline.png")

# 9. Compound event timeline
plot_event_timeline(df, 'compound_target', "Compound Event Timeline", "09_Compound_event_timeline.png")

print("EDA Figures generation complete.")
