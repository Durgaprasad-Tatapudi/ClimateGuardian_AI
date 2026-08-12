import os
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = Path("01_Raw_Datasets")

def get_temporal_resolution(dates):
    if len(dates) < 2:
        return "UNKNOWN"
    dates = sorted(dates)
    diffs = np.diff(dates)
    # Most common difference
    if len(diffs) > 0:
        val, counts = np.unique(diffs, return_counts=True)
        common_diff = val[np.argmax(counts)]
        days = common_diff.astype('timedelta64[D]').astype(int)
        if days == 1:
            return "Daily"
        elif 28 <= days <= 31:
            return "Monthly"
        elif 365 <= days <= 366:
            return "Yearly"
        else:
            return f"{days} Days"
    return "UNKNOWN"

def inspect_file(filepath):
    print(f"Inspecting: {filepath}")
    info = {}
    info['relative_path'] = str(filepath.relative_to(root_dir.parent))
    info['filename'] = filepath.name
    info['file_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
    
    # Read metadata header if present
    metadata_lines = []
    skiprows = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [f.readline().strip() for _ in range(50)]
        
    has_nasa_header = any("-END HEADER-" in line for line in lines)
    is_emdat = "emdat" in filepath.name.lower()
    
    if has_nasa_header:
        for i, line in enumerate(lines):
            metadata_lines.append(line)
            if "-END HEADER-" in line:
                skiprows = i + 1
                break
    elif is_emdat:
        # EM-DAT typically has a 6-line header
        if len(lines) > 6 and "Dis No" in lines[6]:
            skiprows = 6
            metadata_lines = lines[:6]
    
    info['metadata'] = "\\n".join(metadata_lines[:15]) + ("..." if len(metadata_lines) > 15 else "")
    
    try:
        if skiprows > 0:
            df = pd.read_csv(filepath, skiprows=skiprows, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)
            
        info['num_rows'] = len(df)
        info['num_cols'] = len(df.columns)
        info['columns'] = list(df.columns)
        info['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Head and tail
        info['head'] = df.head().to_dict(orient='records')
        info['tail'] = df.tail().to_dict(orient='records')
        
        # Missing values
        info['missing_counts'] = df.isnull().sum().to_dict()
        info['missing_pct'] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Duplicates
        info['duplicate_count'] = int(df.duplicated().sum())
        
        # Dates
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'year', 'month'])]
        info['date_cols'] = date_cols
        info['min_date'] = "NOT CONFIRMED"
        info['max_date'] = "NOT CONFIRMED"
        info['temporal_resolution'] = "NOT CONFIRMED"
        
        if date_cols:
            for dc in date_cols:
                try:
                    if str(df[dc].dtype) == 'object' or 'datetime' not in str(df[dc].dtype):
                        parsed_dates = pd.to_datetime(df[dc].dropna(), errors='coerce')
                    else:
                        parsed_dates = df[dc].dropna()
                    
                    if len(parsed_dates) > 0 and parsed_dates.notnull().sum() > 0:
                        info['min_date'] = str(parsed_dates.min())
                        info['max_date'] = str(parsed_dates.max())
                        info['temporal_resolution'] = get_temporal_resolution(parsed_dates)
                        break # Found one valid
                except:
                    pass
        
        # Geospatial
        lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
        lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude', 'long']), None)
        info['lat_col'] = lat_col
        info['lon_col'] = lon_col
        
        if lat_col and lon_col:
            info['geo_coverage'] = {
                'lat_min': float(df[lat_col].min()), 'lat_max': float(df[lat_col].max()),
                'lon_min': float(df[lon_col].min()), 'lon_max': float(df[lon_col].max())
            }
            # Unique locations
            info['unique_locations'] = int(df[[lat_col, lon_col]].drop_duplicates().shape[0])
        else:
            info['geo_coverage'] = "NOT CONFIRMED"
            info['unique_locations'] = "NOT CONFIRMED"
            
        # Descriptive stats for numerical
        num_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[num_cols].describe().to_dict()
        info['num_stats'] = stats
        
        # Categorical unique values (up to 10)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_counts = {}
        for col in cat_cols:
            uniques = df[col].dropna().unique()
            if len(uniques) <= 15:
                cat_counts[col] = list(uniques)
            else:
                cat_counts[col] = f"{len(uniques)} unique values (too many to list)"
        info['cat_counts'] = cat_counts
        
    except Exception as e:
        info['error'] = str(e)
        
    return info

results = []
for file_path in sorted(root_dir.rglob('*.csv')):
    results.append(inspect_file(file_path))

import json
with open('inspection_results.json', 'w') as f:
    json.dump(results, f, indent=2)
