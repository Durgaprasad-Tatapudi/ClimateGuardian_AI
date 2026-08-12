import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

root_dir = Path("01_Raw_Datasets")

def get_temporal_resolution(dates):
    if len(dates) < 2:
        return "UNKNOWN"
    dates = sorted(dates)
    diffs = np.diff(dates)
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
    info = {}
    info['relative_path'] = str(filepath.relative_to(root_dir.parent)).replace("\\\\", "/")
    info['filename'] = filepath.name
    info['file_size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
    info['metadata'] = ""
    
    skiprows = 0
    is_emdat = "emdat" in filepath.name.lower()
    
    try:
        if filepath.suffix == '.csv':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline().strip() for _ in range(50)]
            has_nasa_header = any("-END HEADER-" in line for line in lines)
            if has_nasa_header:
                metadata_lines = []
                for i, line in enumerate(lines):
                    metadata_lines.append(line)
                    if "-END HEADER-" in line:
                        skiprows = i + 1
                        break
                info['metadata'] = "\\n".join(metadata_lines[:15]) + ("..." if len(metadata_lines) > 15 else "")
            
            if skiprows > 0:
                df = pd.read_csv(filepath, skiprows=skiprows, low_memory=False)
            else:
                df = pd.read_csv(filepath, low_memory=False)
                
        elif filepath.suffix == '.xlsx':
            # EM-DAT typically has a 6-line header, meaning row 6 is the header.
            if is_emdat:
                df_test = pd.read_excel(filepath, nrows=15)
                # find header row
                header_row = 0
                for i, row in df_test.iterrows():
                    if "Dis No" in str(row.values) or "Year" in str(row.values) or "Country" in str(row.values):
                        header_row = i + 1
                        break
                df = pd.read_excel(filepath, skiprows=header_row)
                info['metadata'] = f"EM-DAT Header skipped {header_row} rows."
            else:
                df = pd.read_excel(filepath)
                
        info['num_rows'] = len(df)
        info['num_cols'] = len(df.columns)
        info['columns'] = list(df.columns)
        info['dtypes'] = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        
        info['head'] = df.head().to_dict(orient='records')
        info['tail'] = df.tail().to_dict(orient='records')
        
        info['missing_counts'] = {str(k): v for k, v in df.isnull().sum().to_dict().items()}
        info['missing_pct'] = {str(k): v for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()}
        
        info['duplicate_count'] = int(df.duplicated().sum())
        
        date_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['date', 'time', 'year', 'month', 'start', 'end'])]
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
                        break
                except:
                    pass
        
        lat_col = next((col for col in df.columns if str(col).lower() in ['lat', 'latitude']), None)
        lon_col = next((col for col in df.columns if str(col).lower() in ['lon', 'longitude', 'long']), None)
        info['lat_col'] = lat_col
        info['lon_col'] = lon_col
        
        if lat_col and lon_col:
            info['geo_coverage'] = {
                'lat_min': float(df[lat_col].min()), 'lat_max': float(df[lat_col].max()),
                'lon_min': float(df[lon_col].min()), 'lon_max': float(df[lon_col].max())
            }
            info['unique_locations'] = int(df[[lat_col, lon_col]].drop_duplicates().shape[0])
        else:
            info['geo_coverage'] = "NOT CONFIRMED"
            info['unique_locations'] = "NOT CONFIRMED"
            
        num_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[num_cols].describe().to_dict()
        info['num_stats'] = {str(k): v for k, v in stats.items()}
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        cat_counts = {}
        for col in cat_cols:
            uniques = df[col].dropna().unique()
            if len(uniques) <= 15:
                cat_counts[str(col)] = [str(x) for x in uniques]
            else:
                cat_counts[str(col)] = f"{len(uniques)} unique values (too many to list)"
        info['cat_counts'] = cat_counts
        
    except Exception as e:
        info['error'] = str(e)
        
    return info

results = []
for ext in ['*.csv', '*.xlsx']:
    for file_path in sorted(root_dir.rglob(ext)):
        results.append(inspect_file(file_path))

with open("inspection_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Generate Markdown Report
md = "# Dataset Inspection Report\\n\\n"

md += "## A. Dataset Inventory\\n"
for res in results:
    md += f"- **{res['filename']}**: {res['relative_path']} ({res.get('file_size_mb', 0):.2f} MB, {res.get('num_rows', 'ERROR')} rows)\\n"

def get_datasets_with_keywords(keywords):
    matched = []
    for res in results:
        cols = " ".join(res.get('columns', [])).lower()
        if any(k in cols for k in keywords) or any(k in res['filename'].lower() for k in keywords):
            matched.append(res['filename'])
    return matched

md += "\\n## B. Flood-relevant Datasets\\n"
flood_ds = get_datasets_with_keywords(['flood', 'rain', 'precipitation', 'runoff', 'soil', 'moisture', 'prectot'])
md += ", ".join(flood_ds) if flood_ds else "None identified"
md += "\\n"

md += "\\n## C. Heatwave-relevant Datasets\\n"
heat_ds = get_datasets_with_keywords(['heat', 'temperature', 't2m', 'temp', 'evaporation', 'humidity', 'wind', 'rh2m'])
md += ", ".join(heat_ds) if heat_ds else "None identified"
md += "\\n"

md += "\\n## D. Common Climate Datasets\\n"
common_ds = [r['filename'] for r in results if 'common' in r['relative_path'].lower()]
md += ", ".join(common_ds) if common_ds else "None identified"
md += "\\n"

md += "\\n## E. Disaster-event Datasets\\n"
event_ds = [r['filename'] for r in results if 'event' in r['relative_path'].lower() or 'emdat' in r['filename'].lower()]
md += ", ".join(event_ds) if event_ds else "None identified"
md += "\\n"

md += "\\n## F. Recent Datasets\\n"
recent_ds = [r['filename'] for r in results if 'recent' in r['relative_path'].lower()]
md += ", ".join(recent_ds) if recent_ds else "None identified"
md += "\\n"

md += "\\n## G. Date-range Compatibility\\n"
for res in results:
    if 'error' not in res:
        md += f"- **{res['filename']}**: {res['min_date']} to {res['max_date']} ({res['temporal_resolution']})\\n"
md += "\\n"

md += "\\n## H. Geographic Compatibility\\n"
for res in results:
    if 'error' not in res:
        md += f"- **{res['filename']}**: "
        if res['geo_coverage'] != "NOT CONFIRMED":
            geo = res['geo_coverage']
            md += f"Lat: {geo['lat_min']:.2f} to {geo['lat_max']:.2f}, Lon: {geo['lon_min']:.2f} to {geo['lon_max']:.2f} ({res['unique_locations']} locations)\\n"
        else:
            md += "NOT CONFIRMED\\n"
md += "\\n"

md += "\\n## I. Missing-data Summary\\n"
for res in results:
    if 'error' not in res:
        missing = [(k,v) for k,v in res['missing_pct'].items() if v > 0]
        if missing:
            miss_str = ", ".join([f"{k} ({v:.1f}%)" for k,v in missing])
            md += f"- **{res['filename']}**: {miss_str}\\n"
        else:
            md += f"- **{res['filename']}**: No missing data\\n"
md += "\\n"

md += "\\n## J. Duplicate Summary\\n"
for res in results:
    if 'error' not in res:
        md += f"- **{res['filename']}**: {res['duplicate_count']} duplicate rows\\n"
md += "\\n"

md += "\\n## K. Potential Data Quality Issues\\n"
md += "- Checked minimum/maximum values. Suspicious values include `-999` (common missing value indicator) in NASA POWER. Negative values in variables that should be positive (like precipitation) should be checked.\\n"
for res in results:
    if 'error' not in res:
        stats = res.get('num_stats', {})
        issues = []
        for col, s in stats.items():
            if 'min' in s and s['min'] == -999:
                issues.append(f"{col} has min -999")
            if 'rain' in col.lower() or 'precip' in col.lower():
                if 'min' in s and s['min'] < 0 and s['min'] != -999:
                    issues.append(f"{col} has negative rain: {s['min']}")
        if issues:
            md += f"- **{res['filename']}**: {', '.join(issues)}\\n"

md += "\\n## L. Potential Target/Label Candidates\\n"
md += "- **Flood**: `India_Flood_Events_GFD_2000_2018.csv` (contains event data), `public_emdat_custom_request...xlsx` (Flood disaster types)\\n"
md += "- **Heatwave**: `public_emdat_custom_request...xlsx` (Extreme Temperature disaster types)\\n"

md += "\\n## M. Recommended Next Step for Data Cleaning\\n"
md += "1. Replace `-999` with `NaN` in NASA POWER data.\\n"
md += "2. Parse date columns to standardized `datetime` format across all datasets.\\n"
md += "3. Align temporal resolution (e.g., resample Daily data if any dataset is Monthly).\\n"
md += "4. Filter EM-DAT dataset to Flood and Extreme Temperature specifically for India.\\n"
md += "5. Align geographic grids or locations between Point-based (station/city) and Gridded data.\\n"

md += "\\n## Detailed Dataset Inspection\\n\\n"
for res in results:
    md += f"### {res['filename']}\\n"
    if 'error' in res:
        md += f"**ERROR**: {res['error']}\\n\\n"
        continue
    md += f"- **Path**: `{res['relative_path']}`\\n"
    md += f"- **Size**: {res['file_size_mb']:.2f} MB\\n"
    md += f"- **Rows x Cols**: {res['num_rows']} x {res['num_cols']}\\n"
    md += f"- **Columns**: {', '.join(res['columns'])}\\n"
    md += f"- **Data Types**: {res['dtypes']}\\n"
    md += f"- **Temporal**: {res['min_date']} to {res['max_date']} ({res['temporal_resolution']})\\n"
    
    geo = res['geo_coverage']
    if geo != "NOT CONFIRMED":
        md += f"- **Geography**: Lat {geo['lat_min']:.2f} to {geo['lat_max']:.2f}, Lon {geo['lon_min']:.2f} to {geo['lon_max']:.2f}\\n"
    else:
        md += f"- **Geography**: NOT CONFIRMED\\n"
        
    if res['metadata']:
        md += f"- **Metadata/Header Notes**: {res['metadata'].replace(chr(10), ' ')}\\n"
        
    md += "\\n**Head (First 2 rows)**:\\n```json\\n" + json.dumps(res['head'][:2], indent=2) + "\\n```\\n\\n"
    md += "\\n**Tail (Last 2 rows)**:\\n```json\\n" + json.dumps(res['tail'][-2:], indent=2) + "\\n```\\n\\n"

with open("data_inspection_report.md", "w") as f:
    f.write(md)

print("Report generated successfully.")
