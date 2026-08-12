# Feature Dictionary
| Feature | Description | Type |
|---|---|---|
| rainfall | Daily rainfall (mm) | Numeric |
| rainfall_3d | 3-day cumulative rainfall | Numeric |
| rainfall_5d | 5-day cumulative rainfall | Numeric |
| rainfall_7d | 7-day cumulative rainfall | Numeric |
| rainfall_14d | 14-day cumulative rainfall | Numeric |
| runoff | Total runoff (m) | Numeric |
| surface_runoff | Surface runoff (m) | Numeric |
| runoff_3d | 3-day cumulative runoff | Numeric |
| runoff_7d | 7-day cumulative runoff | Numeric |
| soil_moisture_layer_1 | Soil moisture layer 1 | Numeric |
| soil_moisture_rolling_7d | 7-day average soil moisture | Numeric |
| soil_moisture_anomaly | Soil moisture standardized anomaly (relative to train) | Numeric |
| temperature_avg | Average temperature (C) | Numeric |
| temperature_min | Min temperature (C) | Numeric |
| temperature_max | Max temperature (C) | Numeric |
| rolling_temperature_3d | 3-day avg temperature | Numeric |
| temperature_anomaly | Temperature anomaly from train baseline | Numeric |
| dewpoint | Dewpoint (C) | Numeric |
| dewpoint_min | Min Dewpoint (C) | Numeric |
| dewpoint_max | Max Dewpoint (C) | Numeric |
| surface_pressure | Surface Pressure (hPa) | Numeric |
| pressure_anomaly | Pressure anomaly | Numeric |
| u_component | U component of wind (m/s) | Numeric |
| v_component | V component of wind (m/s) | Numeric |
| wind_speed | Wind speed (m/s) | Numeric |
| total_evaporation | Total evaporation (m) | Numeric |
| month | Month of the year | Integer |
| day_of_year | Day of year (1-365/366) | Integer |
| season | Season (1=Win,2=Sum,3=Mon,4=Post) | Integer |
| monsoon_indicator | 1 if Jun-Sep, else 0 | Binary |
| rainfall_lag1 | rainfall value 1 day(s) ago | Numeric |
| rainfall_lag2 | rainfall value 2 day(s) ago | Numeric |
| rainfall_lag3 | rainfall value 3 day(s) ago | Numeric |
| runoff_lag1 | runoff value 1 day(s) ago | Numeric |
| runoff_lag2 | runoff value 2 day(s) ago | Numeric |
| runoff_lag3 | runoff value 3 day(s) ago | Numeric |
| temperature_max_lag1 | temperature_max value 1 day(s) ago | Numeric |
| temperature_max_lag2 | temperature_max value 2 day(s) ago | Numeric |
| temperature_max_lag3 | temperature_max value 3 day(s) ago | Numeric |
| surface_pressure_lag1 | surface_pressure value 1 day(s) ago | Numeric |
| surface_pressure_lag2 | surface_pressure value 2 day(s) ago | Numeric |
| surface_pressure_lag3 | surface_pressure value 3 day(s) ago | Numeric |
| soil_moisture_layer_1_lag1 | soil_moisture_layer_1 value 1 day(s) ago | Numeric |
| soil_moisture_layer_1_lag2 | soil_moisture_layer_1 value 2 day(s) ago | Numeric |
| soil_moisture_layer_1_lag3 | soil_moisture_layer_1 value 3 day(s) ago | Numeric |
