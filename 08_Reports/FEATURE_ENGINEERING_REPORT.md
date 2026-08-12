# Feature Engineering Report
## Feature Construction

## Feature Selection & Multicollinearity
Checked for extreme multicollinearity (Pearson |r| > 0.98).
Found highly redundant pairs:
- rainfall_5d and rainfall_3d
- rainfall_7d and rainfall_5d
- runoff_3d and runoff
- soil_moisture_rolling_7d and soil_moisture_layer_1
- soil_moisture_anomaly and soil_moisture_layer_1
- soil_moisture_anomaly and soil_moisture_rolling_7d
- temperature_min and temperature_avg
- temperature_max and temperature_avg
- rolling_temperature_3d and temperature_avg
- rolling_temperature_3d and temperature_min
- rolling_temperature_3d and temperature_max
- temperature_anomaly and temperature_avg
- temperature_anomaly and temperature_min
- temperature_anomaly and temperature_max
- temperature_anomaly and rolling_temperature_3d
- dewpoint_min and dewpoint
- dewpoint_max and dewpoint
- dewpoint_max and dewpoint_min
- pressure_anomaly and surface_pressure
- day_of_year and month
- rainfall_lag1 and rainfall_3d
- runoff_lag1 and runoff_3d
- runoff_lag2 and runoff_3d
- temperature_max_lag1 and temperature_avg
- temperature_max_lag1 and temperature_max
- temperature_max_lag1 and rolling_temperature_3d
- temperature_max_lag1 and temperature_anomaly
- temperature_max_lag2 and temperature_avg
- temperature_max_lag2 and temperature_max
- temperature_max_lag2 and rolling_temperature_3d
- temperature_max_lag2 and temperature_anomaly
- temperature_max_lag2 and temperature_max_lag1
- temperature_max_lag3 and rolling_temperature_3d
- temperature_max_lag3 and temperature_max_lag1
- temperature_max_lag3 and temperature_max_lag2
- soil_moisture_layer_1_lag1 and soil_moisture_layer_1
- soil_moisture_layer_1_lag1 and soil_moisture_rolling_7d
- soil_moisture_layer_1_lag1 and soil_moisture_anomaly
- soil_moisture_layer_1_lag2 and soil_moisture_layer_1
- soil_moisture_layer_1_lag2 and soil_moisture_rolling_7d
- soil_moisture_layer_1_lag2 and soil_moisture_anomaly
- soil_moisture_layer_1_lag2 and soil_moisture_layer_1_lag1
- soil_moisture_layer_1_lag3 and soil_moisture_layer_1
- soil_moisture_layer_1_lag3 and soil_moisture_rolling_7d
- soil_moisture_layer_1_lag3 and soil_moisture_anomaly
- soil_moisture_layer_1_lag3 and soil_moisture_layer_1_lag1
- soil_moisture_layer_1_lag3 and soil_moisture_layer_1_lag2
Decided to keep them as tree models and temporal models (LSTM) can handle multicollinearity natively, and physical meaning is distinct.
