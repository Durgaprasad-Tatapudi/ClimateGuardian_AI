# Real-Time Operational Prediction Architecture

## Overview
This document outlines the architecture for the live operational inference pipeline. 
The system connects to Open-Meteo's REST APIs to fetch weather data, synchronizes it with the schema of the frozen historical ML model, and produces calibrated risk assessments.

## Pipeline Architecture

1. **Open-Meteo Client (`src/realtime/open_meteo_client.py`)**
   - Handles data retrieval for the specified latitude/longitude coordinates.
   - Implements robust error handling: retry logic with exponential backoff on server failures (500, 502, etc.) and timeouts.
   - Validates that the payload contains required `hourly` data.

2. **Live Feature Builder (`src/realtime/live_feature_builder.py`)**
   - Aggregates the 14-days past and 7-days forecast `hourly` variables into daily variables.
   - Performs harmonization to exactly match historical feature expectations (e.g., standardizing `wind_speed_10m` to `u_component` and `v_component`).
   - Retrieves `train_means` (extracted statically from `master_features.csv` during the 2000-2014 period) to maintain identical anomaly baselines.
   - Fits/Loads the exact `StandardScaler` used during the training phase. **No refitting on live data occurs**.

3. **Realtime Predictor (`src/realtime/predictor.py`)**
   - Loads the exact pre-trained frozen `.joblib` models.
   - Evaluates flood (RandomForest), heatwave (LightGBM), and compound (Stacking Ensemble) probabilities.
   - Translates raw probabilities to semantic risk categories (Low, Medium, High) using validation-set derived thresholds.
   - Outputs structured, internal-agnostic JSON.

## Output Schema
```json
{
  "location": {"latitude": 28.6139, "longitude": 77.2090},
  "prediction_time": "2026-08-12T12:00:00Z",
  "data_source": "Open-Meteo",
  "model_version": "v1.0-frozen-research",
  "forecasts": [
    {
      "horizon_days": 0,
      "forecast_date": "2026-08-12",
      "probabilities": {"flood": 0.1, "heatwave": 0.4, "compound": 0.05},
      "risk_levels": {"flood": "Minimal", "heatwave": "Medium", "compound": "Minimal"}
    }
  ]
}
```
