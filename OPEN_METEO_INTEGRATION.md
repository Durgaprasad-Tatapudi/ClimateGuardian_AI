# Open-Meteo Integration Guide

## Overview
This document describes the integration of the Open-Meteo API into the live operational prediction pipeline.

## API Usage
We use the `v1/forecast` endpoint from Open-Meteo. 
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Past Days**: 14 days (required to compute historical lags and rolling windows up to 14 days).
- **Forecast Days**: 7 days (the target horizons for operational predictions).

## Source Harmonization
The historical ML models were trained using ERA5, CHIRPS, and NASA POWER datasets. Since Open-Meteo uses an ensemble of weather models (e.g., GFS, ECMWF), we apply standard harmonization:
- **Units**: Open-Meteo outputs metric units compatible with our historical data (Celsius, mm, hPa).
- **Temporal Resolution**: Open-Meteo is requested at `hourly` resolution and aggregated to daily intervals (mean/min/max/sum) exactly matching the historical aggregation methods.
- **Wind Vector Conversion**: Open-Meteo provides `wind_speed_10m` and `wind_direction_10m`. We convert this into meteorological `u` and `v` components to match the historical model inputs.
- **Anomalies**: Baseline metrics from the training dataset (2000-2014) are explicitly loaded and applied to Open-Meteo data to compute anomalies, preventing data leakage and distribution shifts.

## Variables
The following variable mappings are applied:
- `temperature_2m` -> `temperature_avg`, `temperature_min`, `temperature_max`
- `dew_point_2m` -> `dewpoint`, `dewpoint_min`, `dewpoint_max`
- `precipitation` -> `rainfall`
- `runoff` -> `runoff`, `surface_runoff`
- `soil_moisture_0_to_1cm` -> `soil_moisture_layer_1`
- `surface_pressure` -> `surface_pressure`
- `evapotranspiration` -> `total_evaporation`

## Error Handling
The `OpenMeteoClient` is wrapped with automatic retries for common server errors (408, 429, 50x) with exponential backoff and connection timeouts to prevent blocking operational queues.
