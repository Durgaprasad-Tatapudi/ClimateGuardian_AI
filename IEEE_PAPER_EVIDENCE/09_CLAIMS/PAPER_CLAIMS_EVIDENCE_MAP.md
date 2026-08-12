# PAPER CLAIMS EVIDENCE MAP

| CLAIM | VERIFIED VALUE/STATEMENT | SOURCE FILE | EVIDENCE STATUS |
|---|---|---|---|
| Dataset size | 6940 daily records | scratch/audit_stats.json | VERIFIED |
| Feature count | 45 features | feature_cols.joblib | VERIFIED |
| Target distributions | Flood:1880, Heatwave:510, Compound:255 | labels.csv / audit_stats.json | VERIFIED |
| Chronological split | Train(00-12), Val(13-15), Test(16-18) | train_models.py | VERIFIED |
| Leakage remediation | Test excluded from all fit(), proxies removed for HW | code inspection | VERIFIED |
| Flood metrics | F1=0.4686 | MODEL_RESULTS.csv | VERIFIED |
| Heatwave metrics | F1=0.8878 | MODEL_RESULTS.csv | VERIFIED |
| Compound stacking metrics | F1=0.3419 | MODEL_RESULTS.csv | VERIFIED |
| GRU benchmark metrics | F1=0.4343 | MODEL_RESULTS.csv | VERIFIED |
| Lead-time results | 1D=0.85, 3D=0.74, 5D=0.65, 7D=0.62 | LEAD_TIME_RESULTS.csv | VERIFIED |
| Ablation results | Temporal lags significantly increase F1 | ABLATION_RESULTS.csv | VERIFIED |
| Calibration results | Platt Brier=0.0338 | CALIBRATION_RESULTS.csv | VERIFIED |
| SHAP verification | SHAP plots internal dependencies correctly | 16_SHAP_Summary_flood_target.png | VERIFIED |
| Realtime verification | Successfully executes over REST APIs | Realtime testing | VERIFIED |
| Testing verification | 10 tests passing | Pytest execution | VERIFIED |
