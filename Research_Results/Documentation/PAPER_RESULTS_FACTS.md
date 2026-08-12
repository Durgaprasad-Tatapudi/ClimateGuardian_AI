# PAPER RESULTS FACTS

*This document contains strictly verified numerical outcomes generated directly from project evaluation files. All statements are safe for inclusion in academic reporting.*

## 1. Dataset Dimensions
- Master Feature Data Points: 6,940 days (45 Features per day)
- Observation Start: 2000-01-01
- Observation End: 2018-12-31
- Missing Values (Features): 30
- Total Positive Flood Samples: 1,880 (27.09%)
- Total Positive Heatwave Samples: 510 (7.35%)
- Total Positive Compound Samples: 255 (3.67%)

## 2. Chronological Splitting Boundaries
- **Train Split (2000-2012):** 4,749 samples (68.4%)
- **Validation Split (2013-2015):** 1,095 samples (15.8%)
- **Test Split (2016-2018):** 1,096 samples (15.8%)
- **Training Accuracy:** NOT AVAILABLE — DO NOT REPORT. (Epoch-level training performance was not explicitly persisted as a finalized metric block, preventing verifiable training accuracy statements).

## 3. Preprocessing Evidence
- Scaler Training: strictly constrained to 4,749 sample array.
- Missing Value Imputation: fitted strictly to `X_train.mean()` without utilizing future temporal samples.
- Backwards propagation (leakage): Fully removed prior to evaluation.

## 4. Primary Test Set Metrics (Test 2016-2018)

### Flood
- **Selected Model:** RandomForest
- Accuracy: 69.16%
- Precision: 0.3539
- Recall: 0.6930
- F1-Score: 0.4686
- ROC-AUC: 0.7184
- PR-AUC: 0.3468
- Brier Score: 0.2391

### Heatwave
- **Selected Model:** LightGBM (Restricted to 41 features)
- Accuracy: 97.99%
- Precision: 0.8788
- Recall: 0.8969
- F1-Score: 0.8878
- ROC-AUC: 0.9969
- PR-AUC: 0.9711
- Brier Score: 0.0149

### Compound Risk
- **Highest Evaluating Offline Model:** GRU
  - F1-Score: 0.4343
  - ROC-AUC: 0.9598
  - PR-AUC: 0.4838
- **Selected Operational Model:** Stacking Ensemble (XGBoost + LightGBM → LogisticRegression)
  - F1-Score: 0.3419
  - ROC-AUC: 0.9495
  - PR-AUC: 0.3891
  - Logistic Regression Meta-Feature Input Size: 2 (`[xgb_prob, lgbm_prob]`)

## 5. Temporal Lead-Time Metrics (Heatwave F1-Score)
- 1-Day Horizon: 0.8500
- 3-Day Horizon: 0.7400
- 5-Day Horizon: 0.6500
- 7-Day Horizon: 0.6200

## 6. Ablation Experiment (Compound Target F1-Score)
- Climate Only: 0.2400
- Hydro Only: 0.2500
- Climate + Hydro: 0.2286
- Climate + Hydro + Temporal/Lag: 0.4051
- Full Features: 0.4051

## 7. Model Calibration (Compound Target Brier Score)
- Uncalibrated: 0.2013
- Platt Scaling: 0.0338
- Isotonic Regression: 0.0386

## 8. Realtime Readiness 
- The inference test battery passed cleanly against point-in-time Open-Meteo REST calls matching the exact 45-feature dimensionality array expected by the scaling layers.
- Realtime probabilities successfully generated without interpolation from historical arrays. (DO NOT report these independent realtime outputs as historical test metrics).
