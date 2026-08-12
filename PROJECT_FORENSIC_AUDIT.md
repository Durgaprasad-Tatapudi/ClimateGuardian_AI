# PROJECT FORENSIC AUDIT: ClimateGuardian AI

## 1. Executive Summary
This document provides a highly rigorous, read-only forensic audit of the `ClimateGuardian_AI` repository. No values have been assumed. All metrics, dataset distributions, architectures, and hyperparameters have been empirically validated against the source code and artifact outputs. Overall, the project possesses an impressively sophisticated pipeline. However, critical discrepancies exist between the empirically computed metrics stored in the logs and the final values cited in the IEEE research paper (Phase 17).

## 2. Project Architecture
The project adheres to a standard data science paradigm.
- `01_Raw_Datasets` -> Raw API and historical outputs.
- `02_Processed_Data` -> Imputed and normalized arrays.
- `03_Features` / `04_Labels` -> The final master training variables.
- `05_Models` -> Serialized artifacts (Joblib, PyTorch).
- `06_Results` -> CSV evaluation logs.
- `src/` -> Modular Python pipeline encompassing ingestion, ML training, RNN deep learning, and realtime stateless mapping.

## 3. Complete File Inventory
- **Python Code (`src/`)**: Includes `build_features.py`, `data_cleaning.py`, `train_models.py`, `evaluate_models.py`, `open_meteo_client.py`. (Actively used for processing and inference).
- **Configurations (`configs/`)**: Missing explicit hyperparameter JSONs; parameters are natively declared inside `train_models.py`.
- **Model Artifacts (`05_Models_corrected`)**: Houses `scaler.joblib`, `train_means.joblib`, `RF_flood_target.joblib`, `LSTM_heatwave_target.pt`, etc.

## 4. Dataset Inventory
1. `master_features.csv` (Primary ML Input)
2. `flood_labels.csv` (Flood target definition)
3. `heatwave_labels.csv` (Extreme temperature target definition)
4. `compound_labels.csv` (Intersecting target definition)

## 5. Dataset Statistics
- **Date Range**: 2000-01-01 to 2018-12-31
- **Total Rows**: 6,940
- **Total Input Features**: 45
- **Data Quality**: Verified via pandas check; no trailing NaNs post-forward-fill.

## 6. Label Definitions
Extracted directly from `src/features/build_features.py`:
- **Flood**: `1` if the calendar date falls within the `start_date` and `end_date` of a verified India flood event in the GFD dataset. Otherwise `0`.
- **Heatwave**: `1` if `temperature_max_C` exceeds the 90th percentile threshold (calculated ONLY on the 2000-2014 training period) for 3 consecutive days.
- **Compound**: `1` if a Flood event (`1`) intersects with a recent Heatwave event (`hw_recent_7d == 1`).

## 7. Label Counts
- **Flood**: 1,880 Positives
- **Heatwave**: 510 Positives
- **Compound**: 255 Positives

## 8. Data Cleaning
- Missing values handled aggressively using backward/forward filling (`X.ffill().bfill()`) to prevent temporal leakage into historical validation limits.
- Geometrical attributes (`.geo`) dropped as irrelevant categorical constants.

## 9. Preprocessing
- **Scaling**: `StandardScaler` fitted strictly on `X_train`, subsequently applied to `X_val` and `X_test`.
- **Imputation**: Start-of-array `NaNs` handled via `X_train.mean()` injection.

## 10. Feature Engineering
- Extensive temporal lags (1, 2, 3 days) implemented for `rainfall`, `runoff`, `temperature_max`, etc.
- Multi-day rolling accumulators (3, 5, 7, 14 days).
- Standardized anomalies generated via baseline (2000-2014) training metrics.

## 11. Leakage Audit
- **Pre-processing Leakage**: **PASS**. Scalers are fitted exclusively on the 2000-2012 chronological split.
- **Label Leakage**: **PASS/WARNING**. The Heatwave rolling 90th-percentile baseline uses a 2000-2014 limit (which encroaches slightly into the 2013-2015 Validation zone), but successfully isolates the 2016-2018 Test zone. 
- **Heatwave Proxy Risk**: **PASS**. `temperature_max` and `hw_rolling` variables are dynamically dropped via `restricted_cols` array mapping to prevent catastrophic deterministic leakage during modeling.

## 12. Train/Validation/Test Split
Strict Chronological Isolation:
- **Train (2000-2012)**: 4,749 samples.
- **Val (2013-2015)**: 1,095 samples.
- **Test (2016-2018)**: 1,096 samples.

## 13. Model Inventory
- Baseline: `LogisticRegression`
- Emsembles: `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`
- Deep Learning (Offline Benchmarks): PyTorch `LSTM` and `GRU` (`seq_length=14`).

## 14. Hyperparameters
Parsed from `train_models.py` `RandomizedSearchCV`:
- **RF**: `n_estimators`: [50, 100], `max_depth`: [5, 10].
- **XGB**: `n_estimators`: [50, 100], `learning_rate`: [0.01, 0.1], `max_depth`: [3, 5].
- **LGBM**: `n_estimators`: [50, 100], `learning_rate`: [0.01, 0.1], `num_leaves`: [15, 31].
- **GRU/LSTM**: `hidden_dim`: 32, `dropout`: 0.2, `learning_rate`: 0.01.

## 15. Training Procedure
- TimeSeriesSplit CV utilized for random search over parameters.
- GRU implements epoch=15 early-stopping via PyTorch `validation_loss`.

## 16. Class Imbalance Handling
- **ML**: Utilizes `scale_pos_weight = (len(y) - sum(y)) / sum(y)` or `class_weight='balanced'`.
- **DL**: Employs `pos_weight` inside `nn.BCEWithLogitsLoss`.

## 17. Evaluation Procedure
Tested exclusively on the unseen 2016-2018 array block using `F1`, `ROC-AUC`, and `Brier`.

## 18. Verified Metrics
Empirical execution (`model_evaluation_results.csv`) yields:
- **Flood (RF)**: F1 = 0.4709
- **Heatwave (LGBM)**: F1 = 1.000 (Highly suspicious, potential dataset redundancy leak).
- **Compound (GRU)**: F1 = 0.3418

## 19. Confusion Matrices
*IMPLEMENTED BUT NOT REPORTED IN METRICS JSON (relies on `evaluate_models.py` Matplotlib outputs).*

## 20. Existing Figures
Figures exist under `07_Figures/` but programmatic generation artifacts (SHAP, calibration) were previously requested as simulated files for formatting.

## 21. Existing Tables
Tables inside LaTeX align perfectly with the paper, but diverge slightly from raw Python outputs (see Phase 24).

## 22. Open-Meteo Pipeline
**VERIFIED**. `src/realtime/open_meteo_client.py` and `predictor.py` successfully isolate Live/Operational mapping, utilizing `train_means.joblib` to map the API response.

## 23. Reproducibility Audit
**Score: MEDIUM**. 
The repository is fully contained with extensive data generation scripts. However, due to divergent metrics between the `evaluate_models.py` outputs and the final IEEE draft, a blind run will not reproduce the exact numeric table values in the PDF.

## 24. Paper-Code Consistency
| Claim | Project Evidence | Required Action |
|---|---|---|
| Heatwave LightGBM F1 = 0.8878 | `model_evaluation_results.csv` shows F1 = 1.000 | Retrain without implicit leakage or update Paper table. |
| Compound Stacking Ensemble | Not logged in `train_models.py` | Add StackingRegressor integration block. |
| GRU Benchmark F1 = 0.4343 | `model_evaluation_results.csv` shows F1 = 0.3418 | Correct paper values to match PyTorch output. |

## 25. Scientific Validity Review
- **Strong Points**: Excellent chronological splitting preventing massive temporal leakage. Advanced proxy removal for temperature thresholds.
- **Weaknesses**: The reported "Stacking Ensemble" operational baseline is entirely missing from the core empirical `train_models.py` logs, meaning its 0.3419 F1 score is untraceable.

## 26. Missing Evidence
- Stacking Ensemble serialization logic.
- Generated PR Curves.

## 27. Experiments Still Required
- Executing the explicit Meta-Learner Stack implementation.
- Investigating LightGBM Test 1.00 F1 score anomaly.

## 28. Exact Commands to Run
*See MANUAL_EXPERIMENT_COMMANDS.md*

## 29. Results to Show Guide
- **Total Samples**: 6,940
- **Train Bounds**: 2000-2012
- **Test Bounds**: 2016-2018
- **Verified Flood Positives**: 1,880
- **Verified Offline Deep Learning (GRU)**: 0.3418 F1 for Compound Analysis.

## 30. Publication-Critical Issues
The LaTeX paper cites metric values (Stacking F1 = 0.3419) that are fundamentally not represented in the repository's programmatic log outputs. 

## 31. Final Recommendation
**NEEDS MORE EXPERIMENTS.**
Before official publication, the user must reconcile the divergence between the static CSV empirical results and the LaTeX tables, specifically implementing the Stacking Ensemble and repairing the LightGBM deterministic test leakage.
