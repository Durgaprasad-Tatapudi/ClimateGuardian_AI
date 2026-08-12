# ClimateGuardian_AI: Research Results Summary

## 1. Research Objective
The goal was to build a rigorous, multi-horizon operational prediction system for three extreme weather hazards: **Floods, Heatwaves, and Compound events**. We aim to transcend simple heuristic thresholds by using historical ML algorithms decoupled into probability scores and validated risk thresholds, designed for integration with the real-time Open-Meteo forecasting API.

## 2. Datasets & Preprocessing
- **Sources**: NASA POWER (temperature/humidity/wind), CHIRPS (precipitation), ERA5-Land (runoff/soil moisture), Global Flood Database.
- **Preprocessing**: Data was localized to New Delhi coordinates, standardized to a daily frequency spanning 2000-2023. Missing values were median-imputed.
- **Feature Engineering**: We engineered temporal lags ($t-1$ to $t-14$), rolling windows (3d, 7d), physical anomaly baselines (calculated from a static 2000-2014 mean), and seasonality vectors (sin/cos).

## 3. Modeling Methodology
- **Validation**: Strict temporal split (Train: 2000-2014, Validate: 2015-2018, Test: 2019-2023).
- **Algorithms**: Evaluated Logistic Regression, RandomForest, LightGBM, XGBoost, GRU, and LSTM.
- **Mitigation of Imbalance**: Class weights (`class_weight='balanced'`) were applied. F1 and PR-AUC were prioritized over raw accuracy.
- **Stacking**: A Stacking Ensemble (Meta-Learner) was trained on cross-validated validation fold predictions.
- **Calibration**: Isotonic Regression was utilized to transform raw tree-based output into reliable risk probabilities.

## 4. Final Empirical Results (Test Set: 2019-2023)
*Note: The numbers below are extracted automatically from `FINAL_MODEL_COMPARISON.csv`.*

### Best Model by Hazard
- **Flood**: RandomForest (F1: 0.471, PR-AUC: 0.355)
- **Heatwave**: LightGBM (F1: 1.000, PR-AUC: 1.000)
- **Compound**: StackingEnsemble (F1: 0.487, PR-AUC: 0.370)

### Key Metric Averages
- **Precision**: Flood 0.355, Heatwave 1.000, Compound 0.392
- **Recall (Sensitivity)**: Flood 0.698, Heatwave 1.000, Compound 0.644
- **F1 Score**: Flood 0.471, Heatwave 1.000, Compound 0.487
- **PR-AUC**: Flood 0.355, Heatwave 1.000, Compound 0.370

## 5. Ablation & Lead-Time Findings
- **Ablation**: Removing lagged meteorological features severely degrades short-term prediction performance.
- **Lead Time**: As forecast horizons extend beyond 5 days, prediction confidence bounds widen, and recall degrades due to chaotic atmospheric divergence.

## 6. Open-Meteo Integration
The pipeline achieved full operational capability:
1. Hourly API inputs are mapped to historical daily features.
2. The exact `StandardScaler` fitted on the 2000-2014 data normalizes live inputs.
3. The frozen historical models output probabilities seamlessly.

## 7. Limitations & Future Work
1. **Spatial Generalization**: The model currently focuses on a single coordinate region. Future work should implement spatial convolutions (GNNs/CNNs).
2. **Feature Exhaustion**: Additional explicit indices (e.g., ENSO, MJO) could improve the prediction horizon beyond 7 days.
