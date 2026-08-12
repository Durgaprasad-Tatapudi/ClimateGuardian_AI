# Final Leakage Audit Report

## Objective
The purpose of this audit is to guarantee that the machine learning models developed for the ClimateGuardian_AI project strictly avoid data leakage. Leakage occurs when a model is inadvertently provided with information about the target variable or future states during the training process, leading to artificially inflated performance metrics.

## 1. Temporal Split Violations
**Audit Result**: CLEAR
- The dataset is strictly partitioned temporally. Data from `2000-01-01` to `2014-12-31` is used for **Training**.
- Data from `2015-01-01` to `2018-12-31` is used for **Validation** (early stopping, hyperparameter tuning, stacking meta-features).
- Data from `2019-01-01` to `2023-12-31` is used exclusively for **Testing**.
- By adhering to a strict chronological split without shuffling, the models never peek into the future.

## 2. Feature Leakage (Target / Future Information)
**Audit Result**: CLEAR
- **Lag Features**: Variables such as `rainfall_lag1`, `runoff_lag2` are strictly constructed by shifting the data positively, ensuring time step $t$ only uses information from time steps $t-1, t-2, \dots$.
- **Rolling Windows**: Rolling aggregations (`rolling_3d`, `rolling_7d`) use right-aligned windows (e.g., `min_periods=1, closed='right'`) meaning they aggregate from $t-w$ up to exactly $t$, with no forward-looking components.
- **Anomalies**: Baseline metrics (`temperature_anomaly`, `soil_moisture_anomaly`) are computed using the **static mean and standard deviation of the Training partition only** (2000-2014). This prevents the leakage of future baseline shifts into the training set.

## 3. Scaler and Imputer Leakage
**Audit Result**: CLEAR
- The `StandardScaler` and any imputation strategies are fitted strictly on `X_train`. 
- `X_val` and `X_test` are standardized using the `.transform()` method of the already fitted scaler, guaranteeing that the distribution statistics of the test set do not influence the preprocessing of the training set.
- This logic is replicated perfectly in `live_feature_builder.py` for operational real-time inference.

## 4. Stacking Leakage
**Audit Result**: CLEAR
- To train the Stacking Ensemble Meta-Learner, we utilized `TimeSeriesSplit` cross-validation on the combined Train/Validation set. The meta-features (predictions of base models) for fold $k$ were generated purely by base models trained on folds $1 \dots k-1$.
- At no point was the meta-learner trained on predictions generated from overlapping data, preserving the out-of-sample prediction integrity.

## 5. Calibration Leakage
**Audit Result**: CLEAR
- Probability calibration (Platt Scaling / Isotonic Regression) was fitted exclusively on the `Validation` dataset outputs, not the test outputs.

## Conclusion
The ClimateGuardian_AI modeling pipeline is architecturally sound and free from future, target, and preprocessing leakage. Metrics reported on the Test set accurately reflect true generalization performance.
