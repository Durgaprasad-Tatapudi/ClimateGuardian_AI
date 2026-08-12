# IEEE Methods Notes
**Title**: ClimateGuardian_AI: Operationalizing Historical Climate Data for Multi-Hazard Prediction
**Authors**: [Author Names]

## Methodology
The framework formulates extreme weather prediction—specifically floods, heatwaves, and their compound co-occurrence—as a set of supervised multi-horizon binary classification tasks. A sliding window architecture with lagged features extracts short-to-medium range meteorological context (from $t-14$ to $t$). A multi-model ensemble approach is adopted to establish robust baselines against inherent class imbalances in extreme event datasets.

## Data Preprocessing
- **Sources**: Integration of NASA POWER (meteorology), CHIRPS (precipitation), ERA5-Land (hydro-thermodynamics), and Global Flood Database (GFD).
- **Temporal Alignment**: All datasets are synchronized to a daily interval from Jan 1, 2000 to Dec 31, 2023. Missing historical values are imputed using median-filling derived solely from the training distribution.
- **Normalization**: Continuous predictors are standardized ($\mu=0, \sigma=1$) via a scaler fit exclusively on the chronological training split (2000–2014) to prevent information leakage.

## Model Architecture
- **Base Learners**: A suite of tree-based models (RandomForest, XGBoost, LightGBM) and deep temporal sequences (LSTM, GRU).
- **Stacking**: For complex hazard interplay, a meta-learning ensemble was configured utilizing out-of-fold probability predictions from XGBoost and LightGBM base models, generalized via a Logistic Regression meta-learner.
- **Calibration**: Uncalibrated tree-based probabilities were adjusted using Isotonic Regression and Platt Scaling fitted on the validation set, transforming outputs into reliable risk quotients.

## Evaluation Metrics
Evaluation emphasizes metrics invariant to extreme class imbalance:
- **Primary**: Precision-Recall AUC (PR-AUC), F1-Score, and Matthews Correlation Coefficient (MCC).
- **Secondary**: Receiver Operating Characteristic AUC (ROC-AUC), Sensitivity (Recall), Specificity.
- **Probabilistic**: Brier Score.

## Experimental Setup
- **Splits**: Strict temporal boundaries (Train: 2000-2014, Validate: 2015-2018, Test: 2019-2023).
- **Hyperparameters**: Tree-based models leverage `class_weight='balanced'` and `scale_pos_weight` heuristics to mitigate negative majority domination. Deep learning sequence models utilize a batch size of 64 and a sequence length of 14, optimized via Adam over 50 epochs with Early Stopping (patience=10).

## Results Interpretation
Predictions are decoupled into raw probabilities and operational risk thresholds. Thresholds (Low, Medium, High) are deterministically bound to the 90th/95th percentile activation densities evaluated purely on the validation partition. Statistical superiority between architectures is assessed utilizing McNemar's Test for matched-pair predictions.

## Limitations
- **Spatial Granularity**: Current models perform aggregate national-level classification, lacking high-resolution regional masking.
- **Data Proxy Assumptions**: Surface runoff relies on modeled estimates from global reanalysis (ERA5) which may deviate from actual physical gauge measurements.
