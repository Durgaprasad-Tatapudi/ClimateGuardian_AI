# RESEARCH EVIDENCE MATRIX

| Claim | Evidence File | Evidence Type | Actual Value | Verified? | Paper Section | Confidence | Notes |
|---|---|---|---|---|---|---|---|
| **6,940 total daily records** | `03_Features/master_features.csv` | Empirical Row Count | 6,940 rows | **YES** | Dataset | High | Extracted via pandas length check. |
| **45 input features** | `03_Features/master_features.csv` | Empirical Column Count | 46 columns (incl date) -> 45 features | **YES** | Dataset | High | Extracted via pandas shape check. |
| **Time Period 2000-2018** | `03_Features/master_features.csv` | Datetime Min/Max | 2000-01-01 to 2018-12-31 | **YES** | Dataset | High | Strict chronological bounds confirmed. |
| **Flood: 1,880 Positives** | `04_Labels/flood_labels.csv` | Sum of binary target | 1,880 | **YES** | Labels | High | Extracted via pandas sum(). |
| **Heatwave: 510 Positives** | `04_Labels/heatwave_labels.csv` | Sum of binary target | 510 | **YES** | Labels | High | Extracted via pandas sum(). |
| **Compound: 255 Positives** | `04_Labels/compound_labels.csv` | Sum of binary target | 255 | **YES** | Labels | High | Extracted via pandas sum(). |
| **Train/Val/Test Split** | `src/models/train_models.py` | AST Code parsing | Train: 2000-2012 (4,749), Val: 2013-2015 (1,095), Test: 2016-2018 (1,096) | **YES** | Methodology | High | Chronological isolation is strictly enforced in code. |
| **Preprocessing leakage free** | `src/models/train_models.py` | AST Code parsing | `StandardScaler.fit(X_train)` only | **YES** | Methodology | High | Scaler explicitly fitted only on Train split. |
| **Heatwave Proxy Exclusion** | `src/models/train_models.py` | AST Code parsing | `restricted_cols = [c for c in X.columns if not ('temperature_max' in c...)]` | **YES** | Feature Engineering | High | Deterministic feature leakage successfully blocked in training code. |
| **Flood F1 = 0.4686 (RF)** | `06_Results_corrected/model_evaluation_results.csv` | CSV Result logs | RF F1: 0.4709 | **MISMATCH** | Results | High | Paper slightly under-reports the actual Random Forest test result (0.4709 > 0.4686). |
| **Heatwave F1 = 0.8878 (LGBM)** | `06_Results_corrected/model_evaluation_results.csv` | CSV Result logs | LGBM F1: 1.000 | **MISMATCH** | Results | High | Paper reports 0.8878, but code evaluates at 1.000 on test split (suggests deterministic leakage in actual LGBM execution). |
| **Compound F1 = 0.3419 (Stacking)** | `06_Results_corrected/model_evaluation_results.csv` | CSV Result logs | LGBM F1: 0.4117 (Stacking not logged) | **NOT VERIFIED** | Results | Low | The actual Stacking ensemble implementation is missing from `train_models.py` results. |
| **GRU Compound F1 = 0.4343** | `06_Results_corrected/model_evaluation_results.csv` | CSV Result logs | GRU F1: 0.3418 | **MISMATCH** | Results | High | Paper significantly over-reports the GRU capability compared to empirical logs (0.3418). |
| **Stateless Open-Meteo Inference** | `src/realtime/open_meteo_client.py` | API code inspection | Implemented | **YES** | Operational | High | Realtime JSON to feature-vector pipeline exists. |
