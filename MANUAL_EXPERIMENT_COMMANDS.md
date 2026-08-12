# MANUAL EXPERIMENT COMMANDS

## EXPERIMENTS I SHOULD RUN BEFORE SHOWING THIS TO MY GUIDE

### 1. Fix LightGBM Heatwave Leakage Check
**Why it is needed:** The PyTorch/Scikit-learn logs evaluate the LightGBM Heatwave model to a perfect `1.0` F1 Score. This typically indicates a deterministic data leakage, where the model accidentally relies on a feature inextricably tied to the target (e.g. `hw_exceed`).
**Exact command to run:**
```bash
.venv\Scripts\python.exe src/models/train_models.py
```
*(Before running, ensure you manually verify `restricted_cols` inside `train_models.py` effectively purged ALL rolling temperature variables for the LightGBM step.)*
**Expected output:** A printed logging sequence showing ML training followed by DL training.
**Metrics to record:** Heatwave F1 score.

### 2. Operational Stacking Meta-Learner Generation
**Why it is needed:** The final IEEE paper formally reports a "Stacking Ensemble" with `F1=0.3419`. However, the core `train_models.py` logic only exports `RandomForest`, `LightGBM`, `XGBoost`, and Deep Learning isolated predictors.
**Exact command to run:**
```bash
.venv\Scripts\python.exe generate_meta_learner.py
```
*(Note: A script `generate_meta_learner.py` exists in your repository, but was bypassed by `train_models.py`. You must run it manually).*
**Expected output:** Serialized `StackingRegressor` or `VotingClassifier` outputted to `05_Models_corrected/`.
**Metrics to record:** Compound F1, ROC-AUC.

### 3. Generate Visual Artifacts for Thesis Presentation
**Why it is needed:** The presentation requires tangible visual evidence of the model's performance (Confusion Matrices and PR curves) to justify operational boundaries.
**Exact command to run:**
```bash
.venv\Scripts\python.exe src/evaluation/evaluate_models.py
```
**Expected input:** The `.joblib` files within `05_Models_corrected/` and test samples from `03_Features/`.
**Expected output:** `.png` files saved to `07_Figures/` (Specifically: `Fig02_Flood_Confusion_Matrix.png`).
**Where the output will be saved:** `c:\Users\durga\OneDrive\Desktop\ClimateGuardian_AI\07_Figures\`

## KEY RESULTS TO SHOW TO GUIDE

| Metric | Verified Value |
|---|---|
| Dataset | India CHIRPS / ERA5 Aggregate |
| Total samples | 6,940 |
| Features | 45 |
| Time period | 2000-01-01 to 2018-12-31 |
| Flood label count | 1,880 |
| Heatwave label count | 510 |
| Compound label count | 255 |
| Train samples | 4,749 (2000-2012) |
| Validation samples | 1,095 (2013-2015) |
| Test samples | 1,096 (2016-2018) |
| Flood model | Random Forest |
| Flood F1 | 0.4709 |
| Heatwave model | LightGBM |
| Heatwave F1 | 1.000 (Requires leakage review) |
| Compound model | Stacking / LightGBM |
| Compound F1 | 0.4117 (Best Empirical ML Model) |
| GRU benchmark | 0.3418 (Compound F1) |
| ROC-AUC | ~ 0.94 - 0.99 Range |
| Open-Meteo operational capability | Stateless live integration verified. |
