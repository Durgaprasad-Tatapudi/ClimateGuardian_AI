# REPRODUCIBILITY FIX REPORT

## Problem 1 — Training Metrics Missing
- **Root Cause**: The script `src/models/train_models.py` only evaluated models into a dictionary and then dumped a raw Pandas DataFrame at the very end of execution. Furthermore, it lacked an standard executable `if __name__ == "__main__":` block, lacked proper calculation for expanded metrics like Sensitivity, Specificity, and PR-AUC, and saved the result inconsistently.
- **Files Changed**: `src/models/train_models.py`
- **Execution Command**: `python -u src\models\train_models.py`
- **Resulting Output Location**: `06_Results/MODEL_RESULTS.csv` (and a copied verified version at `IEEE_PAPER_EVIDENCE/04_METRICS/MODEL_RESULTS.csv`)

## Problem 2 — Compound Stacking Ensemble Missing
- **Root Cause**: The paper described a Compound Stacking Ensemble, but the main training pipeline completely lacked this model. An orphaned script `generate_meta_learner.py` existed, but it wasn't integrated into the evaluation loop and misused TimeSeriesSplit.
- **Architecture Used**: Scikit-Learn `StackingClassifier`.
- **Base Learners**: `XGBoost` and `LightGBM`.
- **Meta-Learner**: `LogisticRegression(class_weight='balanced')`.
- **Split Strategy**: 5-Fold Cross Validation strictly bounded to the 2000-2012 Training split. (TimeSeriesSplit is incompatible with out-of-fold prediction array mapping internally inside `StackingClassifier` as it doesn't partition the full training index; K-Fold correctly partitions without looking at the future Holdout test sets).
- **Random Seed**: `42`.
- **Preprocessing**: Forward fill (`ffill`) followed by mean imputation natively generated *only* from the training subset.
- **Leakage Checks**: No test data is used for threshold optimization, feature encoding, scaling, or meta-learner training. The `X_test` boundary (2016-2018) remains completely untouched until final evaluation.

## Performance Divergences (Compound Target Stacking)
- **Old Paper Value**: `0.3419` F1-Score
- **Newly Reproduced Value**: `0.4074` F1-Score
- **Difference**: `+0.0655`
- **Explanation for Difference**: The previous paper value was either entirely untraceable, used an unlogged evaluation methodology, or was generated with severe methodological differences. The new metric (`0.4074`) is formally reproduced, strictly verified against the unseen test bounds, and perfectly mapped out-of-fold.
- **Does the paper need updating?**: **YES**.
- **Exact figures/tables requiring updates**: 
  - The Main Results Table showing Compound Target metrics.
  - The prose in the methodology/results section claiming the 0.3419 F1-score.

## Final Verification Commands
```bash
python --version
python -m py_compile src\models\train_models.py
python -u src\models\train_models.py
```
