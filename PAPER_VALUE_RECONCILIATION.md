# PAPER VALUE RECONCILIATION

This table reconciles the values currently claimed in the LaTeX IEEE Paper against the **scientifically clean and verified** experiment execution.

| Claimed Metric | Paper Value | Old Logged Value | Clean Verified Value | Source File | Status | Recommended Final Value | Reason for Change |
|---|---|---|---|---|---|---|---|
| **Flood F1 (Random Forest)** | `0.4686` | `0.4709` | **`0.4169`** | `FINAL_VERIFIED_METRICS.csv` | MISMATCH | **`0.4169`** | Removed `bfill()` temporal leakage and strictly split dataset before processing. |
| **Heatwave F1 (LightGBM)** | `0.8878` | `1.000` | **`0.7033`** | `FINAL_VERIFIED_METRICS.csv` | LEAKAGE DETECTED | **`0.7033`** | The 1.00 score was caused by target leakage (temperature features directly predicting threshold). The `0.7033` value is the true, proxy-free prediction. |
| **Compound F1 (Stacking)** | `0.3419` | `NOT IMPLEMENTED` | **`0.4072`** | `FINAL_VERIFIED_METRICS.csv` | MISMATCH | **`0.4072`** | A rigorous Stacking Ensemble was implemented from scratch using strictly Out-of-Fold training to prevent meta-learner leakage. |
| **Compound F1 (GRU)** | `0.4343` | `0.3418` | **`0.4795`** | `FINAL_VERIFIED_METRICS.csv` | MISMATCH | **`0.4795`** | Strictly chronological PyTorch execution yielded better generalization on the unseen test boundary than previously logged. |
| **Flood ROC-AUC (Random Forest)** | `0.7184` | `0.7202` | **`0.6996`** | `FINAL_VERIFIED_METRICS.csv` | MISMATCH | **`0.6996`** | Resolved forward-temporal leakage. |
| **Heatwave Threshold Calculation** | N/A | `2000-2014` | **`2000-2012`** | `train_models_clean.py` | METHODOLOGY ERROR | **`2000-2012`** | The threshold calculation leaked into the 2013-2015 validation bounds. It is now strictly confined to the Train split. |

## Next Actions
All "Recommended Final Values" must be directly transcribed into the `ClimateGuardian_IEEE_Paper/main.tex` document to ensure absolute scientific validity prior to publication.
