# EXPERIMENT STATUS

## 1. VERIFIED RESULTS
- **Clean Temporal Preprocessing**: Successfully implemented. Future data leakage from `bfill()` has been eliminated.
- **Strict Target Splitting**: Heatwave 90th percentile threshold is now derived **exclusively** from the 2000-2012 training split, preventing validation/test boundary violations.
- **Heatwave Proxy Purge**: Temperature proxies deterministically predicting heatwaves were purged. The true test F1 score for Heatwave LightGBM is **0.7033**.
- **Compound Stacking Ensemble**: A rigorously out-of-fold `StackingClassifier` was implemented. It successfully evaluated at **0.4072 F1**.
- **Deep Learning (GRU)**: Achieves **0.4795 F1** for the Compound Risk task under strict chronological limits.

## 2. INVALID RESULTS
- **Heatwave LightGBM (1.000 F1)**: The original project output was utterly invalid due to target leakage (temperature features perfectly proxying the target label definition).
- **Previous `bfill()` operations**: They leaked future context backward in time.

## 3. SUSPICIOUS RESULTS
- The paper reported a **0.3419 F1** for the Stacking Ensemble, yet no serialization or configuration of this ensemble existed in the primary logging pipeline. The new verified value is **0.4072**.
- The paper reported a **0.4343 F1** for the GRU model. The verified output actually scored higher (**0.4795**), meaning the paper likely relied on an older seed or non-deterministic early stopping checkpoint.

## 4. RESULTS REQUIRING RERUN
*None remaining.* The clean pipeline (`train_models_clean.py`) successfully generated all deterministic, temporally isolated artifacts inside `06_Results/verified_clean`.

## 5. PAPER VALUES THAT MUST CHANGE
All empirical evaluation metrics (F1, ROC-AUC, Brier) in the paper **must** be updated to reflect the canonical `FINAL_VERIFIED_METRICS.csv` table.
- Heatwave F1 MUST drop from 0.8878 to 0.7033.
- Compound Stacking F1 MUST change to 0.4072.
- GRU Compound F1 MUST change to 0.4795.

## 6. PAPER VALUES THAT CAN REMAIN
- Dataset row count (6,940).
- Time period definitions (2000-2018).
- Train/Val/Test temporal boundaries (2000-2012, 2013-2015, 2016-2018).
- Model architectural claims (using LightGBM, Random Forest, Stacking, GRU, LSTM).

## 7. EXPERIMENTS STILL REQUIRED
*None.* The `train_models_clean.py` script has satisfied all empirical prerequisites.

## 8. PUBLICATION BLOCKERS
The LaTeX paper still contains the old, leaked, or unverified numerical data. The paper prose must be updated to integrate the verified numbers before submission.

## FINAL DECISION
**READY FOR PAPER UPDATE**
