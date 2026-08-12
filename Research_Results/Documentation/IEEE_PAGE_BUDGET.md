# IEEE PAPER STRUCTURE AND PAGE BUDGET (6-7 Pages Target)

This document provides a recommended spatial distribution and content outline designed to fit a standard double-column IEEE template.

## I. INTRODUCTION (0.75 Pages)
- **Context:** Rising frequency of climate-driven extreme weather and compound risks.
- **Problem Statement:** Real-world gap between historical ML accuracy and operational stateless predictability.
- **Contribution:** An operational, multi-hazard, early-warning framework preventing temporal leakage and preserving point-in-time readiness.

## II. RELATED WORK (0.5 Pages)
- Briefly cover classical hydro-meteorological forecasting limits.
- Discuss recent advances in deep learning (LSTMs, GRUs) and highlight their computational barriers in stateless REST architectures.

## III. DATA AND METHODOLOGY (2.0 Pages)
- **A. Data Collection:** Detail the 45-feature integration of climate events and hydrological states across 2000-2018.
- **B. Feature Engineering:** Explain the necessity of trailing memory (lags, rolling aggregates) and anomaly detection. 
- **C. Chronological Partitioning:** Explicitly outline the leakage-proof Train (2000-2012) / Val / Test (2016-2018) boundaries. 
- **D. Operational Architecture Constraints:** Contrast the 14-day trailing requirement of RNNs against the $O(1)$ independent execution time of Stacking Ensembles. 
- **E. Leakage Remediation:** Briefly justify the elimination of proxy variables (`temperature_max`, `hw_exceed`) for the heatwave subsystem.
- *Recommended inclusions: Table: Dataset Statistics, Table: Data Splits, Fig. 1 (Architecture).*

## IV. EXPERIMENTAL SETUP AND EVALUATION (1.5 Pages)
- **A. Offline Benchmarking:** Analyze baseline RandomForest, XGBoost, LightGBM, GRU, and Stacking.
- **B. Realtime Stacking Deployment:** Highlight how XGBoost and LightGBM probabilities mathematically feed the LogisticRegression Meta-Learner.
- **C. Ablation Studies:** Present the failure of static "Climate Only" configurations and the power of temporal lag infusion.
- *Recommended inclusions: Table: Model Performance, Fig. 2 (Flood Confusion Matrix), Fig. 3 (Heatwave ROC).*

## V. RESULTS AND DISCUSSION (1.5 Pages)
- **A. Multi-Hazard Performance:** Dissect F1 and ROC-AUC for Flood, Heatwave, and Compound targets. 
- **B. Lead-Time Resilience:** Analyze degradation trajectories from 1-day to 7-day warning horizons. (Highlight *Fig. 6: Lead-Time Performance*).
- **C. Explainability:** Use *Fig. 5 (SHAP)* to explore the physical intuition learned by the models. 
- **D. Calibration:** Show probabilistic trustworthiness improvements via Isotonic/Platt scaling using *Fig. 4*.
- *Recommended inclusions: Fig. 4 (Calibration), Fig. 5 (SHAP), Fig. 6 (Lead Time), Fig. 7 (Ablation).*

## VI. REALTIME OPERATIONAL PIPELINE (0.25 Pages)
- Briefly document the implementation of the live `LiveFeatureBuilder` extracting current parameters via Open-Meteo REST API, feeding identically scaled tensors into the frozen Stacking artifact.

## VII. CONCLUSION AND LIMITATIONS (0.5 Pages)
- Summarize operational success over purely historical accuracy metrics.
- Address missing variables (e.g., spatial gridded topologies).

## RECOMMENDED FIGURES (Max 7)
1. **Fig. 1:** Custom System Architecture Flow Diagram. (Must draw externally)
2. **Fig. 2:** `11_Confusion_Matrices_Norm_flood_target.png`
3. **Fig. 3:** `12_ROC_Curves_heatwave_target.png` 
4. **Fig. 4:** `14_Calibration_compound_target.png` 
5. **Fig. 5:** `16_SHAP_Summary_flood_target.png` 
6. **Fig. 6:** `17_Lead_Time_Performance_heatwave_target.png` 
7. **Fig. 7:** `18_Ablation_Study_compound_target.png` 

## RECOMMENDED TABLES (Max 4-5)
1. **Table I:** Dataset Statistics and Class Imbalance.
2. **Table II:** Chronological Train/Validation/Test Partitions.
3. **Table III:** Final Offline Test Performance (F1, ROC-AUC, etc).
4. **Table IV:** Final Operational Architecture Selection Rationale.
5. **Table V:** Feature Ablation F1-Scores.
