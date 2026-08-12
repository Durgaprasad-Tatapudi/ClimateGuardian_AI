# Results Provenance

This document maps the metrics, tables, and figures stated in the IEEE paper to their canonical origins in this repository.

### Table III: Final Offline Test Performance
- **Source**: `results/canonical/MODEL_RESULTS.csv`
- **Mapping**: 
  - Compound GRU values (Accuracy, Precision, Recall, F1) are drawn directly from the `compound_target` row for `GRU`.
  - LightGBM and Random Forest values correspond to `heatwave_target` and `flood_target`.

### Table IV: Feature Ablation Results
- **Source**: `results/canonical/ABLATION_RESULTS.csv`
- **Mapping**: Compound F1-scores for "Climate Only", "Hydro Only", and "Full Features" correspond to this file's test results.

### Lead-Time Analysis (Fig 4)
- **Source**: `results/canonical/LEAD_TIME_RESULTS.csv`
- **Mapping**: Evaluation of F1-scores extending to the 7-day horizon.

### Calibration (Fig 6)
- **Source**: `results/canonical/CALIBRATION_RESULTS.csv`
- **Mapping**: Comparison of Brier scores (Uncalibrated, Platt, Isotonic).

### Paper Figures
- **Source Directory**: `Paper_Figures/`
- **Mapping**:
  - `Fig_01_Architecture.pdf` -> Diagram
  - `Fig_02_Flood_Confusion_Matrix.png` -> Test set evaluation.
  - `Fig_03_Heatwave_ROC.png` -> Test set evaluation.
  - `Fig_04_Heatwave_LeadTime.png` -> Derived from `LEAD_TIME_RESULTS.csv`.
  - `Fig_05_Compound_Ablation.png` -> Derived from `ABLATION_RESULTS.csv`.
  - `Fig_06_Compound_Calibration.png` -> Derived from `CALIBRATION_RESULTS.csv`.
  - `Fig_07_Flood_SHAP.png` -> XAI Output.

### Textual Claims
All heatwave accuracy assertions (e.g. 0.9790) align seamlessly with `MODEL_RESULTS.csv`.
