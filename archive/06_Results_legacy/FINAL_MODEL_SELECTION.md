# Final Model Selection

## Target: flood_target
**Selected Model:** RandomForest

### Justification:
- Highest combined F1 and ROC-AUC score.
- F1-Score: 0.4710
- ROC-AUC: 0.7202
- PR-AUC: 0.3550
- Recall: 0.6977

---

## Target: heatwave_target
**Selected Model:** LightGBM

### Justification:
- Highest combined F1 and ROC-AUC score.
- F1-Score: 1.0000
- ROC-AUC: 1.0000
- PR-AUC: 1.0000
- Recall: 1.0000

---

## Target: compound_target
**Selected Model:** StackingEnsemble

### Justification:
- Highest combined F1 and ROC-AUC score.
- F1-Score: 0.4874
- ROC-AUC: 0.9552
- PR-AUC: 0.3699
- Recall: 0.6444

---
