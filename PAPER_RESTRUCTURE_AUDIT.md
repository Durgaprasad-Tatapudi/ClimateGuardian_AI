# PAPER RESTRUCTURE AUDIT

## 1. Current Paper Structure
- Abstract
- Keywords
- I. Introduction
- II. Related Work
- III. Data and Methodology
- IV. Experimental Setup and Evaluation
- V. Results and Discussion
- VI. Realtime Operational Pipeline
- VII. Conclusion and Limitations
- References

## 2. Current Title
`ClimateGuardian AI: A Leakage-Aware Multi-Hazard Climate Risk Prediction Framework with Stateless Real-Time Inference`

## 3. Current Abstract
The current abstract is a placeholder/draft abstract discussing the 45-feature dataset, chronological split, leakage remediation, specific algorithms (Random Forest, LightGBM, Stacking, GRU), and the Open-Meteo realtime inference. According to the audit, the submitted project abstract was replaced by a different research abstract.

## 4. Current Table Inventory
- Table I: Dataset Statistics (2000–2018)
- Table II: Chronological Data Partition
- Table III: Final Offline Test Performance (2016–2018)
- Table IV: Operational Model Selection Rationale
- Table V: Feature Ablation Results for the Compound-Risk Target

## 5. Current Figure Inventory
- Fig. 1: System Architecture (Fig01_System_Architecture.pdf)
- Fig. 2: Flood Confusion Matrix (Fig02_Flood_Confusion_Matrix.png)
- Fig. 3: Heatwave ROC Curve (Fig03_Heatwave_ROC.png)
- Fig. 4: Compound Calibration (Fig04_Compound_Calibration.png)
- Fig. 5: Flood SHAP (Fig05_Flood_SHAP.png)
- Fig. 6: Heatwave Lead-Time Performance (Fig06_Heatwave_LeadTime.png)
- Fig. 7: Compound Ablation Study (Fig07_Compound_Ablation.png)

## 6. Current Reference Inventory
17 total verified references utilized in the bibliography.

## 7. Current Author Block
Contains placeholder metadata (Author 1 Name, Institution Placeholder, Guide Name).

## 8. Structural Problems Identified
1. Title does not properly match the actual research scope.
2. Placeholder author metadata exists.
3. Batch/roll-number/signature content appears in the first page and should not be part of the IEEE research paper.
4. The submitted project abstract was replaced by a different research abstract.
5. Figures are being pushed toward the end of the PDF instead of appearing near their relevant discussion.
6. Tables are not consistently positioned near the sections that discuss them.
7. Some tables are visually compressed or poorly structured.
8. Figure/table numbering and placement need a complete audit.
9. The paper must read as one complete finished research article.
10. No section should look unfinished.
11. No figure/table dump at the end.
12. The final paper should target approximately 8–10 pages naturally, without artificial padding.

## 9. Files Scheduled for Modification
- `main.tex`
- `sections/introduction.tex`
- `sections/related_work.tex`
- `sections/methodology.tex`
- `sections/experimental_setup.tex`
- `sections/results_discussion.tex`
- `sections/realtime.tex`
- `sections/conclusion.tex`

## 10. Integrity Confirmation
- No scientific results were modified.
- No model artifacts were modified.
- No new or fabricated metrics were introduced.
- The existing research and offline benchmark isolation remains strictly protected.
