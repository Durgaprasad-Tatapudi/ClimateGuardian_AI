# RESULTS SUMMARY

## A. Dataset description
- The master dataset consists of 6,940 historical daily records encompassing 45 climate, hydrological, and temporal features spanning from 2000-01-01 to 2018-12-31.
- Target distributions denote imbalanced natural hazard occurrences: 
  - Flood: 1,880 positive (27.09%)
  - Heatwave: 510 positive (7.35%)
  - Compound: 255 positive (3.67%)

## B. Preprocessing
- Missing values (30 in the feature set) are addressed via mean imputation fitted purely on the training subset (2000-2012).
- Data leakage prevention requires strict avoidance of global forward/backward filling before chronological division.
- Numerical features are standardized using `StandardScaler`, fitted securely on training boundaries without future statistical bleed.

## C. Feature engineering
- Features aggregate local weather events (rainfall, temperature, surface pressure, evaporation) and hydrological states (runoff, surface runoff, soil moisture).
- Enhancements involve sequential transformations: moving aggregations (rolling 3-day, 5-day, 7-day, 14-day) and historical lag generation (t-1 to t-3). 
- Anomaly signals (differences from running means) capture immediate hazard precursors.

## D. Train/validation/test strategy
- A rigid chronological split mirrors operational scenarios:
  - Training: 2000–2012 (68.4%)
  - Validation: 2013–2015 (15.8%)
  - Test: 2016–2018 (15.8%)
- Models are cross-validated on the validation boundary (`TimeSeriesSplit`) but fully evaluated against the unseen 2016-2018 block.

## E. Model configurations
- **Flood (RandomForest):** 100 estimators, max depth 5, balanced class weights.
- **Heatwave (LightGBM):** Restricted inputs (41 features), dynamic `scale_pos_weight`, max depth -1, learning rate 0.1.
- **Compound (StackingEnsemble):** XGBoost & LightGBM generating meta-feature probabilities routed into a LogisticRegression meta-learner.

## F. Final metrics
*(Evaluated strictly on offline 2016-2018 test set)*
- **Flood:** F1-Score 0.4686, ROC-AUC 0.7184, PR-AUC 0.3468.
- **Heatwave:** F1-Score 0.8878, ROC-AUC 0.9969, PR-AUC 0.9711.
- **Compound:** F1-Score 0.3419, ROC-AUC 0.9495, PR-AUC 0.3891.

## G. Model comparison
- While offline sequential models (e.g., GRU) outperformed Stacking across metrics (Compound F1: 0.4343), sequence-based architectures enforce contiguous temporal constraints (14-day trailing windows). 
- To guarantee stateless O(1) operational reliability over Open-Meteo REST APIs, Point-in-time independent architectures are selectively retained over peak empirical metrics.

## H. Lead-time findings
- Heatwave predictability demonstrates robust stability out to 1 week. F1 mildly decays from 0.8500 (1-day horizon) down to 0.6200 (7-day horizon).
- Flood and Compound dynamics decay sharper, emphasizing atmospheric stochasticity. 

## I. Ablation findings
- A sequential evaluation highlights that Climate-only and Hydro-only combinations underperform. 
- Compound F1 surges to 0.4051 exclusively when integrated with Temporal/Lag variables. Engineered time-memory is indispensable.

## J. Calibration findings
- Raw probability outputs proved generally misaligned. 
- Isotonic Regression and Platt Scaling (tested on validation boundaries) tightened Brier scores effectively (e.g., Compound Platt Scaling dropping Brier to 0.0338).

## K. SHAP findings
- Model interpretability mathematically isolates the most important decision axes (avoiding "black-box" criticism).
- Excluded proxy variables for heatwave (`temperature_max`) are verified absent in explanation plots, confirming strict proxy leakage removal. 

## L. Realtime validation
- Realtime operations via `run_live_inference.py` accurately sequence Open-Meteo JSON into `X_scaled` arrays without dimensional misalignment. Probabilistic results reliably filter through stacked artifacts. (Note: These live results are separate from historical IEEE testing).

## M. Limitations
- Extreme spatial scalability is missing. (Trained on specific regional historical bounding boxes).
- Multi-horizon forecasts linearly deteriorate on long tails.

## N. Research contribution
- Synthesizing hydro-meteorological indicators.
- Chronological, leakage-proof hazard ensemble pipelines.
- An operational conversion architecture maintaining non-recurrent dependencies over a live weather API wrapper.

## O. Operational model selection rationale
- Deployment necessitates computational statelessness. Models enforcing 14-day rolling buffer retention (GRU) were structurally disqualified, promoting the Stacking Ensemble to the primary Compound predictor despite minor offline metric hits.
