# PAPER CLAIMS AUDIT

*This document ensures the scientific integrity of the language used in the IEEE paper. It dictates what claims are mathematically supported by the ClimateGuardian AI codebase and what phrases must be omitted.*

## 1. SUPPORTED CLAIMS
*These claims are thoroughly backed by the executed codebase, generated metrics, and the reproducibility audit.*

- **"Operational/Realtime capable"**: The system architecture incorporates live Open-Meteo REST endpoints, bypassing static historical datasets, explicitly rendering it operational.
- **"Leakage-free evaluation"**: Strict chronological masks (2000-2012 Train, 2013-2015 Validation, 2016-2018 Test) and target proxy removals (`temperature_max`, `hw_exceed`) guarantee an honest evaluation landscape.
- **"Multi-hazard prediction"**: Models successfully and independently calculate separate risk profiles for Flood, Heatwave, and concurrent Compound states.
- **"Temporal variables drive compound risk predictability"**: Empirically proven by the Ablation study, where F1-scores doubled upon injecting temporal and lag variables over static climatic variables alone.
- **"Calibration improves probabilistic reliability"**: Mathematically supported by Brier score reductions through Isotonic and Platt scaling (Compound baseline dropping from 0.2013 to 0.0338).
- **"Maintains predictive strength up to 7 days (Heatwave)"**: The lead-time degradation metrics explicitly prove that heatwave recognition remains above 0.60 F1 out to 7 days ahead.

## 2. PARTIALLY SUPPORTED CLAIMS (Require Nuance)
*These claims have merit but must be heavily caveated with context to avoid misleading the IEEE reviewers.*

- **"Highly accurate"**: While the Heatwave model achieves exceptional accuracy (F1=0.8878), predicting stochastic events like Floods (F1=0.4686) and Compound Risks (F1=0.3419) requires tempered language. 
  - *Better terminology:* "Statistically robust relative to extreme class imbalance," or "Provides significant discriminatory power (ROC-AUC > 0.70)."
- **"Stacking Ensemble is the optimal architecture"**: While selected for operational deployment due to independent O(1) inference, the GRU sequence model strictly outperformed it on offline metrics (F1=0.4343). 
  - *Better terminology:* "The Stacking Ensemble represents the optimal balance of predictive reliability and stateless operational feasibility."
- **"SHAP identifies causation"**: SHAP demonstrates feature correlation and directional importance to the model's logic, but does NOT prove physical meteorological causation.
  - *Better terminology:* "SHAP values demonstrate the model's internal dependency hierarchy."

## 3. CLAIMS TO AVOID
*Using these phrases compromises the scientific validity of the paper and contradicts the empirical limits of the codebase.*

- **"100% accurate" / "Perfect prediction"**: False. All models experience precision/recall trade-offs.
- **"Best model in every category"**: False. RandomForest is best for Flood, LightGBM for Heatwave, GRU for offline Compound, and Stacking for operational Compound.
- **"Novel invention of RandomForest/LightGBM/XGBoost/SHAP/Stacking"**: These are industry-standard tools. The novelty lies in the *hydro-climate integration*, the *leakage-free methodology*, and the *operational pipeline combination*, not the algorithms themselves.
- **"The model learns to predict weather"**: False. The model predicts the *hazard labels* driven by historical anomalies; it leverages external API providers (Open-Meteo) for the actual weather prediction physics. 
- **Conflating live inference with accuracy**: Do NOT present output from the live REST API script (`run_live_inference.py`) as historical proof of accuracy. Live output represents real-world execution capacity, not a historical accuracy percentage.
