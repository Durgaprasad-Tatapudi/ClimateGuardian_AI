# A. Traditional Flood/Hydrological Prediction
## Existing Work
Numerical and physically-based hydrological models have traditionally dominated flood prediction.
## Limitations/Gaps
Computationally expensive and often unsuited for rapid real-time multi-hazard alerting without massive HPC infrastructure.
## Relevance to ClimateGuardian AI
Motivates the shift towards data-driven, stateless ML approximations.
## References
[Mosavi2018]

# B. Machine Learning Flood Prediction
## Existing Work
Use of Random Forest, SVM, and ANNs for river stage and flood risk forecasting.
## Limitations/Gaps
Many studies evaluate on random cross-validation splits, risking temporal leakage.
## Relevance to ClimateGuardian AI
Supports the choice of Random Forest while emphasizing our chronological evaluation.
## References
[Mosavi2018], [Kaufman2012]

# C. Heatwave Prediction
## Existing Work
Deep learning (CNN/RNN) and gradient boosting methods applied to spatial-temporal meteorological data.
## Limitations/Gaps
Frequent target-proxy leakage where maximum temperature trivially drives the target definition.
## Relevance to ClimateGuardian AI
Justifies our methodology of strictly dropping the 4 proxy/deterministic features (like temperature_max) from the LightGBM input.
## References
[Chattopadhyay2020]

# D. Compound/Multi-Hazard Prediction
## Existing Work
Frameworks defining concurrent extremes (e.g., heavy precipitation plus extreme heat).
## Limitations/Gaps
Operational forecasting systems often predict hazards in isolation rather than probabilistically modeling compound states.
## Relevance to ClimateGuardian AI
Drives the explicit Compound target formulation.
## References
[Zscheischler2018], [Zscheischler2020]

# E. LSTM/GRU Sequence Models
## Existing Work
State-of-the-art for environmental time-series predictions.
## Limitations/Gaps
Requires stateful buffers (e.g., 14-day trailing sequence), complicating stateless REST API inference.
## Relevance to ClimateGuardian AI
Justifies why GRU is treated strictly as an offline benchmark, despite potentially higher F1 scores.
## References
[Kratzert2018], [Cho2014]

# F. Tree-Based Models
## Existing Work
Random Forest, XGBoost, and LightGBM dominate tabular environmental data competitions.
## Limitations/Gaps
Require feature engineering (lags, rolling stats) to capture temporal dynamics inherently caught by RNNs.
## Relevance to ClimateGuardian AI
Forms the backbone of the operational system (Flood RF, Heatwave LGBM, Compound XGB+LGBM).
## References
[Breiman2001], [Chen2016], [Ke2017]

# G. Explainable Climate ML
## Existing Work
Application of SHAP to interpret non-linear environmental predictions.
## Limitations/Gaps
SHAP provides feature dependence/attribution, not physical causality.
## Relevance to ClimateGuardian AI
Guides the non-causal interpretability of our models.
## References
[Lundberg2017]

# H. Probabilistic Calibration
## Existing Work
Platt scaling and isotonic regression to correct model confidence.
## Limitations/Gaps
Rarely applied consistently across multi-hazard ensembles.
## Relevance to ClimateGuardian AI
Our Stacking ensemble is evaluated with Brier scores before and after calibration.
## References
[Platt1999], [Zadrozny2002]

# I. Operational/Real-time Forecasting
## Existing Work
Stateless deployment of ML models using live API feeds.
## Limitations/Gaps
Bridging the gap between offline sequence models and live inference constraints.
## Relevance to ClimateGuardian AI
Explains the architectural choice of Stacking over GRU for the Compound hazard.
## References
[Wolpert1992], [OpenMeteo2023]

# J. Climate Data and Early Warning
## Existing Work
IPCC reports detailing the physical necessity of early warning mechanisms.
## Limitations/Gaps
Global urgency requires localized, rapid-inference approximations.
## Relevance to ClimateGuardian AI
Provides the global socio-scientific justification for the framework.
## References
[IPCC2021]
