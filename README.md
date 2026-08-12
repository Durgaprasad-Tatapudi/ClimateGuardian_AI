# ClimateGuardian AI

A leakage-free multi-hazard climate risk prediction and early-warning system for Flood, Heatwave, and Compound hazards using machine learning, explainable AI, lead-time analysis, and probabilistic risk assessment.

## 1. Project Overview
ClimateGuardian AI is designed to integrate heterogeneous meteorological and hydrological data into actionable risk probabilities. It emphasizes strict out-of-sample temporal validation to prevent data leakage and ensure real-world operational viability.

## 2. Research Problem
Standard ML approaches in climate science often suffer from temporal leakage (e.g., k-fold cross-validation on time-series data or calculating thresholds using future test data). This project demonstrates a strictly chronological, causally robust architecture.

## 3. Main Features
- **Multi-Hazard Modeling**: Individual LightGBM/RandomForest pipelines for Floods and Heatwaves, unified by a Stacking Ensemble for Compound Risks.
- **Leakage-Free Temporal Splitting**: Strict chronological splits.
- **Probability Calibration**: Platt scaling and Isotonic regression for reliable warning thresholds.
- **Explainable AI (XAI)**: SHAP-based feature importance interpretation.
- **Lead-Time Analysis**: Performance degradation tracked up to 7-day horizons.

## 4. Architecture
The system consists of an offline scientific validation pipeline (which trains models on historical data) and a stateless real-time inference path.

## 5. Dataset
- Raw meteorological/hydrological data mapped to engineered features.
- See `data/processed/` for the final feature structures.
- **Train Period**: 2000–2012
- **Validation Period**: 2013–2015
- **Test Period**: 2016–2018

## 6. Models
Base Learners: LightGBM, XGBoost, Random Forest, Logistic Regression.
Meta Learner: Stacking Ensemble.
Benchmark: GRU (Offline Sequence Learning).

## 7. Leakage-Free Methodology
Target variables are strictly thresholded using parameters (e.g., 90th percentile temperatures) calculated **only** from the training partition. 

## 8. Final Results
The canonical scientific results are located in `results/canonical/`. These values correspond exactly to the published research paper.

## 9. Repository Structure
- `src/`: Source code (preprocessing, models, evaluation).
- `scripts/`: Verification scripts.
- `results/canonical/`: Frozen publication artifacts.
- `results/experiment_runs/`: Output directory for new local runs.
- `paper/`: IEEE LaTeX source code.
- `Paper_Figures/`: The canonical figures embedded in the paper.

## 10. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 11. Running Experiments
To run the full pipeline locally:
```bash
python run_full_pipeline.py
```
**Important**: The canonical results are frozen publication artifacts. Ordinary experiment runs will safely output to a timestamped folder inside `results/experiment_runs/` and will not overwrite the canonical publication results.

## 12. Reproducibility Verification
Run the verification scripts to ensure environment consistency:
```bash
python scripts/verify_reproducibility.py
python scripts/verify_paper_consistency.py
```

## 13. Paper Compilation
To compile the IEEE paper locally:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 14. License
MIT License.
