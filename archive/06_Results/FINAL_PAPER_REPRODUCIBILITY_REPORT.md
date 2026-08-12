# Final Paper Reproducibility Report

## Experimental Commands
- **Training and Evaluation Command**: `python src/run_full_pipeline.py` (Executes the end-to-end extraction, engineering, and cross-validated training/stacking pipeline)
- **Result Generation Command**: Evaluated transparently within `src/run_full_pipeline.py` using canonical metrics.
- **Figure Generation Command**: The figures were generated using `generate_final_tables.py` and the plotting routines in `src/run_full_pipeline.py`.

## Result Artifacts
- **Primary Metrics CSV**: `C:\Users\durga\OneDrive\Desktop\ClimateGuardian_AI\06_Results\MODEL_RESULTS.csv`
- **Ablation Metrics CSV**: `C:\Users\durga\OneDrive\Desktop\ClimateGuardian_AI\06_Results\ABLATION_RESULTS.csv`

## PDF Compilation
- **Paper Compilation Commands**: 
  1. `C:\Users\durga\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe main.tex`
  2. `C:\Users\durga\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe main`
  3. `C:\Users\durga\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe main.tex`
  4. `C:\Users\durga\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe main.tex`
- **Final PDF Path**: `C:\Users\durga\OneDrive\Desktop\ClimateGuardian_AI\ClimateGuardian_IEEE_Paper\main.pdf`

## Implementation and Consistency Status
- **Compound Stacking Implementation Status**: The Logistic Regression Meta-Learner (Stacking Ensemble) is fully integrated into `run_full_pipeline.py`. It correctly operates over cross-validated out-of-fold probability predictions generated from XGBoost and LightGBM bases on the 2000-2012 training split, thus eliminating leakage. It is now rigorously evaluated on the 2016-2018 test split.
- **Paper/Code Value Consistency Status**: 100% Consistent. All experimental metrics in `sections/results.tex`, `sections/results_discussion.tex`, and `sections/experimental_setup.tex` have been thoroughly audited and synchronized exactly with `MODEL_RESULTS.csv` and `ABLATION_RESULTS.csv`.

## Remaining Issues
- **Remaining Differences**: None. There are zero remaining discrepancies between the source code output and the paper content.
- **Remaining LaTeX Warnings/Errors**: None. Compilation succeeded perfectly without any fatal errors, missing references, or undefined citations.
