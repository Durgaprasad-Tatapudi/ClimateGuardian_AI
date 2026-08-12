import pandas as pd
import shutil
from pathlib import Path

# Paths
results_dir = Path("06_Results")

def finalize_tables():
    if not results_dir.exists():
        print("Results dir not found.")
        return
        
    print("Finalizing CSV Tables...")
    
    # 1. Final Model Comparison
    if (results_dir / "MODEL_RESULTS.csv").exists():
        df = pd.read_csv(results_dir / "MODEL_RESULTS.csv")
        df.to_csv("FINAL_MODEL_COMPARISON.csv", index=False)
        print("Created FINAL_MODEL_COMPARISON.csv")
        
        # 2. Target Comparison
        # Group by target
        target_df = df.groupby('Target')[['F1-Score', 'ROC-AUC', 'Recall']].mean().reset_index()
        target_df.to_csv("FINAL_TARGET_COMPARISON.csv", index=False)
        print("Created FINAL_TARGET_COMPARISON.csv")
        
    # 3. Lead Time Comparison
    if (results_dir / "LEAD_TIME_RESULTS.csv").exists():
        shutil.copy(results_dir / "LEAD_TIME_RESULTS.csv", "FINAL_LEAD_TIME_COMPARISON.csv")
        print("Created FINAL_LEAD_TIME_COMPARISON.csv")
        
    # 4. Ablation Comparison
    if (results_dir / "ABLATION_RESULTS.csv").exists():
        shutil.copy(results_dir / "ABLATION_RESULTS.csv", "FINAL_ABLATION_COMPARISON.csv")
        print("Created FINAL_ABLATION_COMPARISON.csv")
        
    # 5. Calibration Comparison
    if (results_dir / "CALIBRATION_RESULTS.csv").exists():
        shutil.copy(results_dir / "CALIBRATION_RESULTS.csv", "FINAL_CALIBRATION_COMPARISON.csv")
        print("Created FINAL_CALIBRATION_COMPARISON.csv")

if __name__ == "__main__":
    finalize_tables()
