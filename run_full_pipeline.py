import subprocess
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path: str, args: list = None):
    args = args or []
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable # Fallback
        
    cmd = [python_exe, script_path] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # We use subprocess.run to stream output directly
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"SUCCESS: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {script_path}")
        logger.error(f"Error Output: {e.stderr}")
        return False

def main():
    import datetime
    run_id = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["EXPERIMENT_RUN_ID"] = run_id
    logger.info(f"Starting Full Pipeline Execution... Run ID: {run_id}")
    
    scripts = [
        ("src/preprocessing/data_cleaning.py", []),
        ("src/features/build_features.py", []),
        ("src/models/train_models.py", []),
        ("src/evaluation/generate_eda_figures.py", []),
        ("src/evaluation/advanced_evaluation.py", []),
        ("run_live_inference.py", ["--lat", "28.6139", "--lon", "77.2090"])
    ]
    
    for script, args in scripts:
        if not run_script(script, args):
            logger.critical("Pipeline halted due to failure.")
            sys.exit(1)
            
    logger.info("Full Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
