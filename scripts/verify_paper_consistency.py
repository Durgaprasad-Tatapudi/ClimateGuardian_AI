import os
import re

def verify():
    print("="*60)
    print("PAPER CONSISTENCY VERIFICATION")
    print("="*60)
    
    tex_dir = "paper/sections"
    if not os.path.exists(tex_dir):
        print("FAIL: paper/sections directory missing.")
        return False
        
    def check_value_in_tex(val, name):
        found = False
        for f in os.listdir(tex_dir):
            if f.endswith(".tex"):
                with open(os.path.join(tex_dir, f), 'r', encoding='utf-8') as file:
                    if val in file.read():
                        found = True
                        break
        if found:
            print(f"PASS: {name} ({val}) found in paper source.")
            return True
        else:
            print(f"FAIL: {name} ({val}) missing from paper source.")
            return False

    checks = [
        # Table III - Compound GRU (Offline)
        ("0.8457", "Compound GRU Accuracy"),
        ("0.2010", "Compound GRU Precision"),
        ("0.9111", "Compound GRU Recall"),
        ("0.3293", "Compound GRU F1"),
        # Table IV - Ablation values
        ("0.2469", "Compound Climate Only F1"),
        ("0.1905", "Compound Hydro Only F1"),
        ("0.4000", "Compound Full Features F1"),
        # Lead time
        ("7-day", "Lead Time Mention"),
        # Other text
        ("0.9790", "Overall Heatwave Text Claim")
    ]
    
    failed = False
    for val, desc in checks:
        if not check_value_in_tex(val, desc):
            failed = True
            
    # Figure references check
    figures = [
        "Fig_01_Architecture.pdf",
        "Fig_02_Flood_Confusion_Matrix.png",
        "Fig_03_Heatwave_ROC.png",
        "Fig_04_Heatwave_LeadTime.png",
        "Fig_05_Compound_Ablation.png",
        "Fig_06_Compound_Calibration.png",
        "Fig_07_Flood_SHAP.png"
    ]
    
    for fig in figures:
        if not check_value_in_tex(fig, f"Figure Ref: {fig}"):
            failed = True

    if failed:
        print("\nPAPER CONSISTENCY: FAIL")
        return False
    else:
        print("\nPAPER CONSISTENCY: PASS")
        return True

if __name__ == "__main__":
    import sys
    if not verify():
        sys.exit(1)
