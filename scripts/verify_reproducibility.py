import os
import json
import hashlib
from pathlib import Path

def get_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def verify():
    print("="*60)
    print("REPRODUCIBILITY VERIFICATION")
    print("="*60)
    
    manifest_path = "results/canonical/CANONICAL_MANIFEST.json"
    if not os.path.exists(manifest_path):
        print(f"FAIL: Manifest not found at {manifest_path}")
        return False
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    checksums = manifest.get("checksums", {})
    failed = False
    
    for filename, expected_hash in checksums.items():
        if filename.endswith(".csv"):
            filepath = os.path.join("results/canonical", filename)
        else:
            filepath = os.path.join("Paper_Figures", filename)
            
        if not os.path.exists(filepath):
            print(f"FAIL: Missing canonical file -> {filepath}")
            failed = True
            continue
            
        actual_hash = get_hash(filepath)
        if actual_hash != expected_hash:
            print(f"FAIL: Checksum mismatch for {filepath}")
            print(f"  Expected: {expected_hash}")
            print(f"  Actual:   {actual_hash}")
            failed = True
        else:
            print(f"PASS: {filename}")
            
    if failed:
        print("\nREPRODUCIBILITY AUDIT: FAIL")
        return False
    else:
        print("\nREPRODUCIBILITY AUDIT: PASS")
        return True

if __name__ == "__main__":
    import sys
    if not verify():
        sys.exit(1)
