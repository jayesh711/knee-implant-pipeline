import subprocess
import os
import sys
from pathlib import Path

# Paths to scan for DICOM data
DICOM_ROOTS = [
    Path("data/DICOM"),
    Path(r"D:\Knee CT Images")
]

def find_dicom_series(root_path):
    """Recursively find all unique folders that contain at least 20 DICOM files."""
    dicom_series = []
    print(f"Scanning for DICOM series in: {root_path}...")
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        dcm_count = len([f for f in filenames if f.lower().endswith(".dcm")])
        if dcm_count > 20: # Threshold to ignore localizers/scouts
            # Use the parent's parent name if it's S0001/P00001 style
            p = Path(dirpath)
            # Try to get a meaningful name from the path
            # Strategy: Go up until we find a name that isn't S0001, P00001, A2025...
            parts = p.parts
            display_name = p.name
            for part in reversed(parts):
                if not any(x in part.upper() for x in ["S0001", "P00001", "A2025", "DICOM", "SCAN"]):
                    display_name = part
                    break
            
            # Clean display name (remove spaces)
            safe_name = display_name.replace(" ", "_")
            dicom_series.append({"name": safe_name, "path": str(p)})
            
    return dicom_series

def run_command(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)

def process_patient(patient):
    name = patient["name"]
    path = patient["path"]
    
    print(f"\n" + "="*60)
    print(f" CLINICAL PIPELINE: {name}")
    print(f" PATH: {path}")
    print("="*60)
    
    # 1. Ingestion (High-Order Resampling + Anisotropic Denoising)
    run_command(["python", "-m", "scripts.ingest_dicom", path, "--name", name])
    
    # 2. Segmentation (Bone Labeling)
    run_command(["python", "-m", "scripts.phase1.02_segment", "--name", name])
    
    # 3. Meshing (Taubin Smoothing + Manifold Repair)
    run_command(["python", "-m", "scripts.phase1.03_extract_and_mesh", "--name", name])
    
    # 4. Verify
    try:
        run_command(["python", "-m", "scripts.phase1.verify_labels", "--name", name])
    except:
        pass

if __name__ == "__main__":
    all_series = []
    for root in DICOM_ROOTS:
        if root.exists():
            all_series.extend(find_dicom_series(root))
    
    print(f"Found {len(all_series)} candidates for processing.")
    
    # Deduplicate by name
    seen = set()
    unique_series = []
    for s in all_series:
        if s["name"] not in seen:
            unique_series.append(s)
            seen.add(s["name"])
            
    for series in unique_series:
        try:
            process_patient(series)
        except Exception as e:
            print(f"\n[!] Error processing {series['name']}: {e}")
            continue

    print("\n" + "#"*40)
    print(" BATCH CLINICAL PROCESSING FINISHED")
    print("#"*40)
