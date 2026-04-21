import argparse
import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd_list):
    print(f"\n>> Executing: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full clinical pipeline for a single patient DICOM folder.")
    parser.add_argument("dicom_path", type=str, help="Path to the folder containing DICOM files (.dcm)")
    parser.add_argument("--name", type=str, required=True, help="Patient/Volume name for outputs")
    parser.add_argument("--canal", action="store_true", help="Also run medullary canal analysis (skipped by default)")
    
    args = parser.parse_args()
    
    dicom_path = Path(args.dicom_path)
    name = args.name
    
    if not dicom_path.exists():
        print(f"Error: DICOM path '{dicom_path}' does not exist.")
        sys.exit(1)
        
    print(f"\n" + "="*70)
    print(f" STARTING FULL CLINICAL RECONSTRUCTION: {name}")
    print(f" INPUT: {dicom_path}")
    print("="*70)
    
    python_exe = sys.executable # Use current python (D:\conda_envs\knee-pipeline\python.exe)
    
    # Set PYTHONPATH to project root
    project_root = str(Path(__file__).parent.parent)
    os.environ['PYTHONPATH'] = project_root
    
    try:
        # 1. Ingestion & Clinical Preprocessing
        # (HU Windowing, Denoising, Normalization, Resampling)
        run_command([python_exe, "-m", "scripts.ingest_dicom", str(dicom_path), "--name", name])
        
        # 2. AI Segmentation (Bones)
        run_command([python_exe, "-m", "scripts.phase1.02_segment", "--name", name])
        
        # 3. 3D Mesh Reconstruction (Taubin Smoothing + QEM)
        run_command([python_exe, "-m", "scripts.phase1.03_extract_and_mesh", "--name", name])
        
        # 4. Medullary Canal Analysis (Centerline + Diameter) — opt-in via --canal
        if args.canal:
            run_command([python_exe, "-m", "scripts.canal.canal_measurement", "--name", name])
        else:
            print("\n>> Skipping canal measurement (use --canal to enable)")
        
        # 5. Optional: Ground Truth Reconstruction (HU Signal Baseline)
        run_command([python_exe, "-m", "scripts.reconstruct_ground_truth", "--name", name])
        
        print("\n" + "="*70)
        print(f" RECONSTRUCTION COMPLETE: {name}")
        print(f" RUN THIS TO VISUALIZE:")
        print(f" python -m scripts.visualize_results --name {name}")
        print("="*70)
        
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed during: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
