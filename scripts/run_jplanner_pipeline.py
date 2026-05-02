import argparse
import subprocess
import os
import sys
from pathlib import Path

def run_command(cmd_list):
    print(f"\n>> Executing: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full JPlanner-A clinical pipeline for a patient CT.")
    parser.add_argument("dicom_path", type=str, help="Path to the folder containing DICOM files")
    parser.add_argument("--name", type=str, required=True, help="Patient/Volume name for outputs")
    parser.add_argument("--has-metal", action="store_true", help="Enable HU-based hardware filtering")
    
    args = parser.parse_args()
    
    dicom_path = Path(args.dicom_path)
    name = args.name
    
    if not dicom_path.exists():
        print(f"Error: DICOM path '{dicom_path}' does not exist.")
        sys.exit(1)
        
    print(f"\n" + "="*70)
    print(f" JPLANNER-A END-TO-END CLINICAL PIPELINE: {name}")
    print(f" INPUT: {dicom_path}")
    print("="*70)
    
    python_exe = sys.executable
    
    # Set PYTHONPATH to project root
    project_root = str(Path(__file__).parent.parent)
    os.environ['PYTHONPATH'] = project_root
    
    try:
        # 1. Ingestion & Clinical Preprocessing
        print("\n[PHASE 1] Ingestion and Clinical Prep (PURE mode for high-fidelity native resolution)...")
        run_command([python_exe, "-m", "scripts.ingest_dicom", str(dicom_path), "--name", name, "--pure"])
        
        # 2. AI Segmentation (JPlanner-A ONNX)
        print("\n[PHASE 2] AI Segmentation (JPlanner-A Two-Stage)...")
        run_command([python_exe, "-m", "scripts.phase1.02_segment_jplanner", "--name", name])
        
        # 3. 3D Mesh Reconstruction (High-Quality Extraction)
        print("\n[PHASE 3] 3D Mesh Reconstruction...")
        mesh_cmd = [python_exe, "-m", "scripts.phase1.03_extract_and_mesh", "--name", name]
        if args.has_metal:
            mesh_cmd.append("--has-metal")
        run_command(mesh_cmd)
        
        print("\n" + "="*70)
        print(f" PIPELINE COMPLETE: {name}")
        print(f" Output Location: data/meshes/{name}_*")
        print("="*70)
        
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed during: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
