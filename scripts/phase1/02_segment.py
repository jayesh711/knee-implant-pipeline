import subprocess
import sys
import os
from pathlib import Path
from config import DATA

def run_segmentation(volume_name="S0001"):
    input_file = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    output_dir = DATA / "segmentations" / "phase1" / volume_name
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Starting TotalSegmentator for {volume_name} ---")
    print(f"Input:  {input_file}")
    # Use 'total' task on PURE SIGNAL (raw resampled) to fix 'holey' bones
    print(f"Running Pure Signal TotalSegmentator (Task: total) on {input_file}...")
    
    # Run totalsegmentator via subprocess
    try:
        subprocess.run([
            r"D:\conda_envs\knee-pipeline\Scripts\totalsegmentator.exe",
            "-i", str(input_file),
            "-o", str(output_dir),
            "--ml",           # Multi-label output in a single file
            "--task", "total", # Overall segmentation (includes bones)
            "--quiet"          # Reduce terminal noise
        ], check=True)
        print(f"Successfully completed segmentation for {volume_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during TotalSegmentator run: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TotalSegmentator on NIfTI volume")
    parser.add_argument("--name", type=str, default="S0001", help="Volume name to segment")
    
    args = parser.parse_args()
    run_segmentation(args.name)
