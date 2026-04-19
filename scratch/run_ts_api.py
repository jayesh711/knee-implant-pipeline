import os
from pathlib import Path
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

def run():
    input_path = "data/NIfTI/AB_72Y_Male_Left_raw.nii.gz"
    output_path = "data/segmentations/phase1/AB_72Y_Male_Left"
    
    # Ensure input exists
    if not Path(input_path).exists():
        print(f"Input {input_path} not found")
        return

    # Ensure output dir exists
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting TotalSegmentator API run for {input_path}")
    try:
        totalsegmentator(input_path, output_path, task="bones", ml=True)
        print("Success")
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    run()
