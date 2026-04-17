import os
import subprocess

# Found Slicer path
found_slicer = r"D:\3D Slicer 5.10.0\Slicer.exe"

if not os.path.exists(found_slicer):
    print(f"Error: 3D Slicer not found at {found_slicer}")
    exit(1)

# Define file paths
base_dir = r"d:\knee-implant-pipeline\knee-implant-pipeline"
files_to_open = [
    # Background Image Volume
    os.path.join(base_dir, r"data\NIfTI\AB_72_Y_Male-Right_Knee.nii.gz"),
    # Segmentation Labels
    os.path.join(base_dir, r"data\segmentations\phase1\AB_72_Y_Male-Right_Knee.nii"),
    # Reconstructed Meshes
    os.path.join(base_dir, r"data\meshes\AB_72_Y_Male-Right_Knee_femur.stl"),
    os.path.join(base_dir, r"data\meshes\AB_72_Y_Male-Right_Knee_tibia.stl"),
    os.path.join(base_dir, r"data\meshes\AB_72_Y_Male-Right_Knee_hardware.stl")
]

# Check existence
existing_files = [f for f in files_to_open if os.path.exists(f)]
missing_files = [f for f in files_to_open if not os.path.exists(f)]

if missing_files:
    print("Warning: Some files were not found:")
    for f in missing_files:
        print(f"  - {f}")

if existing_files:
    print(f"Launching Slicer with {len(existing_files)} files...")
    # Passing files as arguments to Slicer usually works for loading them
    subprocess.Popen([found_slicer] + existing_files)
else:
    print("Error: No files found to open.")
