import os
import subprocess
import glob

slicer_paths = [
    r"C:\Program Files\Slicer*\Slicer.exe",
    r"C:\Users\jdongre\AppData\Local\NA-MIC\Slicer*\Slicer.exe",
    r"D:\Slicer*\Slicer.exe",
    r"C:\Program Files\Slicer*\Slicer.exe"
]

found_slicer = None
for p in slicer_paths:
    matches = glob.glob(p)
    if matches:
        found_slicer = matches[0]
        break

meshes = [
    r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes\S0001_Debug_Run_femur.stl",
    r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes\S0001_Debug_Run_tibia.stl"
]

# Check if meshes exist
existing_meshes = [m for m in meshes if os.path.exists(m)]

if found_slicer:
    if existing_meshes:
        print(f"Found Slicer at {found_slicer}. Launching with meshes...")
        subprocess.Popen([found_slicer] + existing_meshes)
    else:
        print(f"Found Slicer at {found_slicer}, but meshes not found at paths: {meshes}")
else:
    print("Could not find 3D Slicer executable automatically.")
    print("Please manually drag and drop these files into 3D Slicer:")
    for m in meshes:
        print(f" - {m}")
