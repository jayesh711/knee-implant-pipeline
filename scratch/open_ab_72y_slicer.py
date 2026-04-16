import os
import subprocess
import glob

slicer_paths = [
    r"C:\Program Files\Slicer*\Slicer.exe",
    r"C:\Users\jdongre\AppData\Local\NA-MIC\Slicer*\Slicer.exe",
    r"D:\Slicer*\Slicer.exe"
]

found_slicer = None
for p in slicer_paths:
    matches = glob.glob(p)
    if matches:
        found_slicer = matches[-1] # Usually most recent version
        break

meshes = [
    r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes\AB_72_Y_Male-Right_Knee_femur.stl",
    r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes\AB_72_Y_Male-Right_Knee_tibia.stl"
]

existing_meshes = [m for m in meshes if os.path.exists(m)]

if found_slicer and existing_meshes:
    print(f"Found Slicer at {found_slicer}. Launching with meshes: {existing_meshes}")
    subprocess.Popen([found_slicer] + existing_meshes)
elif not found_slicer:
    print("Could not find 3D Slicer executable.")
else:
    print(f"Meshes not found: {meshes}")
