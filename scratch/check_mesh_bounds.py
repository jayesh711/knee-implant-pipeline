import trimesh
from pathlib import Path

mesh_dir = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes")
femur_path = mesh_dir / "AB_72_Y_Male-Right_Knee_femur.stl"
tibia_path = mesh_dir / "AB_72_Y_Male-Right_Knee_tibia.stl"

for p in [femur_path, tibia_path]:
    if p.exists():
        mesh = trimesh.load(str(p))
        print(f"Mesh: {p.name}")
        print(f"  Bounds: {mesh.bounds}")
        print(f"  Center: {mesh.centroid}")
    else:
        print(f"File not found: {p.name}")
