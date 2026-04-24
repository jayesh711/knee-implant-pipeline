import trimesh
from pathlib import Path

meshes_dir = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes")
name = "AS_60_male"

tibia_path = meshes_dir / f"{name}_tibia_full.stl"
femur_path = meshes_dir / f"{name}_femur_full.stl"
gt_path = meshes_dir / f"{name}_ground_truth.stl"

for p in [femur_path, tibia_path, gt_path]:
    if p.exists():
        mesh = trimesh.load(str(p))
        bounds = mesh.bounds
        z_min, z_max = bounds[0, 2], bounds[1, 2]
        print(f"{p.name}: Z-range [{z_min:.2f}, {z_max:.2f}], Height: {z_max - z_min:.2f} mm")
    else:
        print(f"{p.name} NOT FOUND")
