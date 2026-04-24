import trimesh
from pathlib import Path

meshes_dir = Path(r"d:\knee-implant-pipeline\knee-implant-pipeline\data\meshes")
name_ac = "AC_67_male"
name_as = "AS_60_male"

def get_stats(name):
    tibia_path = meshes_dir / f"{name}_tibia_full.stl"
    femur_path = meshes_dir / f"{name}_femur_full.stl"
    res = {}
    for bone, p in [("femur", femur_path), ("tibia", tibia_path)]:
        if p.exists():
            mesh = trimesh.load(str(p))
            res[bone] = mesh.bounds[:, 2]
    return res

ac_stats = get_stats(name_ac)
as_stats = get_stats(name_as)

print("AC_67_male (Yellow likely):")
for bone, bounds in ac_stats.items():
    print(f"  {bone}: [{bounds[0]:.2f}, {bounds[1]:.2f}], H: {bounds[1]-bounds[0]:.2f}")

print("\nAS_60_male (Brown likely):")
for bone, bounds in as_stats.items():
    print(f"  {bone}: [{bounds[0]:.2f}, {bounds[1]:.2f}], H: {bounds[1]-bounds[0]:.2f}")
