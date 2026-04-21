import subprocess
import sys
from pathlib import Path
import argparse
from config import DATA, SLICER_PATH

def _find_slicer():
    """Find the 3D Slicer executable."""
    slicer = Path(SLICER_PATH)
    if slicer.exists():
        return str(slicer)
    
    # Try common Windows locations
    common_paths = [
        Path(r"D:\3D Slicer 5.10.0\Slicer.exe"),
        Path(r"C:\Program Files\Slicer 5.10.0\Slicer.exe"),
        Path(r"C:\Program Files\Slicer\Slicer.exe"),
    ]
    for p in common_paths:
        if p.exists():
            return str(p)

    return None

def visualize_with_slicer(name="S0001"):
    """Launch 3D Slicer with all patient meshes and skeletons."""
    slicer_exe = _find_slicer()
    if not slicer_exe:
        print("3D Slicer not found. Falling back to PyVista viewer...")
        visualize_with_pyvista(name)
        return

    print(f"\n--- Launching 3D Slicer for {name} ---")
    
    # Collect all files to load
    files_to_load = []
    
    mesh_dir = DATA / "meshes"
    for mesh_name in [f"{name}_femur.stl", f"{name}_tibia.stl", f"{name}_ground_truth.stl"]:
        mesh_path = mesh_dir / mesh_name
        if mesh_path.exists():
            files_to_load.append(str(mesh_path))
            print(f"  [OK] Loading mesh: {mesh_name}")
    
    canal_dir = DATA / "canal" / name
    for bone in ["femur", "tibia"]:
        skel_path = canal_dir / f"{bone}_skeleton.nii.gz"
        if skel_path.exists():
            files_to_load.append(str(skel_path))
            print(f"  [OK] Loading canal skeleton: {bone}")
    
    if not files_to_load:
        print(f"Error: No output files found for patient '{name}'. Run the pipeline first.")
        return
    
    # Launch 3D Slicer with all files as arguments
    cmd = [slicer_exe] + files_to_load
    print(f"\n  Starting 3D Slicer with {len(files_to_load)} file(s)...")
    
    try:
        subprocess.Popen(cmd)
        print(f"  [OK] 3D Slicer launched successfully.")
        print(f"  (Slicer is running in a separate window)")
    except Exception as e:
        print(f"  Error launching 3D Slicer: {e}")
        print(f"  Falling back to PyVista viewer...")
        visualize_with_pyvista(name)

def visualize_with_pyvista(name="S0001"):
    """Fallback PyVista-based viewer (original implementation)."""
    import pyvista as pv
    import nibabel as nib
    import numpy as np

    print(f"\n--- Launching PyVista Clinical Viewer for {name} ---")
    plotter = pv.Plotter(title=f"Clinical Diagnostic View: {name}")
    plotter.set_background("black")
    
    mesh_dir = DATA / "meshes"
    femur_path = mesh_dir / f"{name}_femur.stl"
    tibia_path = mesh_dir / f"{name}_tibia.stl"
    
    # 1. Load Bone Meshes (Clinical Standards)
    if femur_path.exists():
        print(f"  Loading Femur Mesh...")
        femur = pv.read(str(femur_path))
        plotter.add_mesh(femur, color="ivory", label="Femur (Clinical Standard)", smooth_shading=True, opacity=0.8)
    
    if tibia_path.exists():
        print(f"  Loading Tibia Mesh...")
        tibia = pv.read(str(tibia_path))
        plotter.add_mesh(tibia, color="lightblue", label="Tibia (Clinical Standard)", smooth_shading=True, opacity=0.8)
        
    # 2. Load Medullary Canal Centerlines (Skeletons)
    _add_canal_skeleton_pyvista(plotter, name, "femur", color="red")
    _add_canal_skeleton_pyvista(plotter, name, "tibia", color="yellow")
    
    # 3. Load Ground Truth (HU Ghost - Wireframe)
    gt_path = mesh_dir / f"{name}_ground_truth.stl"
    if gt_path.exists():
        print(f"  Loading Ground Truth Signal (Ghost Layer)...")
        gt = pv.read(str(gt_path))
        plotter.add_mesh(gt, color="gray", opacity=0.2, label="Raw HU Signal (Unsmoothed)", style="wireframe")
    
    plotter.add_legend()
    plotter.add_axes()
    print(f"Viewer Ready. Displaying 3D surgical plan for {name}...")
    plotter.show()

def _add_canal_skeleton_pyvista(plotter, patient_name, bone_name="femur", color="red"):
    """Load and add the medullary canal skeleton to the PyVista plotter."""
    import pyvista as pv
    import nibabel as nib
    import numpy as np

    skel_path = DATA / "canal" / patient_name / f"{bone_name}_skeleton.nii.gz"
    if skel_path.exists():
        print(f"  Loading {bone_name.capitalize()} Canal Centerline...")
        skel_img = nib.load(str(skel_path))
        skel_data = skel_img.get_fdata()
        affine = skel_img.affine
        
        # 1. Get voxel coordinates of skeleton
        coords = np.argwhere(skel_data > 0)
        
        if len(coords) > 0:
            # 2. Transform Voxel -> World (Affine)
            world_coords = (affine[:3, :3] @ coords.T).T + affine[:3, 3]
            
            # 3. Sort by Z-axis to ensure the line follows the bone shaft
            z_sorted_idx = np.argsort(world_coords[:, 2])
            sorted_coords = world_coords[z_sorted_idx]
            
            # 4. Create a continuous 3D Tube
            if len(sorted_coords) > 1:
                line = pv.MultipleLines(points=sorted_coords)
                tube = line.tube(radius=2.0) # 4mm diameter tube for clear visibility
                
                plotter.add_mesh(
                    tube, 
                    color=color, 
                    label=f"{bone_name.capitalize()} Canal",
                    opacity=1.0,
                    emissive=True
                )
            else:
                # Fallback to points if only one coordinate
                poly = pv.PolyData(world_coords)
                plotter.add_mesh(poly, color=color, point_size=10, render_points_as_spheres=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001", help="Patient name to visualize")
    parser.add_argument("--pyvista", action="store_true", help="Force PyVista viewer instead of 3D Slicer")
    args = parser.parse_args()
    
    if args.pyvista:
        visualize_with_pyvista(args.name)
    else:
        visualize_with_slicer(args.name)

