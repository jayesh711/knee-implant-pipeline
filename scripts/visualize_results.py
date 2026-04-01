import pyvista as pv
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse
from config import DATA

def add_canal_skeleton(plotter, patient_name, bone_name="femur", color="red"):
    """Load and add the medullary canal skeleton to the plotter."""
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
            # (Crucial for MultipleLines/Spline to work correctly)
            z_sorted_idx = np.argsort(world_coords[:, 2])
            sorted_coords = world_coords[z_sorted_idx]
            
            # 4. Create a continuous 3D Tube
            # We use MultipleLines for the centerline and then 'tube' it
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

def visualize_patient(name="S0001"):
    print(f"\n--- Launching Clinical Diagnostic Viewer for {name} ---")
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
    add_canal_skeleton(plotter, name, "femur", color="red")
    add_canal_skeleton(plotter, name, "tibia", color="yellow")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001", help="Patient name to visualize")
    args = parser.parse_args()
    
    visualize_patient(args.name)
