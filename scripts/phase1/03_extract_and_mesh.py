import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from skimage import measure
from pathlib import Path
import os
from config import DATA, MAX_TRIANGLES, SMOOTH_ITERS

def extract_mesh(seg_data, label_id, affine):
    """Generate a high-fidelity mesh for a specific label ID using full affine alignment."""
    # 1. Create binary mask
    mask = (seg_data == label_id).astype(np.uint8)
    if not np.any(mask):
        print(f"Warning: Label {label_id} not found in segmentation.")
        return None
        
    # 2. Marching Cubes (Base Surface in Voxel Space)
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    
    # 3. Apply Multi-Stage Affine Transform (Voxel to World)
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
    
    # 4. Clinical Refinement
    # --- Stage 1: Trimesh Robust Repair ---
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    mesh.process() # Standardized cleanup
    mesh.remove_infinite_values()
    
    # --- Stage 2: PyMeshLab Clinical Refinement ---
    print(f"  Starting PyMeshLab clinical refinement...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    
    try:
        ms.apply_filter("meshing_close_holes", maxholesize=100)
    except Exception as e:
        print(f"    Warning: Could not close holes ({e}). Continuing...")
    
    print(f"    Applying Taubin Smoothing (Steps: {SMOOTH_ITERS}, lambda: 0.5, mu: -0.53)...")
    ms.apply_coord_taubin_smoothing(stepsmoothnum=SMOOTH_ITERS, lambda_=0.5, mu=-0.53)
    
    current_mesh = ms.current_mesh()
    if current_mesh.face_number() > MAX_TRIANGLES:
        print(f"    Decimating from {current_mesh.face_number()} to {MAX_TRIANGLES} faces...")
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=MAX_TRIANGLES)
        
    final_m = ms.current_mesh()
    return trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())

def fallback_to_hu(volume_name, bone_name="tibia"):
    """Surgical fallback using Anatomical Z-Sorting of high-HU components."""
    prepped_volume = DATA / "NIfTI" / f"{volume_name}_prepped.nii.gz"
    if not prepped_volume.exists():
        prepped_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
        if not prepped_volume.exists():
            print(f"    Error: No suitable volume found for fallback for {volume_name}")
            return None
        
    print(f"  [SURGICAL FALLBACK] AI missed the {bone_name}. Applying Anatomical Z-Sorting...")
    img = nib.load(str(prepped_volume))
    affine = img.affine
    
    # 1. Threshold for bone
    print(f"    Scanning volume for bone density (Memory-Safe Mode)...")
    
    # Identify components in a 2x lower-res space directly from disk
    ds_factor = 2
    mask_ds = (img.dataobj[::ds_factor, ::ds_factor, ::ds_factor] > 250).astype(np.uint8)
    
    labels_ds = measure.label(mask_ds)
    props = measure.regionprops(labels_ds)
    
    # 2. HEAVYWEIGHT SELECTION: Identify main bone structures by volume
    bone_props = [p for p in props if p.area > 5000] # Min volume filter
    if len(bone_props) < 2: 
        print(f"    Error: Found only {len(bone_props)} bone structures. Need at least 2.")
        return None
        
    # Pick top 3 largest components to capture Pelvis/Femur/Tibia while ignoring noise
    bone_props.sort(key=lambda x: x.area, reverse=True)
    candidates = bone_props[:3]
    
    # Sort these main bones by Z-centroid (Vertical position)
    candidates.sort(key=lambda x: x.centroid[2], reverse=True)
    
    if bone_name == "femur":
        # If 2 bones found: top is Femur. If 3: [Pelvis, Femur, Tibia]
        target_prop = candidates[0] if len(candidates) == 2 else candidates[1]
    else: # tibia
        target_prop = candidates[-1] # Bottom-most among the main structures
        
    print(f"    Selected component at Z-centroid {target_prop.centroid[2] * ds_factor:.1f} as {bone_name}")
    
    # 4. Extract High-Res Subvolume
    minr, minc, minz, maxr, maxc, maxz = target_prop.bbox
    minr, minc, minz = minr * ds_factor, minc * ds_factor, minz * ds_factor
    maxr, maxc, maxz = maxr * ds_factor, maxc * ds_factor, maxz * ds_factor
    
    # Add a small buffer (10 voxels)
    buffer = 10
    minr, minc, minz = max(0, minr-buffer), max(0, minc-buffer), max(0, minz-buffer)
    maxr, maxc, maxz = min(img.shape[0], maxr+buffer), min(img.shape[1], maxc+buffer), min(img.shape[2], maxz+buffer)
    
    sub_vol = np.asarray(img.dataobj[minr:maxr, minc:maxc, minz:maxz])
    bone_mask = (sub_vol > 250).astype(np.uint8)
    
    # Generate mesh with local offset
    verts, faces, _, _ = measure.marching_cubes(bone_mask, level=0.5)
    # Apply offset + affine
    verts[:, 0] += minr
    verts[:, 1] += minc
    verts[:, 2] += minz
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
    
    # --- Stage 1: Trimesh Robust Repair ---
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    mesh.process() # All-in-one cleanup
    mesh.remove_infinite_values()
    
    # --- Stage 2: PyMeshLab Clinical Refinement ---
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    
    try:
        ms.apply_filter("meshing_close_holes", maxholesize=100)
    except Exception as e:
        print(f"    Warning: Could not close holes for {bone_name} ({e}).")
        
    ms.apply_coord_taubin_smoothing(stepsmoothnum=SMOOTH_ITERS, lambda_=0.5, mu=-0.53)
    
    current_mesh = ms.current_mesh()
    if current_mesh.face_number() > MAX_TRIANGLES:
        print(f"    Decimating from {current_mesh.face_number()} to {MAX_TRIANGLES} faces...")
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=MAX_TRIANGLES)
        
    final_m = ms.current_mesh()
    return trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())

def process_volume(volume_name="S0001", is_mr=False):
    potential_paths = [
        DATA / "segmentations" / "phase1" / f"{volume_name}.nii",
        DATA / "segmentations" / "phase1" / f"{volume_name}.nii.gz",
        DATA / "segmentations" / "phase1" / volume_name / "multilabel.nii.gz"
    ]
    
    seg_file = None
    for p in potential_paths:
        if p.exists():
            seg_file = p
            break
            
    if not seg_file:
        if is_mr:
            print(f"Error: No segmentation for {volume_name}. Skipping MRI fallback.")
            return
        print(f"Error: No segmentation for {volume_name}. Fallback triggered.")
        femur = fallback_to_hu(volume_name, "femur")
        tibia = fallback_to_hu(volume_name, "tibia")
        if femur: femur.export(DATA / "meshes" / f"{volume_name}_femur.stl")
        if tibia: tibia.export(DATA / "meshes" / f"{volume_name}_tibia.stl")
        return
        
    print(f"--- UPGRADED Mesh generation for {volume_name} (Mode: {'MRI' if is_mr else 'CT'}) ---")
    img = nib.load(str(seg_file))
    data = img.get_fdata()
    
    unique_labels = np.unique(data).astype(int)
    
    # Candidates for various TotalSegmentator models/versions
    femur_candidates = [13, 76, 77, 75, 44, 43, 2, 1, 25, 24] 
    tibia_candidates = [2, 46, 45, 4, 3, 27, 26] 
    
    actual_femur = next((c for c in femur_candidates if c in unique_labels), None)
    actual_tibia = next((c for c in tibia_candidates if c in unique_labels), None)
    
    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    voxel_vol = np.prod(img.header.get_zooms()) / 1000.0 # in cc
    
    min_vol = 200 if is_mr else 400
    
    for label_id, name in [(actual_femur, "femur"), (actual_tibia, "tibia")]:
        print(f"Processing {name} (ID: {label_id})...")
        mesh = None
        
        if label_id:
            mask = (data == label_id).astype(np.uint8)
            vol_cc = np.sum(mask) * voxel_vol
            if vol_cc > min_vol: 
                mesh = extract_mesh(data, label_id, img.affine)
            else:
                print(f"  Warning: AI {name} mesh is too small ({vol_cc:.1f}cc).")
                if not is_mr:
                    print("  Triggering HU Fallback...")
        
        if mesh is None and not is_mr:
            mesh = fallback_to_hu(volume_name, name)
            
        if mesh:
            mesh.export(str(output_dir / f"{volume_name}_{name}.stl"))
            print(f"  Successfully exported clinical {name}: {volume_name}_{name}.stl")
        elif is_mr:
             print(f"  Warning: Skipping {name} for MR as AI result was insufficient and no fallback available.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001")
    parser.add_argument("--mr", action="store_true", help="Process as MRI (disables HU fallback, lowers volume threshold)")
    args = parser.parse_args()
    process_volume(args.name, is_mr=args.mr)
