import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from skimage import measure
from pathlib import Path
import os
from config import DATA, MAX_TRIANGLES, SMOOTH_ITERS, HU_METAL_MIN, MIN_BONE_VOLUME_CC

def extract_mesh(seg_data, label_id, affine, raw_data=None):
    """Generate a high-fidelity mesh for a specific label ID using full affine alignment."""
    # 1. Create binary mask
    mask = (seg_data == label_id).astype(np.uint8)
    
    # --- UPGRADE: Metal Rod Filtering ---
    if raw_data is not None:
        print(f"    Filtering out high-density metal (> {HU_METAL_MIN} HU)...")
        mask[raw_data > HU_METAL_MIN] = 0
    
    if not np.any(mask):
        print(f"Warning: Label {label_id} not found or fully masked out by metal filter.")
        return None
        
    # --- UPGRADE: Connectivity Cleanup ---
    # Keep only the largest component to remove floating noise
    labels = measure.label(mask)
    if labels.max() > 1:
        print(f"    Connectivity cleanup: Found {labels.max()} components. Keeping largest...")
        counts = np.bincount(labels.flat)
        largest_label = np.argmax(counts[1:]) + 1
        mask = (labels == largest_label).astype(np.uint8)

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
    # Must use the raw volume (HU values). The prepped volume is z-score normalized,
    # so the 250 HU bone threshold below would produce an empty mask on prepped data.
    raw_volume = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    if not raw_volume.exists():
        raw_volume = DATA / "NIfTI" / f"{volume_name}_prepped.nii.gz"
        if not raw_volume.exists():
            print(f"    Error: No suitable volume found for fallback for {volume_name}")
            return None

    print(f"  [SURGICAL FALLBACK] AI missed the {bone_name}. Applying Anatomical Z-Sorting...")
    img = nib.load(str(raw_volume))
    affine = img.affine

    # 1. Threshold for bone
    print(f"    Scanning volume for bone density (Memory-Safe Mode)...")

    # Identify components in a 2x lower-res space directly from disk
    ds_factor = 2
    raw_ds = np.asarray(img.dataobj[::ds_factor, ::ds_factor, ::ds_factor])

    # Bone thresholding while EXCLUDING metal rod (> HU_METAL_MIN)
    mask_ds = ((raw_ds > 250) & (raw_ds < HU_METAL_MIN)).astype(np.uint8)
    
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
    # Bone threshold with metal protection
    bone_mask = ((sub_vol > 250) & (sub_vol < HU_METAL_MIN)).astype(np.uint8)
    
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
    
    # Candidates for various TotalSegmentator models/versions (TS v1 and v2)
    femur_candidates = [13, 14, 76, 77, 75, 44, 43, 2, 1, 25, 24] 
    tibia_candidates = [2, 46, 45, 4, 3, 27, 26, 80, 81] 
    
    # ROBUST SELECTION: Pick the largest component among valid candidates
    def get_best_label(candidates, data, unique_labels):
        valid = [c for c in candidates if c in unique_labels]
        if not valid: return None
        # Return label with max voxel count
        return max(valid, key=lambda c: np.sum(data == c))

    actual_femur = get_best_label(femur_candidates, data, unique_labels)
    actual_tibia = get_best_label(tibia_candidates, data, unique_labels)
    
    # --- UPGRADE: Anatomical Spatial Validation ---
    if actual_femur is not None and actual_tibia is not None:
        # Calculate centroids along Z-axis (superior-inferior)
        # Using voxel coordinates is sufficient for relative comparison
        f_z = np.mean(np.argwhere(data == actual_femur)[:, 2])
        t_z = np.mean(np.argwhere(data == actual_tibia)[:, 2])
        
        # Check orientation from affine
        z_direction = np.sign(img.affine[2,2]) # +1 if Z increases superiorly, -1 if inferiorly
        
        # In standard LPS, +Z is Superior. Tibia MUST be below (Inferior to) Femur.
        # So Z_Tibia should be < Z_Femur if z_direction is +
        is_anatomical = (t_z < f_z) if z_direction > 0 else (t_z > f_z)
        
        if not is_anatomical:
            print(f"  [ANATOMICAL ERROR] Tibia candidate (ID: {actual_tibia}, Z: {t_z:.1f}) is positioned ABOVE Femur (ID: {actual_femur}, Z: {f_z:.1f}) — label is wrong.")
            print(f"  Invalidating AI tibia label. HU fallback will be triggered.")
            actual_tibia = None

    
    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    voxel_vol = np.prod(img.header.get_zooms()) / 1000.0 # in cc

    # Load raw volume for metal filtering if available
    raw_vol_path = DATA / "NIfTI" / f"{volume_name}_prepped.nii.gz"
    if not raw_vol_path.exists():
        raw_vol_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    
    raw_data = nib.load(str(raw_vol_path)).get_fdata() if raw_vol_path.exists() else None

    for label_id, name in [(actual_femur, "femur"), (actual_tibia, "tibia")]:
        print(f"Processing {name} (ID: {label_id})...")
        mesh = None
        
        if label_id:
            mask = (data == label_id).astype(np.uint8)
            vol_cc = np.sum(mask) * voxel_vol
            if vol_cc > MIN_BONE_VOLUME_CC: 
                mesh = extract_mesh(data, label_id, img.affine, raw_data=raw_data)
            else:
                print(f"  Warning: AI {name} mesh is too small ({vol_cc:.1f}cc < {MIN_BONE_VOLUME_CC}cc).")
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
