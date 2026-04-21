import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from skimage import measure
from pathlib import Path
import os
from config import DATA, MAX_TRIANGLES, SMOOTH_ITERS, HU_METAL_MIN, MIN_BONE_VOLUME_CC

def extract_mesh(seg_data, label_id, affine, raw_data=None, metal_threshold=HU_METAL_MIN):
    """Generate a high-fidelity mesh for a specific label ID using full affine alignment."""
    # 1. Create binary mask
    mask = (seg_data == label_id).astype(np.uint8)
    
    # --- UPGRADE: Metal Rod Filtering ---
    if raw_data is not None:
        print(f"    Filtering out high-density hardware (> {metal_threshold} HU)...")
        mask[raw_data > metal_threshold] = 0
    
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

def fallback_to_hu(volume_name, bone_name="tibia", superior_bbox=None):
    """Surgical fallback using Anatomical Z-Sorting and Spatial Constraints."""
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

    # 1. Threshold for bone (Upgraded floor for better isolation)
    # Cortical bone is > 400. Spongy bone is 150-300. Loweringfloor to 200 captures distal ends.
    floor_hu = 200 if bone_name == "tibia" else 250
    print(f"    Scanning volume for bone density (Min HU: {floor_hu}, Metal Max: {HU_METAL_MIN})...")

    # Identify components in a 2x lower-res space directly from disk
    ds_factor = 2
    raw_ds = np.asarray(img.dataobj[::ds_factor, ::ds_factor, ::ds_factor])

    # Bone thresholding while EXCLUDING metal rod (> HU_METAL_MIN)
    mask_ds = ((raw_ds > floor_hu) & (raw_ds < HU_METAL_MIN)).astype(np.uint8)
    
    # --- UPGRADE: Anatomical ROI Cropping (v2: Adaptive Joint Detection) ---
    z_dim = mask_ds.shape[2]
    
    # 1. Distal (Ankle) Removal: Adaptive Bottleneck Detection
    # Calculate bone area per slice to find the ankle joint narrowing
    z_profile = np.sum(mask_ds, axis=(0,1))
    
    # Foot bones usually create a massive area "peak" at the very bottom.
    # We look for the "valley" (narrowing) between the Foot Peak and the Tibia Shaft.
    if len(z_profile) > 100:
        # Scan the bottom 25% for the foot peak
        search_range = int(z_dim * 0.25)
        foot_region = z_profile[:search_range]
        if np.any(foot_region > 0):
            foot_peak_z = np.argmax(foot_region)
            # Find the first significant narrowing (valley) above the foot peak
            # We look for a slice where area is < 60% of the foot peak or just small.
            for z in range(foot_peak_z, search_range):
                if z_profile[z] < foot_region[foot_peak_z] * 0.4:
                    print(f"    [OK] Detected Ankle Bottleneck at Z={z * ds_factor}. Detaching feet...")
                    mask_ds[:, :, :z] = 0
                    break

    # 2. Adaptive Superior Constraint (v3: Soft Separation)
    # We no longer wipe out slices above the tibia. Instead, we use the Femur
    # as a "Repelled Space" to guide component selection.
    femur_distal_z = None
    if bone_name == "tibia" and superior_bbox is not None:
        femur_distal_z = superior_bbox[2] // ds_factor
        print(f"    Anatomical Guidance: Femur distal end detected at Z={femur_distal_z * ds_factor}. Seeking tibia below...")

    labels_ds = measure.label(mask_ds)
    props = measure.regionprops(labels_ds, intensity_image=raw_ds)
    
    # 2. SELECTION: Distinguish Bone from Hardware using Intensity and Geometry
    # We want a component that lives in the 400-1500 HU range and isn't too "skinny" (rod-like)
    scored_candidates = []
    
    for p in props:
        if p.area < 5000: continue # Ignore small noise
        
        # Calculate Physical Markers
        mean_hu = p.mean_intensity
        z_len = p.bbox[5] - p.bbox[2]
        width = max(p.bbox[3]-p.bbox[0], p.bbox[4]-p.bbox[1])
        aspect_ratio = z_len / width if width > 0 else 0
        
        # Scoring Logic:
        # 1. Intensity Penalty: Structural metal (rods) usually have Mean HU > 1800
        intensity_score = 1.0
        if mean_hu > 1800:
            intensity_score = 0.1 # Heavily penalize metal hardware
        elif mean_hu < 500:
            intensity_score = 0.5 # Penalize low-density noise
            
        # 2. Geometry Penalty: Rods have high aspect ratios (thin and long)
        # Tibia is thick. Rod is thin.
        geometry_score = 1.0
        if aspect_ratio > 7.0: # Rod-like
             geometry_score = 0.2
             
        # 3. Spatial Score (Anatomical Position)
        # Tibia centroid MUST be significantly below the Femur distal end.
        spatial_score = 1.0
        if femur_distal_z is not None:
            centroid_z = p.centroid[2]
            if centroid_z > femur_distal_z - 5: # Too close or above
                spatial_score = 0.1
                
        # Combined score prioritized by volume (area)
        total_score = p.area * intensity_score * geometry_score * spatial_score
        
        scored_candidates.append({
            'prop': p,
            'score': total_score,
            'mean_hu': mean_hu,
            'aspect': aspect_ratio
        })
    
    if not scored_candidates: 
        print(f"    Error: No valid bone-like structures found for {bone_name}.")
        return None
        
    # Pick the candidate with the highest anatomical score
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    best = scored_candidates[0]
    target_prop = best['prop']
        
    print(f"    Selected {bone_name} candidate (Area: {target_prop.area}, Mean HU: {best['mean_hu']:.1f}, Aspect: {best['aspect']:.2f})")
        
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
    
    # --- UPGRADE: Segment-Level Rod Subtraction ---
    # To remove the internal rod from the tibia, we use a much lower intensity ceiling (1800 HU)
    # compared to the general metal filter (2500+ HU).
    mesh_ceiling = 1800 if bone_name == "tibia" else HU_METAL_MIN
    # --- UPGRADE: Anatomical Joint Separation (V3) ---
    # To prevent the "Plane" cut and restore curved condyles, we avoid hard Z-crops.
    # Instead, we identify the specific bone body and use morphological clean-up.
    # Bone threshold with strict metal subtraction
    sub_mask = ((sub_vol > floor_hu) & (sub_vol < mesh_ceiling)).astype(np.uint8)
    
    # If the sub-volume likely contains both bones (merging at the joint),
    # use morphological opening to break the bridge while keeping curved ends.
    from skimage.morphology import binary_opening, ball
    # A small opening (radius 2) kills the thin "bridges" in the joint space
    # but leaves the large rounded bone heads mostly intact.
    bone_mask = binary_opening(sub_mask, ball(2)).astype(np.uint8)
    
    # --- UPGRADE: Final Connectivity & Morphological Clean ---
    # Morphological opening (radius 1) to break thin bridges to hardware/bone dust
    from skimage.morphology import binary_opening, ball
    bone_mask = binary_opening(bone_mask, ball(1)).astype(np.uint8)

    labels_final = measure.label(bone_mask)
    if labels_final.max() > 1:
        print(f"    Cleaning up residual hardware/dust: Found {labels_final.max()} parts. Keeping largest bone body...")
        counts = np.bincount(labels_final.flat)
        largest_label = np.argmax(counts[1:]) + 1
        bone_mask = (labels_final == largest_label).astype(np.uint8)
    
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
        # Without AI segmentation, we can't use superior_bbox constraints reliably
        femur_mesh = fallback_to_hu(volume_name, "femur")
        tibia_mesh = fallback_to_hu(volume_name, "tibia")
        if femur_mesh: femur_mesh.export(DATA / "meshes" / f"{volume_name}_femur.stl")
        if tibia_mesh: tibia_mesh.export(DATA / "meshes" / f"{volume_name}_tibia.stl")
        return
        
    print(f"--- UPGRADED Mesh generation for {volume_name} (Mode: {'MRI' if is_mr else 'CT'}) ---")
    img = nib.load(str(seg_file))
    data = img.get_fdata()
    
    unique_labels = np.unique(data).astype(int)
    
    # Candidates for various TotalSegmentator models/versions (TS v1 and v2)
    # Candidates for various TotalSegmentator models/versions (TS v1 and v2)
    # v2: 75=femur_left, 76=femur_right, 77=tibia_left, 78=tibia_right
    femur_candidates = [75, 76, 13, 14, 24, 25] 
    tibia_candidates = [77, 78, 2, 46, 45, 26, 27] 
    
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

    # Cache the femur bounding box from AI segmentation to guide Tibia fallback if needed
    femur_bbox = None
    if actual_femur:
        # Calculate bounding box of femur in AI label map
        f_mask = (data == actual_femur)
        f_props = measure.regionprops(f_mask.astype(np.uint8))
        if f_props:
            femur_bbox = f_props[0].bbox

    for label_id, name in [(actual_femur, "femur"), (actual_tibia, "tibia")]:
        print(f"Processing {name} (ID: {label_id})...")
        mesh = None
        
        if label_id:
            mask = (data == label_id).astype(np.uint8)
            vol_cc = np.sum(mask) * voxel_vol
            if vol_cc > MIN_BONE_VOLUME_CC: 
                # Apply the same 1800 HU ceiling for tibia mesh subtraction in AI path
                local_metal_max = 1800 if name == "tibia" else HU_METAL_MIN
                mesh = extract_mesh(data, label_id, img.affine, raw_data=raw_data, metal_threshold=local_metal_max)
            else:
                print(f"  Warning: AI {name} mesh is too small ({vol_cc:.1f}cc < {MIN_BONE_VOLUME_CC}cc).")
                if not is_mr:
                    print("  Triggering HU Fallback...")
        
        if mesh is None and not is_mr:
            # Pass femur_bbox as a superior constraint for tibia fallback
            mesh = fallback_to_hu(volume_name, name, superior_bbox=femur_bbox if name == "tibia" else None)
            
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
