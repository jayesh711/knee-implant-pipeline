import os
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from config import DATA, HU_CANAL_MIN, HU_CANAL_MAX

def calculate_canal_parameters(bone_mask, ct_data, spacing, z_offset=0, is_mr=False):
    """
    Extract medullary canal parameters from a bone mask and CT/MR intensities.
    Refined with Shaft-Only constraints for full-leg clinical accuracy.
    """
    # 1. Bounding box cropping to save memory
    coords = np.argwhere(bone_mask)
    if coords.size == 0:
        return None
    
    x_min, y_min, z_min = coords.min(0)
    x_max, y_max, z_max = coords.max(0)
    
    pad = 5
    x_min = max(0, x_min - pad); y_min = max(0, y_min - pad); z_min = max(0, z_min - pad)
    x_max = min(bone_mask.shape[0], x_max + pad)
    y_max = min(bone_mask.shape[1], y_max + pad)
    z_max = min(bone_mask.shape[2], z_max + pad)
    
    cropped_mask = bone_mask[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_ct = ct_data[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # 2. Isolate potential canal space
    if is_mr:
        # MRI Approach: Use Intensity Percentile within Bone (Marrow is usually hyperintense)
        # We also apply a small erosion to ensure we are truly inside the cortical shell
        bone_core = binary_erosion(cropped_mask, iterations=2)
        if np.any(bone_core):
            intensities = cropped_ct[bone_core > 0]
            # Heuristic: Canal/Marrow is in the top 40% of intensities inside the bone
            thresh = np.percentile(intensities, 60)
            canal_mask = (bone_core > 0) & (cropped_ct >= thresh)
        else:
            canal_mask = cropped_mask > 0
    else:
        # CT Approach: HU-based windowing
        canal_mask = (cropped_mask > 0) & (cropped_ct >= HU_CANAL_MIN) & (cropped_ct <= HU_CANAL_MAX)
    
    # NEW: Shaft-Only Constraint (Ignore hip/pelvis and knee flares)
    z_voxels = np.argwhere(cropped_mask)[:, 2]
    if z_voxels.size == 0: return None
    z_start = np.percentile(z_voxels, 25)
    z_end = np.percentile(z_voxels, 75)
    
    shaft_mask = np.zeros_like(canal_mask)
    shaft_mask[:, :, int(z_start):int(z_end)] = 1
    canal_mask = canal_mask & shaft_mask
    
    # NEW: Vertical Continuity (Largest Connected Component)
    lbl_canal, num_lbls = label(canal_mask)
    if num_lbls > 0:
        bincount = np.bincount(lbl_canal.ravel())
        if len(bincount) > 1:
            main_label = np.argmax(bincount[1:]) + 1
            canal_mask = (lbl_canal == main_label)
    
    if np.sum(canal_mask) == 0:
        print("  Warning: No continuous medullary canal found in the shaft region.")
        return None
        
    # 3. Skeletonize (in shaft region)
    print(f"  Skeletonizing shaft canal (Z-range: {int(z_start)}-{int(z_end)})...")
    skeleton_cropped = skeletonize(canal_mask)
    
    # 4. Measure Diameter using EDT
    edt = distance_transform_edt(canal_mask, sampling=spacing)
    centerline_distances = edt[skeleton_cropped > 0]
    if len(centerline_distances) == 0:
        return None
        
    diameters = centerline_distances * 2.0
    
    # 5. Extract Analytics
    avg_diameter = np.mean(diameters)
    isthmus_diameter = np.min(diameters)
    max_diameter = np.max(diameters)
    
    skel_coords = np.argwhere(skeleton_cropped)
    z_len = (np.max(skel_coords[:, 2]) - np.min(skel_coords[:, 2])) * spacing[2]
    
    # 6. Map back
    skeleton_full = np.zeros_like(bone_mask, dtype=np.uint8)
    skeleton_full[x_min:x_max, y_min:y_max, z_min:z_max] = skeleton_cropped.astype(np.uint8)
    
    return {
        "avg_diameter": avg_diameter,
        "isthmus_diameter": isthmus_diameter,
        "max_diameter": max_diameter,
        "length_mm": z_len,
        "skeleton": skeleton_full,
        "z_min_orig": z_min + z_offset
    }

def process_patient_canal(patient_name, is_mr=False):
    print(f"\n--- Medullary Canal Analysis: {patient_name} (Mode: {'MRI' if is_mr else 'CT'}) ---")
    
    # Load CT/MR (PREPPED)
    ct_path = DATA / "NIfTI" / f"{patient_name}_prepped.nii.gz"
    if not ct_path.exists():
        ct_path = DATA / "NIfTI" / f"{patient_name}_raw.nii.gz"
    
    # Load Segmentation
    potential_seg_paths = [
        DATA / "segmentations" / "phase1" / f"{patient_name}.nii",
        DATA / "segmentations" / "phase1" / f"{patient_name}.nii.gz",
        DATA / "segmentations" / "phase1" / patient_name / "multilabel.nii.gz",
        DATA / "segmentations" / "phase1" / patient_name / f"{patient_name}.nii"
    ]
    
    seg_path = None
    for p in potential_seg_paths:
        if p.exists():
            seg_path = p
            break
            
    if not ct_path.exists() or not seg_path:
        print(f"Error: Missing files for {patient_name} in {ct_path} or (no segmentation found)")
        return
        
    ct_img = nib.load(str(ct_path))
    spacing = ct_img.header.get_zooms()
    seg_img = nib.load(str(seg_path))
    
    # MEMORY-EFFICIENT LABEL CHECK
    print(f"    Scanning segmentation for canal identification (Memory-Efficient)...")
    seg_proxy = seg_img.dataobj
    sample_stride = 5 
    sample_data = np.asarray(seg_proxy[::sample_stride, ::sample_stride, ::sample_stride])
    present_labels = np.unique(sample_data)
    
    femur_labels = [76, 75, 44, 25, 24, 94, 93]
    tibia_labels = [81, 80, 46, 45, 4, 3, 27, 26]

    def pick_best_label(candidates, present, sample):
        valid = [l for l in candidates if l in present]
        if not valid:
            return None
        return max(valid, key=lambda l: np.sum(sample == l))

    # Pre-select both labels and validate anatomical ordering before processing.
    # TotalSegmentator sometimes assigns a hip/pelvis label ID that also appears in
    # the tibia candidate list (e.g. label 80 for left-side scans). If the picked
    # tibia centroid is superior to the femur centroid we must invalidate it so the
    # HU fallback runs instead of computing a canal on the wrong structure.
    ai_femur_id = pick_best_label(femur_labels, present_labels, sample_data)
    ai_tibia_id = pick_best_label(tibia_labels, present_labels, sample_data)

    if ai_femur_id is not None and ai_tibia_id is not None:
        seg_full = np.asarray(seg_proxy)
        z_dir = np.sign(seg_img.affine[2, 2])
        f_z = np.mean(np.argwhere(seg_full == ai_femur_id)[:, 2])
        t_z = np.mean(np.argwhere(seg_full == ai_tibia_id)[:, 2])
        is_anatomical = (t_z < f_z) if z_dir > 0 else (t_z > f_z)
        if not is_anatomical:
            print(f"  [ANATOMICAL ERROR] Tibia label {ai_tibia_id} (Z={t_z:.1f}) is superior to "
                  f"femur label {ai_femur_id} (Z={f_z:.1f}). Invalidating AI tibia — HU fallback will run.")
            ai_tibia_id = None

    bone_id_map = {"femur": ai_femur_id, "tibia": ai_tibia_id}

    for bone_name, target_labels in [("femur", femur_labels), ("tibia", tibia_labels)]:
        print(f"Checking {bone_name}...")
        bone_id = bone_id_map[bone_name]
        if bone_id is None:
            valid_labels = []
        else:
            valid_labels = [bone_id]
        
        mask = None
        bone_slice = None 
        minz = 0
        
        if bone_id:
            print(f"  Found Bone Label ID: {bone_id} for {bone_name}. Loading mask...")
            mask = (np.asarray(seg_proxy) == bone_id).astype(np.uint8)
            voxel_volume = np.prod(spacing) / 1000.0 # in cc
            bone_vol_cc = np.sum(mask) * voxel_volume
            
            if bone_vol_cc < 100: 
                print(f"  Warning: AI segment for {bone_name} is too small ({bone_vol_cc:.1f}cc).")
                mask = None
        
        if mask is None and not is_mr:
            from skimage import measure
            print(f"  [CANAL FALLBACK] Applying Anatomical Z-Sorting for {bone_name}...")
            raw_ct_path = DATA / "NIfTI" / f"{patient_name}_raw.nii.gz"
            fallback_img = nib.load(str(raw_ct_path)) if raw_ct_path.exists() else ct_img
                
            ds = 2
            mask_ds = (np.asarray(fallback_img.dataobj[::ds, ::ds, ::ds]) > 250).astype(np.uint8)
            lbls_ds = measure.label(mask_ds)
            props = measure.regionprops(lbls_ds)
            
            bone_props = [p for p in props if p.area > 5000]
            bone_props.sort(key=lambda x: x.centroid[2], reverse=True) # Top to Bottom
            
            if len(bone_props) < 2:
                print(f"    Error: Found only {len(bone_props)} bone structures. Need at least 2.")
                mask = None
            else:
                if bone_name == "femur":
                    targetp = bone_props[0] if len(bone_props) == 2 else bone_props[1]
                else: # tibia
                    targetp = bone_props[-1]
                
                print(f"    Selected component at Z-centroid {targetp.centroid[2] * ds:.1f} for {bone_name} canal.")
                minx, miny, minz, maxx, maxy, maxz = targetp.bbox
                minx, miny, minz = minx*ds, miny*ds, minz*ds
                maxx, maxy, maxz = maxx*ds, maxy*ds, maxz*ds
                
                sub_vol_data = np.asarray(ct_img.dataobj[minx:maxx, miny:maxy, minz:maxz])
                mask = (sub_vol_data > 250).astype(np.uint8)
                bone_slice = sub_vol_data 
            
        if mask is not None:
            z_offset = minz
            final_bone_slice = bone_slice if bone_slice is not None else np.asarray(ct_img.dataobj)
            stats = calculate_canal_parameters(mask, final_bone_slice, spacing, z_offset=z_offset, is_mr=is_mr)
            
            if stats:
                print(f"  {bone_name.upper()} CANAL DETECTED")
                print(f"  - Length:   {stats['length_mm']:.2f} mm")
                print(f"  - Avg Diam: {stats['avg_diameter']:.2f} mm")
                print(f"  - Isthmus:  {stats['isthmus_diameter']:.2f} mm")
                
                out_dir = DATA / "canal" / patient_name
                os.makedirs(out_dir, exist_ok=True)
                skel_img = nib.Nifti1Image(stats['skeleton'], ct_img.affine)
                nib.save(skel_img, out_dir / f"{bone_name}_skeleton.nii.gz")
                
                with open(out_dir / f"{bone_name}_report.txt", "w") as f:
                    f.write(f"Bone: {bone_name}\n")
                    f.write(f"Length: {stats['length_mm']:.2f} mm\n")
                    f.write(f"Avg_Diameter: {stats['avg_diameter']:.2f} mm\n")
                    f.write(f"Isthmus_Diameter: {stats['isthmus_diameter']:.2f} mm\n")
                    f.write(f"Max_Diameter: {stats['max_diameter']:.2f} mm\n")
            else:
                print(f"  Warning: No canal detected for {bone_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mr", action="store_true", help="Process as MRI (disables HU thresholds)")
    args = parser.parse_args()
    process_patient_canal(args.name, is_mr=args.mr)
