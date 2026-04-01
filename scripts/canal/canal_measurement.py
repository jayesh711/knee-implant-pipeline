import os
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from config import DATA, HU_CANAL_MIN, HU_CANAL_MAX

def calculate_canal_parameters(bone_mask, ct_data, spacing):
    """
    Extract medullary canal parameters from a bone mask and CT intensities.
    Refined with Shaft-Only constraints for full-leg CT accuracy.
    """
    # 1. Bounding box cropping to save memory
    coords = np.argwhere(bone_mask)
    if coords.size == 0:
        return None
    
    z_min, y_min, x_min = coords.min(0)
    z_max, y_max, x_max = coords.max(0)
    
    pad = 5
    z_min = max(0, z_min - pad); y_min = max(0, y_min - pad); x_min = max(0, x_min - pad)
    z_max = min(bone_mask.shape[0], z_max + pad)
    y_max = min(bone_mask.shape[1], y_max + pad)
    x_max = min(bone_mask.shape[2], x_max + pad)
    
    cropped_mask = bone_mask[z_min:z_max, y_min:y_max, x_min:x_max]
    cropped_ct = ct_data[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # 2. Isolate potential canal space
    canal_mask = (cropped_mask > 0) & (cropped_ct >= HU_CANAL_MIN) & (cropped_ct <= HU_CANAL_MAX)
    
    # NEW: Shaft-Only Constraint (Ignore hip/pelvis and knee flares)
    # Filter by vertical percentile (25% to 75% of bone height)
    z_voxels = np.argwhere(cropped_mask)[:, 0]
    z_start = np.percentile(z_voxels, 25)
    z_end = np.percentile(z_voxels, 75)
    
    shaft_mask = np.zeros_like(canal_mask)
    shaft_mask[int(z_start):int(z_end), :, :] = 1
    canal_mask = canal_mask & shaft_mask
    
    # NEW: Vertical Continuity (Largest Connected Component)
    # Removes scattered noise in cancellous bone regions
    lbl_canal, num_lbls = label(canal_mask)
    if num_lbls > 0:
        # Keep only the largest structure (the main canal)
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
    z_len = (np.max(skel_coords[:, 0]) - np.min(skel_coords[:, 0])) * spacing[0]
    
    # 6. Map back
    skeleton_full = np.zeros_like(bone_mask, dtype=np.uint8)
    skeleton_full[z_min:z_max, y_min:y_max, x_min:x_max] = skeleton_cropped.astype(np.uint8)
    
    return {
        "avg_diameter": avg_diameter,
        "isthmus_diameter": isthmus_diameter,
        "max_diameter": max_diameter,
        "length_mm": z_len,
        "skeleton": skeleton_full
    }

def process_patient_canal(patient_name):
    print(f"\n--- Medullary Canal Analysis: {patient_name} ---")
    
    # Load CT (PREPPED)
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
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()
    
    # [OPTIMIZATION] Avoid loading 9GB float64 array with get_fdata()
    # Use ArrayProxy (dataobj) to slice and threshold simultaneously
    print(f"    Scanning volume for canal identification (Memory-Safe)...")
    
    # Label IDs (from config or typical TotalSeg)
    femur_labels = [76, 75, 44, 25, 24]
    tibia_labels = [46, 45, 27, 26]
    
    for bone_name, target_labels in [("femur", femur_labels), ("tibia", tibia_labels)]:
        print(f"Checking {bone_name}...")
        bone_id = next((l for l in target_labels if l in np.unique(seg_data)), None)
        
        mask = None
        if bone_id:
            # Sliced access for AI segment
            mask = (np.asarray(seg_img.dataobj) == bone_id).astype(np.uint8)
            voxel_volume = np.prod(spacing) / 1000.0 # in cc
            bone_vol_cc = np.sum(mask) * voxel_volume
            
            if bone_vol_cc < 500: # Threshold for a typical adult femur/tibia
                print(f"  Warning: AI segment for {bone_name} is too small ({bone_vol_cc:.1f}cc).")
                mask = None
        
        if mask is None:
            print(f"  [CANAL FALLBACK] Applying Anatomical Z-Sorting for {bone_name}...")
            # Identifiy components in a 2x lower-res space directly from disk
            ds = 2
            mask_ds = (np.asarray(img.dataobj[::ds, ::ds, ::ds]) > 250).astype(np.uint8)
            lbls_ds = measure.label(mask_ds)
            props = measure.regionprops(lbls_ds)
            
            # Filter and sort by Z-centroid (Vertical position)
            bone_props = [p for p in props if p.area > 5000]
            bone_props.sort(key=lambda x: x.centroid[2], reverse=True) # Top to Bottom (Z-axis)
            
            if len(bone_props) < 2:
                print(f"    Error: Found only {len(bone_props)} bone structures. Need at least 2.")
                mask = None
            else:
                # Heuristic: [Pelvis, Femur, Tibia] or [Femur, Tibia]
                if bone_name == "femur":
                    targetp = bone_props[0] if len(bone_props) == 2 else bone_props[1]
                else: # tibia
                    targetp = bone_props[-1]
                
                print(f"    Selected component at Z-centroid {targetp.centroid[2] * ds:.1f} for {bone_name} canal.")
                
                # Crop subvolume for high-res canal analysis
                minr, minc, minz, maxr, maxc, maxz = targetp.bbox
                minr, minc, minz = minr*ds, minc*ds, minz*ds
                maxr, maxc, maxz = maxr*ds, maxc*ds, maxz*ds
                
                # Load only the bone region sub-volume
                sub_vol_data = np.asarray(img.dataobj[minr:maxr, minc:maxc, minz:maxz])
                mask = (sub_vol_data > 250).astype(np.uint8)
                bone_slice = sub_vol_data # Use for canal density
            
        if mask is not None:
            z_offset = minz if 'minz' in locals() else 0
            # Pass only the sub-volume to calculate_canal_parameters
            stats = calculate_canal_parameters(mask, bone_slice if 'bone_slice' in locals() else img.dataobj, spacing, z_offset=z_offset)
            
            if stats:
                print(f"  {bone_name.upper()} CANAL DETECTED")
                print(f"  - Length:   {stats['length_mm']:.2f} mm")
                print(f"  - Avg Diam: {stats['avg_diameter']:.2f} mm")
                print(f"  - Isthmus:  {stats['isthmus_diameter']:.2f} mm")
                
                # Save skeleton for visualization
                out_dir = DATA / "canal" / patient_name
                os.makedirs(out_dir, exist_ok=True)
                skel_img = nib.Nifti1Image(stats['skeleton'], ct_img.affine)
                nib.save(skel_img, out_dir / f"{bone_name}_skeleton.nii.gz")
                
                # Write simple text report
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
    args = parser.parse_args()
    process_patient_canal(args.name)
