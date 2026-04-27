import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from scipy.ndimage import label, binary_fill_holes, binary_dilation, generate_binary_structure
from skimage import measure
from pathlib import Path
import argparse
from config import DATA, HU_BONE_MIN, HU_METAL_MIN

def get_bbox(mask, margin=15):
    coords = np.argwhere(mask)
    if coords.size == 0: return None
    min_c = np.maximum(0, coords.min(axis=0) - margin)
    max_c = np.minimum(mask.shape, coords.max(axis=0) + margin)
    return min_c, max_c

def draw_cylinder_in_mask(mask, p1, p2, radius):
    z_min, z_max = int(min(p1[2], p2[2])), int(max(p1[2], p2[2]))
    if z_max == z_min: return
    for z in range(z_min, z_max + 1):
        t = (z - p1[2]) / (p2[2] - p1[2])
        cx = p1[0] + t * (p2[0] - p1[0])
        cy = p1[1] + t * (p2[1] - p1[1])
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask[:, :, z] |= (dist_sq <= radius**2).astype(np.uint8)

def advanced_femur_reconstruction(name="PreOp_CT_Lower_Extremity"):
    print(f"--- ADVANCED HYBRID RECONSTRUCTION v4: {name} ---")
    
    total_path = DATA / "segmentations" / "phase1" / f"{name}_total.nii.gz"
    raw_path = DATA / "NIfTI" / f"{name}_raw.nii.gz"
    
    img_t = nib.load(str(total_path))
    data_t = img_t.get_fdata(dtype=np.float32)
    affine = img_t.affine
    
    # Target Femur (Label 76 for Right)
    femur_ai_mask = (data_t == 76).astype(np.uint8)
    if np.sum(femur_ai_mask) == 0:
        femur_ai_mask = (data_t == 75).astype(np.uint8)
    
    # 1. OPTIMIZED CROP: Find initial bbox and expand it for search
    print("  Determining optimized processing sub-volume...")
    bbox = get_bbox(femur_ai_mask, margin=50) # Large margin for recovery
    # Force superior extension for missing head
    bbox[1][2] = data_t.shape[2] # Extend to top of volume
    
    min_c, max_c = bbox
    femur_ai_mask = femur_ai_mask[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    data_r = nib.load(str(raw_path)).get_fdata(dtype=np.float32)[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ min_c + affine[:3, 3]
    
    # 2. Create Search Region (Fast on cropped volume)
    print("  Creating spatial search region for HU recovery...")
    struct = generate_binary_structure(3, 1)
    # Dilation to bridge internal gaps and search for head
    search_region = binary_dilation(femur_ai_mask, structure=struct, iterations=30).astype(np.uint8)
    
    # Extend search region Superiorly (Z+)
    z_top = np.argwhere(femur_ai_mask)[:, 2].max()
    search_region[:, :, z_top:] = 1 # Allow full recovery at the top
    
    # 3. Hybrid Recovery
    print("  Applying Hybrid HU Recovery (HU > 150)...")
    hu_bone_mask = ((data_r > HU_BONE_MIN) & (data_r < HU_METAL_MIN)).astype(np.uint8)
    recovered_mask = (hu_bone_mask & search_region) | femur_ai_mask
    
    # 4. Bridge Gap
    lbls, n = label(recovered_mask)
    props = measure.regionprops(lbls)
    props.sort(key=lambda x: x.area, reverse=True)
    
    if len(props) >= 2:
        # Sort top 2 by Z
        main_frags = sorted(props[:2], key=lambda x: x.bbox[2])
        d_frag, p_frag = main_frags[0], main_frags[1]
        
        z_gap_start = d_frag.bbox[5]
        z_gap_end = p_frag.bbox[2]
        
        if z_gap_end > z_gap_start:
            print(f"  Bridging Shaft Gap: Z={z_gap_start} to Z={z_gap_end}")
            slice_d = recovered_mask[:, :, z_gap_start-5:z_gap_start]
            slice_p = recovered_mask[:, :, z_gap_end:z_gap_end+5]
            
            c_d = np.mean(np.argwhere(slice_d), axis=0)[:2]
            c_p = np.mean(np.argwhere(slice_p), axis=0)[:2]
            
            r_d = np.sqrt(np.sum(slice_d) / (5 * np.pi))
            r_p = np.sqrt(np.sum(slice_p) / (5 * np.pi))
            
            draw_cylinder_in_mask(recovered_mask, [c_d[0], c_d[1], z_gap_start-2], [c_p[0], c_p[1], z_gap_end+2], (r_d + r_p)/2)
    
    recovered_mask = binary_fill_holes(recovered_mask).astype(np.uint8)
    
    # 5. Meshing
    print("  Generating high-quality mesh...")
    verts, faces, _, _ = measure.marching_cubes(recovered_mask, level=0.5)
    world_verts = (new_affine[:3, :3] @ verts.T).T + new_affine[:3, 3]
    
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(world_verts, faces))
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_close_holes", maxholesize=100)
    ms.apply_coord_taubin_smoothing(stepsmoothnum=15)
    
    if ms.current_mesh().face_number() > 500000:
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=500000)
        
    out_path = DATA / "meshes" / f"{name}_advanced_femur.stl"
    ms.save_current_mesh(str(out_path))
    print(f"SUCCESS: Advanced Femur output to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="PreOp_CT_Lower_Extremity")
    args = parser.parse_args()
    advanced_femur_reconstruction(args.name)
