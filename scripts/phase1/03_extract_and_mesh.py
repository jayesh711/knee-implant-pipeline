import nibabel as nib
import numpy as np
import trimesh
import trimesh.repair
import pymeshlab
import json
import sys
import os
from pathlib import Path
from skimage import measure
from skimage.morphology import ball
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing as scipy_binary_closing,
    distance_transform_edt,
    gaussian_filter,
    binary_dilation,
    generate_binary_structure,
)
import gc

# Ensure root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    BASE_DIR, DATA, MAX_TRIANGLES, SMOOTH_ITERS, HU_METAL_MIN,
    TAUBIN_LAMBDA, TAUBIN_MU, MAX_HOLE_DIAMETER_MM,
    CLOSING_MM_FEMUR, CLOSING_MM_TIBIA, JOINT_GAP_MM, COMPONENT_MIN_PCT,
    MESH_SIGMA, REMESHER_TARGET_LEN, HU_BONE_MIN, HU_BONE_MAX
)

_METAL_THRESHOLD = HU_METAL_MIN

# ── TotalSegmentator v2 label maps ────────────────────────────────────────────
_APPENDICULAR_DATASET_JSON = (
    BASE_DIR / "models" / "totalsegmentator"
    / "Dataset304_appendicular_bones_ext_1559subj"
    / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    / "dataset.json"
)

_TOTAL_FEMUR_LABELS = [75, 76]
_TOTAL_TIBIA_LABELS = []

def _detect_bone_labels(seg_data, task_hint=""):
    present = set(np.unique(seg_data[seg_data > 0]).astype(int))
    if not present: return [], []
    max_label = max(present)
    if "clinical" in task_hint.lower():
        femur_ids, tibia_ids = ([1] if 1 in present else []), ([2] if 2 in present else [])
        task = "clinical"
    elif max_label <= 20 and _APPENDICULAR_DATASET_JSON.exists():
        with open(_APPENDICULAR_DATASET_JSON) as f:
            raw = json.load(f)["labels"]
        label_map = {k.lower(): int(v) for k, v in raw.items()}
        femur_ids = [v for k, v in label_map.items() if "femur" in k and v in present]
        tibia_ids = [v for k, v in label_map.items() if "tibia" in k and v in present]
        task = "appendicular_bones"
    else:
        femur_ids = [l for l in _TOTAL_FEMUR_LABELS if l in present]
        tibia_ids = [l for l in _TOTAL_TIBIA_LABELS if l in present]
        task = "total"
    print(f"  [{task}] femur labels={femur_ids}  tibia labels={tibia_ids}")
    return femur_ids, tibia_ids

def _find_segmentations(volume_name):
    phase1 = DATA / "segmentations" / "phase1"
    result = {}
    for key, suffix in [("primary", ""), ("total", "_total"), ("clinical", "_clinical")]:
        path = phase1 / f"{volume_name}{suffix}.nii.gz"
        if not path.exists(): path = phase1 / f"{volume_name}{suffix}.nii"
        if path.exists(): result[key] = path
    return result

def _get_bbox(mask, margin=15):
    coords = np.argwhere(mask)
    if coords.size == 0: return None
    min_c = np.maximum(0, coords.min(axis=0) - margin)
    max_c = np.minimum(mask.shape, coords.max(axis=0) + margin)
    return min_c, max_c

def _voxel_size_from_affine(affine):
    return np.cbrt(np.abs(np.linalg.det(affine[:3, :3])))

def _save_debug_mask(mask, affine, name, volume_name):
    debug_dir = DATA / "segmentations" / "debug" / volume_name
    os.makedirs(debug_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(debug_dir / f"{name}.nii.gz"))

def _load_nifti(path):
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.uint8), img.affine

def _build_mask(seg_data, label_ids):
    m = np.zeros(seg_data.shape, dtype=np.uint8)
    for lbl in label_ids: m[seg_data == lbl] = 1
    return m

def _cap_open_boundaries(mesh, max_loop_size=1000):
    """Robustly close open boundary loops (bone shafts) to ensure watertightness."""
    import trimesh
    import numpy as np

    try:
        if hasattr(mesh, 'is_watertight') and mesh.is_watertight: return mesh
    except: pass

    print(f"    [CAP] Closing open boundaries (Euler: {int(mesh.euler_number)})...")
    try:
        trimesh.repair.fill_holes(mesh)
        
        outline = mesh.outline()
        if not outline or not hasattr(outline, 'entities') or not outline.entities: 
            return mesh
        
        verts = np.array(mesh.vertices.copy())
        faces = np.array(mesh.faces.copy())
        z_min = float(np.min(verts[:, 2]))
        z_max = float(np.max(verts[:, 2]))
        
        for entity in outline.entities:
            loop = getattr(entity, 'nodes', getattr(entity, 'points', None))
            if loop is None: continue
            if len(loop) < 3: continue
            
            loop_pts = verts[loop]
            centroid = np.mean(loop_pts, axis=0)
            
            # Defensive check for shaft
            z_val = float(centroid[2])
            is_near_z = (abs(z_val - z_max) < 5.0) or (abs(z_val - z_min) < 5.0)
            is_large = (len(loop) > 50)
            
            if is_near_z or is_large:
                center_idx = len(verts)
                new_v = np.array(centroid).reshape(1, 3)
                verts = np.vstack([verts, new_v])
                
                for j in range(len(loop)):
                    v1 = int(loop[j])
                    v2 = int(loop[(j + 1) % len(loop)])
                    faces = np.vstack([faces, [center_idx, v1, v2]])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        trimesh.repair.fix_normals(mesh)
        return mesh
    except Exception as e:
        print(f"    [CAP] Error at {bone_name if 'bone_name' in locals() else 'mesh'}: {str(e)}")
        return mesh

def extract_mesh(mask, affine, raw_data=None, has_metal=False, closing_mm=3.0, bone_name="bone", is_clinical=False):
    mask = mask.copy()
    bbox = _get_bbox(mask)
    if not bbox: return None
    min_c, max_c = bbox
    mask = mask[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    if raw_data is not None: raw_data = raw_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    new_affine = affine.copy()
    new_affine[:3, 3] = affine[:3, :3] @ min_c + affine[:3, 3]
    affine = new_affine

    if has_metal and raw_data is not None: mask[raw_data > _METAL_THRESHOLD] = 0
    if not np.any(mask): return None

    print(f"    Step 1: Closing bone marrow (radius={closing_mm}mm)...")
    v_size = _voxel_size_from_affine(affine)
    mask = scipy_binary_closing(mask, structure=ball(max(1, int(round(closing_mm / v_size))))).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)

    comp_arr = measure.label(mask)
    if comp_arr.max() > 1:
        largest_id = np.argmax(np.bincount(comp_arr.flat)[1:]) + 1
        mask = (comp_arr == largest_id).astype(np.uint8)

    if is_clinical and raw_data is not None:
        print(f"    Step 2: Surface extraction from Raw CT ({HU_BONE_MIN} HU)...")
        dilated = binary_dilation(mask, iterations=5)
        search_vol = raw_data.copy()
        search_vol[~dilated] = -1024
        verts, faces, _, _ = measure.marching_cubes(search_vol, level=HU_BONE_MIN)
    else:
        print(f"    Step 2: Surface extraction from AI Mask...")
        dt = distance_transform_edt(mask) - distance_transform_edt(1 - mask)
        verts, faces, _, _ = measure.marching_cubes(dt, level=0)

    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]
    mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)

    print(f"    Step 3: Mesh Optimization (Smoothing/Decimation)...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.apply_filter("meshing_remove_duplicate_vertices")
    try: ms.apply_filter("meshing_isotropic_explicit_remeshing", targetlen=pymeshlab.PureValue(0.8), iterations=3)
    except: ms.apply_filter("meshing_isotropic_explicit_remeshing", customtargetlen=0.8)
    ms.apply_coord_taubin_smoothing(stepsmoothnum=20, lambda_=0.5, mu=-0.53)
    target_f = 150000 if bone_name == "femur" else 120000
    if ms.current_mesh().face_number() > target_f:
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=target_f)
    
    final_m = ms.current_mesh()
    mesh = trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())
    mesh = _cap_open_boundaries(mesh)
    return mesh

def process_volume(volume_name="S0001", has_metal=False):
    seg_paths = _find_segmentations(volume_name)
    if not seg_paths: return
    
    raw_data = None
    raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    if raw_path.exists(): raw_data = nib.load(str(raw_path)).get_fdata(dtype=np.float16)

    femur_mask, tibia_mask, f_affine, t_affine = None, None, None, None
    is_clinical = "clinical" in seg_paths

    if is_clinical:
        seg_j, aff_j = _load_nifti(seg_paths["clinical"])
        f_ids, t_ids = _detect_bone_labels(seg_j, "clinical")
        if f_ids: femur_mask, f_affine = _build_mask(seg_j, f_ids), aff_j
        if t_ids: tibia_mask, t_affine = _build_mask(seg_j, t_ids), aff_j
    
    if femur_mask is None or tibia_mask is None:
        # Fallback to TS
        if "total" in seg_paths:
            seg_t, aff_t = _load_nifti(seg_paths["total"])
            f_ids_t, _ = _detect_bone_labels(seg_t, "total")
            if f_ids_t and femur_mask is None: femur_mask, f_affine = _build_mask(seg_t, f_ids_t), aff_t
        if "primary" in seg_paths:
            seg_p, aff_p = _load_nifti(seg_paths["primary"])
            _, t_ids_p = _detect_bone_labels(seg_p, "appendicular")
            if t_ids_p and tibia_mask is None: tibia_mask, t_affine = _build_mask(seg_p, t_ids_p), aff_p

    output_dir = DATA / "meshes" / "clinical_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    if femur_mask is not None:
        mesh_f = extract_mesh(femur_mask, f_affine, raw_data, has_metal, CLOSING_MM_FEMUR, "femur", is_clinical)
        if mesh_f: mesh_f.export(str(output_dir / f"{volume_name}_femur_full.stl"))
    if tibia_mask is not None:
        mesh_t = extract_mesh(tibia_mask, t_affine, raw_data, has_metal, CLOSING_MM_TIBIA, "tibia", is_clinical)
        if mesh_t: mesh_t.export(str(output_dir / f"{volume_name}_tibia_full.stl"))

    _verify_gap(volume_name, output_dir)
    _save_final_segmentation(volume_name, femur_mask, tibia_mask, f_affine)

def _save_final_segmentation(volume_name, femur_mask, tibia_mask, affine):
    out_dir = DATA / "segmentations" / "final"
    os.makedirs(out_dir, exist_ok=True)
    final = np.zeros(femur_mask.shape if femur_mask is not None else (1,1,1), dtype=np.uint8)
    if femur_mask is not None: final[femur_mask > 0] = 1
    if tibia_mask is not None: final[tibia_mask > 0] = 2
    nib.save(nib.Nifti1Image(final, affine), str(out_dir / f"{volume_name}_final.nii.gz"))

def _verify_gap(volume_name, mesh_dir):
    f_p, t_p = mesh_dir / f"{volume_name}_femur_full.stl", mesh_dir / f"{volume_name}_tibia_full.stl"
    if not f_p.exists() or not t_p.exists(): return
    try:
        f_m, t_m = trimesh.load(str(f_p)), trimesh.load(str(t_p))
        _, dist, _ = trimesh.proximity.closest_point(f_m, t_m.sample(5000))
        print(f"\n  [GAP] Min: {dist.min():.2f}mm | Mean: {dist.mean():.2f}mm")
    except Exception as e: print(f"  [GAP] Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001")
    parser.add_argument("--has-metal", action="store_true")
    args = parser.parse_args()
    process_volume(args.name, has_metal=args.has_metal)
