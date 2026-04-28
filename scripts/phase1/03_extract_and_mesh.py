import nibabel as nib
import numpy as np
import trimesh
import trimesh.repair
import pymeshlab
import json
import sys
import os
from pathlib import Path

# Ensure root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

print("--- Pipeline Script Initialized ---", flush=True)

from skimage import measure

from skimage.morphology import ball
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing as scipy_binary_closing,
    distance_transform_edt,
    binary_dilation,
    generate_binary_structure,
)
import gc
from pathlib import Path
import os
from config import (
    BASE_DIR, DATA, MAX_TRIANGLES, SMOOTH_ITERS, HU_METAL_MIN,
    TAUBIN_LAMBDA, TAUBIN_MU, MAX_HOLE_DIAMETER_MM,
    CLOSING_MM_FEMUR, CLOSING_MM_TIBIA, JOINT_GAP_MM, COMPONENT_MIN_PCT,
)

# Metal filter is opt-in (has_metal=True only). Raw CT is clamped to 3000 HU at ingest,
# so HU_METAL_MIN (2500) is only safe to apply when an implant is confirmed present.
_METAL_THRESHOLD = HU_METAL_MIN

# ── TotalSegmentator v2 label maps ────────────────────────────────────────────
# appendicular_bones task (Dataset304): tibia=2, femur_auxiliary=13 (often absent)
_APPENDICULAR_DATASET_JSON = (
    BASE_DIR / "models" / "totalsegmentator"
    / "Dataset304_appendicular_bones_ext_1559subj"
    / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    / "dataset.json"
)

# total task (v2 class_map): femur_left=75, femur_right=76. Tibia is NOT in total task.
_TOTAL_FEMUR_LABELS = [75, 76]
_TOTAL_TIBIA_LABELS = []   # tibia absent from total task; use appendicular_bones instead


def _detect_bone_labels(seg_data):
    """
    Identify bone labels in seg_data. Returns (femur_ids, tibia_ids) that are
    confirmed present (non-zero voxel count) in the array.

    Detection strategy:
      max_label <= 20  → appendicular_bones task → parse Dataset304 json by name
      max_label >  20  → total task → use TotalSegmentator v2 hardcoded class_map
    """
    present = set(np.unique(seg_data[seg_data > 0]).astype(int))
    if not present:
        return [], []

    max_label = max(present)

    if max_label <= 20 and _APPENDICULAR_DATASET_JSON.exists():
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
    """
    Return a dict with paths to available segmentation files for this volume.

    Keys:
      'primary'  — appendicular_bones output (preferred source for tibia)
      'total'    — total-task output (preferred source for femur)
    """
    phase1 = DATA / "segmentations" / "phase1"
    result = {}

    primary_candidates = [
        phase1 / f"{volume_name}.nii",
        phase1 / f"{volume_name}.nii.gz",
        phase1 / volume_name / "multilabel.nii.gz",
    ]
    found_primary = next((p for p in primary_candidates if p.exists()), None)
    if found_primary:
        result["primary"] = found_primary

    total_candidates = [
        phase1 / f"{volume_name}_total.nii",
        phase1 / f"{volume_name}_total.nii.gz",
        phase1 / f"{volume_name}_total" / "multilabel.nii.gz",
    ]
    found_total = next((p for p in total_candidates if p.exists()), None)
    if found_total:
        result["total"] = found_total

    return result


def _get_bbox(mask, margin=15):
    """Find the bounding box of the non-zero voxels in the mask, with a margin."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    min_c = np.maximum(0, coords.min(axis=0) - margin)
    max_c = np.minimum(mask.shape, coords.max(axis=0) + margin)
    return min_c, max_c


def _voxel_size_from_affine(affine):
    """Compute the isotropic voxel size (mm) from the affine matrix."""
    return np.cbrt(np.abs(np.linalg.det(affine[:3, :3])))


def _save_debug_mask(mask, affine, name, volume_name):
    """Save an intermediate mask for debugging in 3D Slicer."""
    debug_dir = DATA / "segmentations" / "debug" / volume_name
    os.makedirs(debug_dir, exist_ok=True)
    out_path = debug_dir / f"{name}.nii.gz"
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(img, str(out_path))
    # print(f"    [DEBUG] Saved {out_path.name}")


def _diagnose_components(mask, affine, min_print_voxels=1000):
    """Run a spatial analysis on each retained component for debugging."""
    if not np.any(mask):
        return
    voxel_size_mm = _voxel_size_from_affine(affine)
    
    # Crop to bone region for speed
    bbox = _get_bbox(mask, margin=2)
    if not bbox: return
    min_c, max_c = bbox
    sl = tuple(slice(mn, mx) for mn, mx in zip(min_c, max_c))
    crop = mask[sl]
    
    labels = measure.label(crop)
    regions = measure.regionprops(labels)
    if not regions: return
    
    # Find the largest component (assumed to be the main femur/tibia)
    largest = max(regions, key=lambda r: r.area)
    largest_centroid = np.array(largest.centroid)
    
    print(f"\n    [DIAGNOSTIC] Analyzing {len(regions)} components (showing >{min_print_voxels} vox):")
    for r in regions:
        if r.area < min_print_voxels and r.label != largest.label:
            continue
        centroid = np.array(r.centroid)
        distance_voxels = np.linalg.norm(centroid - largest_centroid)
        distance_mm = distance_voxels * voxel_size_mm
        volume_cm3 = r.area * (voxel_size_mm ** 3) / 1000
        is_main = (r.label == largest.label)
        status = "[MAIN]" if is_main else ""
        print(f"      Component {r.label}: {r.area:,} voxels ({volume_cm3:.1f} cm³), "
              f"dist from main: {distance_mm:.1f} mm {status}")


def _keep_spatially_linked_components(mask, affine, max_distance_mm=40, min_fragment_voxels=5000):
    """
    Keep the largest component plus any smaller components whose centroid
    is within max_distance_mm of the largest component's centroid.
    """
    if not np.any(mask):
        return mask
    voxel_size_mm = _voxel_size_from_affine(affine)
    
    # Spatial Optimization
    bbox = _get_bbox(mask, margin=5)
    if not bbox: return mask
    min_c, max_c = bbox
    sl = tuple(slice(mn, mx) for mn, mx in zip(min_c, max_c))
    crop = mask[sl]
    
    labels = measure.label(crop)
    regions = measure.regionprops(labels)
    if not regions: return mask
    
    largest = max(regions, key=lambda r: r.area)
    largest_centroid = np.array(largest.centroid)
    
    print(f"    Spatially-aware filtering: keeping largest + fragments within {max_distance_mm}mm...")
    
    # Build kept mask in crop space
    kept_mask = (labels == largest.label)
    kept_count = 1
    for r in regions:
        if r.label == largest.label: continue
        if r.area < min_fragment_voxels: continue
        
        centroid_dist_mm = np.linalg.norm(np.array(r.centroid) - largest_centroid) * voxel_size_mm
        if centroid_dist_mm <= max_distance_mm:
            kept_mask |= (labels == r.label)
            kept_count += 1
            
    print(f"    Retained {kept_count}/{len(regions)} components.")
    
    new_mask = np.zeros_like(mask)
    new_mask[sl] = kept_mask.astype(np.uint8)
    return new_mask






def _fill_thin_cortex_gaps(mask, affine, radius_mm=1.5):
    """Fill small holes in the cortex using a conservative morphological closing."""
    voxel_size = _voxel_size_from_affine(affine)
    radius_vox = max(1, int(round(radius_mm / voxel_size)))
    print(f"    Filling thin cortex gaps (radius={radius_mm}mm -> {radius_vox} voxels)...")
    return scipy_binary_closing(mask, structure=ball(radius_vox)).astype(np.uint8)


def _load_nifti(path):
    """Load NIfTI data safely and efficiently."""
    img = nib.load(str(path))
    # Direct uint8 access via dataobj avoids memory spikes
    return np.asarray(img.dataobj, dtype=np.uint8), img.affine


def _build_mask(seg_data, label_ids):
    """Build a binary mask from multiple label IDs."""
    m = np.zeros(seg_data.shape, dtype=np.uint8)
    for lbl in label_ids:
        m[seg_data == lbl] = 1
    return m



# ── Joint Gap Enforcement ─────────────────────────────────────────────────────
# -- Joint Gap Enforcement -----------------------------------------------------
def enforce_joint_gap(femur_mask, tibia_mask, min_gap_voxels=4):
    """
    Ensure a minimum physical gap between femur and tibia masks.
    Only processes the region where both bones are present (the joint)
    to save memory on full-leg scans.
    """
    # Find Z-range of overlap where joint gap is relevant
    f_coords = np.argwhere(femur_mask)
    t_coords = np.argwhere(tibia_mask)
    if f_coords.size == 0 or t_coords.size == 0:
        return femur_mask, tibia_mask

    # Joint is where they meet. Typically femur is proximal (higher Z), tibia is distal (lower Z).
    # Overlap region in Z:
    z_min = max(f_coords[:, 2].min(), t_coords[:, 2].min())
    z_max = min(f_coords[:, 2].max(), t_coords[:, 2].max())
    
    # Add generous margin for EDT accuracy
    margin = min_gap_voxels * 10
    z_min_m = max(0, z_min - margin)
    z_max_m = min(femur_mask.shape[2], z_max + margin)

    # Within this Z-range, find the combined XY bounding box
    combined_joint = (femur_mask[:, :, z_min_m:z_max_m] > 0) | (tibia_mask[:, :, z_min_m:z_max_m] > 0)
    xy_coords = np.argwhere(combined_joint)
    if xy_coords.size == 0:
        return femur_mask, tibia_mask
        
    min_xy = np.maximum(0, xy_coords[:, :2].min(axis=0) - margin)
    max_xy = np.minimum(np.array(femur_mask.shape[:2]), xy_coords[:, :2].max(axis=0) + margin)
    
    sl = (slice(min_xy[0], max_xy[0]), 
          slice(min_xy[1], max_xy[1]), 
          slice(z_min_m, z_max_m))

    crop_shape = (max_xy[0]-min_xy[0], max_xy[1]-min_xy[1], z_max_m-z_min_m)
    print(f"    Gap enforcement: cropping {femur_mask.shape} -> {crop_shape} for EDT")

    # Crop masks
    f_crop = femur_mask[sl].copy()
    t_crop = tibia_mask[sl].copy()

    # Compute distance of every femur voxel from tibia boundary
    # and vice versa. Where distance < min_gap, erode that bone.
    femur_dist = distance_transform_edt(~t_crop.astype(bool))
    tibia_dist = distance_transform_edt(~f_crop.astype(bool))

    # Remove voxels too close to the other bone
    f_crop = f_crop & (femur_dist >= min_gap_voxels)
    t_crop = t_crop & (tibia_dist >= min_gap_voxels)

    # Write back to full-size masks
    removed_f = int(np.sum(femur_mask[sl]) - np.sum(f_crop))
    removed_t = int(np.sum(tibia_mask[sl]) - np.sum(t_crop))
    
    print(f"    Gap enforcement: removed {removed_f:,} voxels from femur, {removed_t:,} voxels from tibia")
    if removed_f > 5000:
        print(f"    [WARNING] High femur erosion ({removed_f:,} vox). Distal femur may be mislabeled or touching too much.")

    femur_mask[sl] = f_crop.astype(np.uint8)
    tibia_mask[sl] = t_crop.astype(np.uint8)

    return femur_mask.astype(np.uint8), tibia_mask.astype(np.uint8)




def _extend_femur_hu(femur_mask, raw_data, iterations=40):
    """
    Hybrid HU-extension: Use AI segmentation as a seed and extend proximal coverage
    using HU thresholding within a dilated search region.
    """
    if raw_data is None:
        return femur_mask

    print(f"    Hybrid HU-extension ({iterations} iters): Expanding AI seed into proximal bone signal...")

    
    # 1. Create spatial search region by dilating the AI seed
    struct = generate_binary_structure(3, 1)
    # Convert to float32 mask temporarily for speed if needed, but bool is fine
    search_region = binary_dilation(femur_mask.astype(bool), structure=struct, iterations=iterations)
    
    # 2. Allow full recovery at the top of the volume (proximal head)
    coords = np.argwhere(femur_mask)
    if coords.size > 0:
        z_max = coords[:, 2].max()
        search_region[:, :, z_max:] = True
        
    # 3. Bone thresholding (150 HU to 2500 HU)
    # raw_data is likely float16 now for memory
    hu_bone = (raw_data > 150) & (raw_data < 2500)
    
    # 4. Union: (HU-Bone in search region) OR (Original AI mask)
    extended = (hu_bone & search_region) | femur_mask.astype(bool)
    
    return extended.astype(np.uint8)



def _cap_open_boundaries(mesh, max_loop_size=400):
    """
    Detect open boundary loops and close them with a triangle fan to a 
    projected centroid, creating a smooth rounded cap.
    """
    import trimesh
    import numpy as np

    try:
        # Find open boundary loops
        print(f"    [CAP] Detecting boundaries via outline...")
        outline = mesh.outline()
        boundary_loops = []
        for entity in outline.entities:
            # Check both .nodes and .points for trimesh compatibility
            loop = getattr(entity, 'nodes', getattr(entity, 'points', None))
            if loop is not None and len(loop) > 3:
                boundary_loops.append(loop)

        if not boundary_loops:
            print(f"    [CAP] No open boundaries found.")
            return mesh

        print(f"    [CAP] Found {len(boundary_loops)} candidate loops.")
        verts = mesh.vertices.copy().tolist()
        faces = mesh.faces.copy().tolist()
        new_vert_indices = []

        bone_centroid = mesh.vertices.mean(axis=0)

        for i, loop in enumerate(boundary_loops):
            if len(loop) > max_loop_size or len(loop) < 3:
                continue
            
            loop_pts = mesh.vertices[loop]
            centroid = loop_pts.mean(axis=0)
            
            # Robust normal calculation
            try:
                # 1. Try plane fit
                _, loop_normal = trimesh.geometry.plane_fit(loop_pts)
            except Exception as e:
                # 2. Fallback: Check if it's a truncation hole (near Z-bounds)
                z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
                if abs(centroid[2] - z_max) < 5.0:
                    loop_normal = np.array([0.0, 0.0, 1.0])
                elif abs(centroid[2] - z_min) < 5.0:
                    loop_normal = np.array([0.0, 0.0, -1.0])
                else:
                    # 3. Final Fallback: Geometric normal
                    v1 = (loop_pts[len(loop)//3] - loop_pts[0]).flatten()
                    v2 = (loop_pts[2*len(loop)//3] - loop_pts[0]).flatten()
                    loop_normal = np.cross(v1, v2)
                    norm = np.linalg.norm(loop_normal)
                    if norm < 1e-9: continue
                    loop_normal /= norm
            
            # Ensure normal points away from bone center
            if np.dot(loop_normal, centroid - bone_centroid) < 0:
                loop_normal = -loop_normal

                
            # Project centroid outward to create a rounded dome (3mm height)
            centroid = centroid + loop_normal * 3.0
            
            centroid_idx = len(verts)
            verts.append(centroid.tolist())
            new_vert_indices.append(centroid_idx)

            # Create triangle fan
            for j in range(len(loop)):
                # Ensure CCW winding
                faces.append([centroid_idx, loop[(j + 1) % len(loop)], loop[j]])

        new_mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
        trimesh.repair.fix_normals(new_mesh)
        
        # Smooth the new dome region
        if new_vert_indices:
            print(f"    [CAP] Smoothing {len(new_vert_indices)} new caps...")
            mask = np.zeros(len(new_mesh.vertices), dtype=bool)
            adj = new_mesh.vertex_adjacency_graph
            near_dome = set(new_vert_indices)
            for _ in range(4): 
                neighbors = set()
                for v in near_dome:
                    if v in adj:
                        neighbors.update(adj.neighbors(v))
                near_dome |= neighbors
            mask[list(near_dome)] = True
            
            for _ in range(20):
                new_pos = new_mesh.vertices.copy()
                for v_idx in np.where(mask)[0]:
                    if v_idx in adj:
                        n = list(adj.neighbors(v_idx))
                        if n: new_pos[v_idx] = new_mesh.vertices[n].mean(axis=0)
                new_mesh.vertices = new_pos

        
        return new_mesh

    except Exception as e:
        print(f"    [CAP] Error during capping: {e}. Returning un-capped mesh.")
        return mesh



def extract_mesh(mask, affine, raw_data=None, has_metal=False, closing_mm=3.0, bone_name="bone"):
    """
    Generate a clean, full-length bone mesh from a binary voxel mask.
    Optimized with spatial cropping for large volumes.

    closing_mm: morphological closing radius in physical mm.
      Converted to voxels using the affine. Default 3mm for femur, 2mm for tibia.
    """
    mask = mask.copy()

    # Spatial Cropping Optimization
    print(f"    Analyzing bone extent for spatial optimization...")
    bbox = _get_bbox(mask)
    if bbox:
        min_c, max_c = bbox
        # print(f"    Cropping to region: {min_c} -> {max_c} (Shape: {max_c - min_c})")

        # Crop mask and raw data
        mask = mask[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
        if raw_data is not None:
            raw_data = raw_data[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]

        # Update affine for the new origin
        new_affine = affine.copy()
        new_affine[:3, 3] = affine[:3, :3] @ min_c + affine[:3, 3]
        affine = new_affine
    else:
        print(f"    Warning: Mask is empty, skipping extraction.")
        return None

    if has_metal and raw_data is not None:
        print(f"    Filtering hardware (>{_METAL_THRESHOLD} HU)...")
        mask[raw_data > _METAL_THRESHOLD] = 0

    if not np.any(mask):
        return None

    # Adaptive closing: convert physical mm to voxel radius
    voxel_size = _voxel_size_from_affine(affine)
    closing_radius = max(1, int(round(closing_mm / voxel_size)))
    print(f"    Bridging intra-bone gaps (closing={closing_mm}mm -> ball radius={closing_radius} voxels at {voxel_size:.2f}mm/vox)...")

    # scipy binary_closing uses boolean dtype (~8x less memory than skimage's float64 closing).
    mask = scipy_binary_closing(mask, structure=ball(closing_radius)).astype(np.uint8)

    # Multi-component retention: keep only components with volume >= 50,000 voxels
    MIN_COMPONENT_VOXELS = 50_000
    comp_arr = measure.label(mask)
    n_comps = comp_arr.max()
    if n_comps > 1:
        counts = np.bincount(comp_arr.flat)[1:]
        kept_ids = np.where(counts >= MIN_COMPONENT_VOXELS)[0] + 1
        if len(kept_ids) == 0:
            kept_ids = [np.argmax(counts) + 1]
        print(f"    Retained {len(kept_ids)}/{n_comps} components (>= {MIN_COMPONENT_VOXELS:,} voxels)")
        mask = np.isin(comp_arr, kept_ids).astype(np.uint8)

    # Internal void closing (marrow spaces) - Robust 3D fill
    print(f"    Filling internal voids (marrow)...")
    mask = binary_fill_holes(mask).astype(np.uint8)

    if not np.any(mask):
        return None

    # Marching Cubes
    print(f"    Marching Cubes...")
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)

    # Voxel → LPS mm
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]

    # ── Mesh Refinement Pipeline ──────────────────────────────────────────────
    print(f"    Trimesh hole-fill pass...")
    tri_mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    trimesh.repair.fill_holes(tri_mesh)
    trimesh.repair.fix_normals(tri_mesh)

    print(f"    PyMeshLab refinement...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(tri_mesh.vertices, tri_mesh.faces))

    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")

    try:
        max_edges = int(MAX_HOLE_DIAMETER_MM / voxel_size)
        ms.apply_filter("meshing_close_holes", maxholesize=max_edges)
    except Exception as e:
        print(f"    Warning: PyMeshLab hole filling skipped ({e})")

    print(f"    Laplacian pre-smooth (1 iteration)...")
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)

    print(f"    Taubin smoothing ({SMOOTH_ITERS} iterations)...")
    ms.apply_coord_taubin_smoothing(
        stepsmoothnum=SMOOTH_ITERS, lambda_=TAUBIN_LAMBDA, mu=TAUBIN_MU
    )

    # ── Diagnostic: Pre-Decimation Watertightness ──────────────────────────
    temp_m = ms.current_mesh()
    pre_dec = trimesh.Trimesh(vertices=temp_m.vertex_matrix(), faces=temp_m.face_matrix())
    
    if bone_name == "femur":
        predec_dir = DATA / "meshes" / "diagnostics"
        os.makedirs(predec_dir, exist_ok=True)
        predec_path = predec_dir / f"{bone_name}_predec.stl"
        pre_dec.export(str(predec_path))
        print(f"    [DIAGNOSTIC] Pre-decimation watertight: {pre_dec.is_watertight}")

    current_faces = ms.current_mesh().face_number()
    if current_faces > MAX_TRIANGLES:
        print(f"    Decimating {current_faces:,} -> {MAX_TRIANGLES:,} faces...")
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse", targetfacenum=MAX_TRIANGLES
        )

    # Manifold cleanup (standard for clinical meshes)
    ms.apply_filter('meshing_repair_non_manifold_edges')
    ms.apply_filter('meshing_close_holes', maxholesize=200)

    # ── Final Cap & Smoothing ─────────────────────────────────────────────
    print(f"    Capping scan truncation holes...")
    final_m = ms.current_mesh()
    mesh = trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())
    mesh = _cap_open_boundaries(mesh)


    # ── Monitoring & Final Repair ────────────────────────────────────────
    # Count open edges (edges in only 1 face)
    open_edges = len(mesh.edges_sorted[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)])
    non_manifold = len(trimesh.repair.broken_faces(mesh))
    print(f"    Post-cap status: {open_edges} open edges, {non_manifold} non-manifold faces")

    if open_edges > 0:
        print(f"    [REPAIR] Closing remaining {open_edges} open edges...")
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        except Exception as e:
            print(f"    [REPAIR] Final trimesh repair failed: {e}")

    return mesh







def process_volume(volume_name="S0001", has_metal=False):
    """
    Dual-segmentation bone reconstruction strategy (Refined Debug Version).
    """
    seg_paths = _find_segmentations(volume_name)
    if not seg_paths:
        print(f"Error: No segmentation found for '{volume_name}'.")
        return

    print(f"\n--- Dual-Segmentation Bone Reconstruction: {volume_name} ---")
    
    # ── 1. Load Raw Data for extension ──────────────────────────────────────
    raw_data = None
    raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    if raw_path.exists():
        print(f"  Loading raw CT for HU-extension...")
        raw_data = nib.load(str(raw_path)).get_fdata(dtype=np.float16)

    # ── 2. Load AI Sources ──────────────────────────────────────────────────
    femur_primary = None
    tibia_mask = None
    femur_affine = None
    tibia_affine = None

    if "primary" in seg_paths:
        print(f"\n  [STEP 1] Loading primary (appendicular) segmentation...")
        seg_data, affine = _load_nifti(seg_paths["primary"])
        femur_ids, tibia_ids = _detect_bone_labels(seg_data)
        if tibia_ids:
            tibia_mask = _build_mask(seg_data, tibia_ids)
            tibia_affine = affine
            print(f"    Tibia (primary): {int(np.sum(tibia_mask)):,} voxels")
        if femur_ids:
            femur_primary = _build_mask(seg_data, femur_ids)
            femur_affine = affine
            print(f"    Femur (appendicular): {int(np.sum(femur_primary)):,} voxels")
            _save_debug_mask(femur_primary, femur_affine, "femur_appendicular", volume_name)
        del seg_data
        gc.collect()

    femur_total = None
    if "total" in seg_paths:
        print(f"\n  [STEP 2] Loading total-task segmentation...")
        seg_data_t, affine_t = _load_nifti(seg_paths["total"])
        femur_ids_t, _ = _detect_bone_labels(seg_data_t)
        if femur_ids_t:
            femur_total = _build_mask(seg_data_t, femur_ids_t)
            if femur_affine is None: femur_affine = affine_t
            print(f"    Femur (total): {int(np.sum(femur_total)):,} voxels")
            _save_debug_mask(femur_total, femur_affine, "femur_total", volume_name)
        del seg_data_t
        gc.collect()

    # ── 3. Union & Extension ────────────────────────────────────────────────
    if femur_primary is None and femur_total is None:
        print("Error: No femur found.")
        return
    
    print(f"\n  [STEP 3] Union & Hybrid Extension...")
    # True Boolean Union (|)
    if femur_total is not None and femur_primary is not None:
        femur_mask = (femur_total | femur_primary).astype(np.uint8)
    else:
        femur_mask = femur_total if femur_total is not None else femur_primary
    
    print(f"    Femur AI combined: {int(np.sum(femur_mask)):,} voxels")
    _save_debug_mask(femur_mask, femur_affine, "femur_combined_ai", volume_name)

    if raw_data is not None:
        femur_mask = _extend_femur_hu(femur_mask, raw_data, iterations=40)
        print(f"    Femur after extension: {int(np.sum(femur_mask)):,} voxels")
        _save_debug_mask(femur_mask, femur_affine, "femur_combined_extended", volume_name)

    # ── 4. Bone Quality Refinement ──────────────────────────────────────────
    print(f"\n  [STEP 4] Bone Quality Refinement (Spatially-Aware Cleanup)...")
    
    # DIAGNOSTIC: Before cleanup
    _diagnose_components(femur_mask, femur_affine)

    # Use spatial filter to preserve head/shaft even if disconnected, but reject distant debris
    femur_mask = _keep_spatially_linked_components(femur_mask, femur_affine, 
                                                   max_distance_mm=40, 
                                                   min_fragment_voxels=5000)
    print(f"    Femur after cleanup: {int(np.sum(femur_mask)):,} voxels")
    _save_debug_mask(femur_mask, femur_affine, "femur_combined", volume_name)

    if tibia_mask is not None:
        tibia_mask = _keep_spatially_linked_components(tibia_mask, tibia_affine, 
                                                       max_distance_mm=40, 
                                                       min_fragment_voxels=5000)


    # Joint Gap Enforcement (AFTER Extension, BEFORE Final Mesh)
    if tibia_mask is not None:
        voxel_size = _voxel_size_from_affine(femur_affine)
        min_gap_voxels = max(1, int(round(JOINT_GAP_MM / voxel_size)))
        print(f"\n  [STEP 5] Enforcing joint gap ({JOINT_GAP_MM}mm)...")
        femur_mask, tibia_mask = enforce_joint_gap(femur_mask, tibia_mask, min_gap_voxels)
        _save_debug_mask(femur_mask, femur_affine, "femur_after_gap", volume_name)

    # Thin Cortex Fill (Final Pass)
    print("\n  [STEP 6] Closing thin cortex gaps...")
    femur_mask = _fill_thin_cortex_gaps(femur_mask, femur_affine, 1.5)
    tibia_mask = _fill_thin_cortex_gaps(tibia_mask, tibia_affine, 1.5)
    _save_debug_mask(femur_mask, femur_affine, "femur_final_mask", volume_name)

    # ── 5. Mesh Extraction ──────────────────────────────────────────────────
    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)
    closing_mm_map = {"femur": CLOSING_MM_FEMUR, "tibia": CLOSING_MM_TIBIA}

    for bone_name, mask, affine in [
        ("femur", femur_mask, femur_affine),
        ("tibia", tibia_mask, tibia_affine),
    ]:
        if mask is None: continue
        print(f"\nProcessing {bone_name.upper()} ({int(np.sum(mask)):,} voxels)...")
        mesh = extract_mesh(mask, affine, raw_data=raw_data, has_metal=has_metal,
                            closing_mm=closing_mm_map[bone_name], bone_name=bone_name)

        if mesh is not None:
            out_path = output_dir / f"{volume_name}_{bone_name}_full.stl"
            mesh.export(str(out_path))
            print(f"  [OK] {out_path.name}  ({len(mesh.faces):,} faces)")

    # ── 6. Final Outputs ────────────────────────────────────────────────────
    _verify_gap(volume_name, output_dir)
    _save_final_segmentation(volume_name, femur_mask, tibia_mask, femur_affine)


def _save_final_segmentation(volume_name, femur_mask, tibia_mask, affine):
    """Save the processed, clean masks as a single multi-label NIfTI file."""
    output_dir = DATA / "segmentations" / "final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined volume: Femur=1, Tibia=2
    final_data = np.zeros(femur_mask.shape, dtype=np.uint8)
    if femur_mask is not None:
        final_data[femur_mask > 0] = 1
    if tibia_mask is not None:
        final_data[tibia_mask > 0] = 2
        
    out_path = output_dir / f"{volume_name}_final.nii.gz"
    print(f"\n  Exporting clinical-grade segmentation mask...")
    img = nib.Nifti1Image(final_data, affine)
    nib.save(img, str(out_path))
    print(f"  [OK] {out_path.name}")


def _export_bone(volume_name, bone_name, mask, affine, has_metal, closing_mm=3.0):
    """Extract and export a single bone mesh."""
    raw_data = None
    if has_metal:
        raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
        if raw_path.exists():
            raw_data = nib.load(str(raw_path)).get_fdata()

    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {bone_name.upper()} ({int(np.sum(mask)):,} voxels)...")
    mesh = extract_mesh(mask, affine, raw_data=raw_data, has_metal=has_metal,
                        closing_mm=closing_mm)
    if mesh is not None:
        out_path = output_dir / f"{volume_name}_{bone_name}_full.stl"
        mesh.export(str(out_path))
        print(f"  [OK] {out_path.name}  ({len(mesh.faces):,} faces)")
    else:
        print(f"  [WARN] No mesh produced for {bone_name}.")


def _verify_gap(volume_name, mesh_dir):
    """Quick post-processing check: measure surface-to-surface gap between femur and tibia."""
    femur_path = mesh_dir / f"{volume_name}_femur_full.stl"
    tibia_path = mesh_dir / f"{volume_name}_tibia_full.stl"

    if not femur_path.exists() or not tibia_path.exists():
        print("\n  [GAP CHECK] Skipped — both meshes required.")
        return

    try:
        femur_mesh = trimesh.load(str(femur_path))
        tibia_mesh = trimesh.load(str(tibia_path))

        # Sample points on tibia surface, query closest point on femur
        tibia_samples = tibia_mesh.sample(5000)
        _, dist, _ = trimesh.proximity.closest_point(femur_mesh, tibia_samples)

        min_gap = dist.min()
        mean_gap = dist.mean()
        print(f"\n  [GAP CHECK] Min gap: {min_gap:.2f}mm | Mean gap: {mean_gap:.2f}mm")
        if min_gap < 1.5:
            print(f"  [GAP CHECK] WARNING: Min gap < 1.5mm — bones may appear fused!")
        else:
            print(f"  [GAP CHECK] PASS: Adequate joint space maintained.")

        # Log to CSV for consistency tracking
        csv_path = DATA / "gap_measurements.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a") as f:
            if write_header:
                f.write("patient,min_gap_mm,mean_gap_mm,status\n")
            status = "PASS" if min_gap >= 1.5 else "FAIL"
            f.write(f"{volume_name},{min_gap:.2f},{mean_gap:.2f},{status}\n")
        print(f"  [GAP CHECK] Logged to {csv_path}")

    except Exception as e:
        print(f"\n  [GAP CHECK] Error during gap measurement: {e}")



if __name__ == "__main__":
    import argparse
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001")
    parser.add_argument(
        "--has-metal", action="store_true",
        help="Enable HU-based hardware filtering (post-op patients with metal implants only)"
    )
    args = parser.parse_args()
    try:
        process_volume(args.name, has_metal=args.has_metal)
    except Exception as e:
        print("\nFATAL ERROR IN PIPELINE:")
        traceback.print_exc()
        exit(1)

