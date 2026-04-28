import nibabel as nib
import numpy as np
import trimesh
import trimesh.repair
import pymeshlab
import json
from skimage import measure
from skimage.morphology import ball
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing as scipy_binary_closing,
    distance_transform_edt,
)
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


# ── Joint Gap Enforcement ─────────────────────────────────────────────────────
def enforce_joint_gap(femur_mask, tibia_mask, min_gap_voxels=4):
    """
    Ensure a minimum physical gap between femur and tibia masks.
    Uses distance_transform_edt to compute the distance of every voxel
    of one bone from the boundary of the other. Voxels closer than
    min_gap_voxels are removed.

    At 0.5mm spacing, min_gap_voxels=4 → 2mm physical gap.
    At 0.5mm spacing, min_gap_voxels=6 → 3mm physical gap.
    """
    # Compute distance of every femur voxel from tibia boundary
    # and vice versa. Where distance < min_gap, erode that bone.
    femur_dist = distance_transform_edt(~tibia_mask.astype(bool))
    tibia_dist = distance_transform_edt(~femur_mask.astype(bool))

    # Remove femur voxels too close to tibia
    femur_mask = femur_mask & (femur_dist >= min_gap_voxels)
    # Remove tibia voxels too close to femur
    tibia_mask = tibia_mask & (tibia_dist >= min_gap_voxels)

    return femur_mask.astype(np.uint8), tibia_mask.astype(np.uint8)


def extract_mesh(mask, affine, raw_data=None, has_metal=False, closing_mm=3.0):
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
        print(f"    Cropping to region: {min_c} -> {max_c} (Shape: {max_c - min_c})")

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
    print(f"    Bridging intra-bone gaps (closing={closing_mm}mm → ball radius={closing_radius} voxels at {voxel_size:.2f}mm/vox)...")

    # scipy binary_closing uses boolean dtype (~8x less memory than skimage's float64 closing).
    mask = scipy_binary_closing(mask, structure=ball(closing_radius)).astype(np.uint8)

    # Multi-component retention: keep only components with volume >= 50,000 voxels
    # (~6.25 cm³ at 0.5mm spacing). Volume-based filter is more robust than
    # percentage-based — doesn't depend on the size of the largest component.
    MIN_COMPONENT_VOXELS = 50_000
    comp_arr = measure.label(mask)
    n_comps = comp_arr.max()
    if n_comps > 1:
        counts = np.bincount(comp_arr.flat)[1:]
        kept_ids = np.where(counts >= MIN_COMPONENT_VOXELS)[0] + 1
        if len(kept_ids) == 0:
            # Fallback: if no component reaches 50k, keep the largest
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
    # Step 1: Trimesh hole-fill first (handles topologically complex holes that
    # PyMeshLab fails silently on)
    print(f"    Trimesh hole-fill pass...")
    tri_mesh = trimesh.Trimesh(vertices=world_verts, faces=faces)
    trimesh.repair.fill_holes(tri_mesh)
    trimesh.repair.fix_normals(tri_mesh)

    # Step 2: PyMeshLab clinical refinement
    print(f"    PyMeshLab refinement...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(tri_mesh.vertices, tri_mesh.faces))

    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")

    # PyMeshLab hole-fill as second pass (catches remaining holes after trimesh)
    try:
        max_edges = int(MAX_HOLE_DIAMETER_MM / voxel_size)
        ms.apply_filter("meshing_close_holes", maxholesize=max_edges)
    except Exception as e:
        print(f"    Warning: PyMeshLab hole filling skipped ({e})")

    # Laplacian pre-smooth: one pass to kill the worst staircase artifacts
    # before Taubin preserves the anatomy
    print(f"    Laplacian pre-smooth (1 iteration)...")
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=1)

    print(f"    Taubin smoothing ({SMOOTH_ITERS} iterations)...")
    ms.apply_coord_taubin_smoothing(
        stepsmoothnum=SMOOTH_ITERS, lambda_=TAUBIN_LAMBDA, mu=TAUBIN_MU
    )

    current_faces = ms.current_mesh().face_number()
    if current_faces > MAX_TRIANGLES:
        print(f"    Decimating {current_faces:,} -> {MAX_TRIANGLES:,} faces...")
        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse", targetfacenum=MAX_TRIANGLES
        )

    final_m = ms.current_mesh()
    return trimesh.Trimesh(vertices=final_m.vertex_matrix(), faces=final_m.face_matrix())


def process_volume(volume_name="S0001", has_metal=False):
    """
    Dual-segmentation bone reconstruction strategy:

      1. appendicular_bones output → tibia (label 2, highly reliable)
      2. total-task output          → femur (labels 75/76, full shaft coverage)

    Falls back to SI-vector split if only one segmentation is available and
    contains both femur and tibia labels.

    After extraction, enforces a minimum joint gap between femur and tibia
    using distance_transform_edt to prevent anatomical fusion.
    """
    seg_paths = _find_segmentations(volume_name)

    if not seg_paths:
        print(f"Error: No segmentation found for '{volume_name}'. Run TotalSegmentator first.")
        return

    print(f"\n--- Dual-Segmentation Bone Reconstruction: {volume_name} ---")
    print(f"  Available: {list(seg_paths.keys())}")

    # ── Collect bone masks from each available segmentation ────────────────────
    femur_mask = None
    femur_affine = None
    tibia_mask = None
    tibia_affine = None

    def _load(path):
        img = nib.load(str(path))
        # float32 avoids the default float64 allocation; uint8 is sufficient for all
        # segmentation labels (TotalSegmentator v2 max label < 200) and 4x smaller than int32.
        return img.get_fdata(dtype=np.float32).astype(np.uint8), img.affine

    def _build_mask(seg_data, label_ids):
        m = np.zeros(seg_data.shape, dtype=np.uint8)
        for lbl in label_ids:
            m[seg_data == lbl] = 1
        return m

    # Primary segmentation (appendicular_bones) → tibia + possibly femur_auxiliary
    if "primary" in seg_paths:
        print(f"\n  Loading primary segmentation: {seg_paths['primary'].name}")
        seg_data, affine = _load(seg_paths["primary"])
        femur_ids, tibia_ids = _detect_bone_labels(seg_data)

        if tibia_ids:
            tibia_mask = _build_mask(seg_data, tibia_ids)
            tibia_affine = affine
            print(f"  Tibia mask: {int(np.sum(tibia_mask)):,} voxels from labels {tibia_ids}")

        if femur_ids and femur_mask is None:
            femur_mask = _build_mask(seg_data, femur_ids)
            femur_affine = affine
            print(f"  Femur mask (primary): {int(np.sum(femur_mask)):,} voxels from labels {femur_ids}")

    # Total-task segmentation → femur (labels 75/76, full shaft)
    if "total" in seg_paths and femur_mask is None:
        print(f"\n  Loading total-task segmentation: {seg_paths['total'].name}")
        seg_data_t, affine_t = _load(seg_paths["total"])
        femur_ids_t, _ = _detect_bone_labels(seg_data_t)

        if femur_ids_t:
            femur_mask = _build_mask(seg_data_t, femur_ids_t)
            femur_affine = affine_t
            print(f"  Femur mask (total): {int(np.sum(femur_mask)):,} voxels from labels {femur_ids_t}")
        else:
            print(f"  Warning: No femur labels found in total-task segmentation either.")

    # ── Fallback: if one segmentation has both, use SI-vector split ───────────
    if femur_mask is None and tibia_mask is None:
        print("Error: No bone masks found in any available segmentation.")
        return

    if femur_mask is None or tibia_mask is None:
        # Only one bone available — check if combined mask can be split by SI-Z
        available_mask = femur_mask if femur_mask is not None else tibia_mask
        available_affine = femur_affine if femur_affine is not None else tibia_affine
        bone_name = "femur" if femur_mask is not None else "tibia"
        print(f"\n  Only {bone_name} found — exporting single bone.")
        closing_mm = CLOSING_MM_FEMUR if bone_name == "femur" else CLOSING_MM_TIBIA
        _export_bone(volume_name, bone_name, available_mask, available_affine, has_metal, closing_mm)
        return

    # ── Both bones found — enforce joint gap, then export independently ───────
    # Joint gap enforcement BEFORE mesh extraction (operates on voxel masks).
    # Ensures femur and tibia don't fuse at the knee joint.
    voxel_size = _voxel_size_from_affine(femur_affine)
    min_gap_voxels = max(1, int(round(JOINT_GAP_MM / voxel_size)))
    print(f"\n  Enforcing joint gap: {JOINT_GAP_MM}mm → {min_gap_voxels} voxels at {voxel_size:.2f}mm/vox")

    # Both masks must be in the same coordinate space for gap enforcement.
    # They share the same grid when from the same segmentation file, or when
    # both segmentations were produced from the same _raw.nii.gz (same affine/shape).
    if np.array_equal(femur_affine, tibia_affine) and femur_mask.shape == tibia_mask.shape:
        femur_mask, tibia_mask = enforce_joint_gap(femur_mask, tibia_mask, min_gap_voxels)
        print(f"  Joint gap enforced. Femur: {int(np.sum(femur_mask)):,}  Tibia: {int(np.sum(tibia_mask)):,} voxels")
    else:
        print(f"  Warning: Masks from different grids — joint gap enforcement skipped.")
        print(f"    (femur affine == tibia affine: {np.array_equal(femur_affine, tibia_affine)}, "
              f"shapes: {femur_mask.shape} vs {tibia_mask.shape})")

    raw_data = None
    if has_metal:
        raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
        if raw_path.exists():
            print(f"  Loading raw CT for metal filtering...")
            raw_data = nib.load(str(raw_path)).get_fdata()

    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)

    # Adaptive closing radii in physical mm (converted to voxels inside extract_mesh)
    closing_mm_map = {"femur": CLOSING_MM_FEMUR, "tibia": CLOSING_MM_TIBIA}

    for bone_name, mask, affine in [
        ("femur", femur_mask, femur_affine),
        ("tibia", tibia_mask, tibia_affine),
    ]:
        print(f"\nProcessing {bone_name.upper()} ({int(np.sum(mask)):,} voxels)...")
        mesh = extract_mesh(mask, affine, raw_data=raw_data, has_metal=has_metal,
                            closing_mm=closing_mm_map[bone_name])
        if mesh is not None:
            out_path = output_dir / f"{volume_name}_{bone_name}_full.stl"
            mesh.export(str(out_path))
            print(f"  [OK] {out_path.name}  ({len(mesh.faces):,} faces)")
        else:
            print(f"  [WARN] No mesh produced for {bone_name}.")

    # ── Automatic gap verification ────────────────────────────────────────────
    _verify_gap(volume_name, output_dir)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="S0001")
    parser.add_argument(
        "--has-metal", action="store_true",
        help="Enable HU-based hardware filtering (post-op patients with metal implants only)"
    )
    args = parser.parse_args()
    process_volume(args.name, has_metal=args.has_metal)
