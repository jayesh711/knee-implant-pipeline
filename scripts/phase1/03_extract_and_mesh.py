import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
import json
from skimage import measure
from skimage.morphology import closing as sk_closing, ball
from pathlib import Path
import os
from config import BASE_DIR, DATA, MAX_TRIANGLES, SMOOTH_ITERS, HU_METAL_MIN, TAUBIN_LAMBDA, TAUBIN_MU, MAX_HOLE_DIAMETER_MM

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


def extract_mesh(mask, affine, raw_data=None, has_metal=False):
    """
    Generate a clean, full-length bone mesh from a binary voxel mask.

    mask:      uint8 binary array in segmentation voxel space.
    affine:    4x4 voxel-to-LPS-mm transform from the segmentation NIfTI.
    raw_data:  HU array (same shape) for optional metal filtering.
    has_metal: enable surgical hardware filter (post-op cases only, opt-in).
    """
    mask = mask.copy()

    if has_metal and raw_data is not None:
        print(f"    Filtering hardware (>{_METAL_THRESHOLD} HU)...")
        mask[raw_data > _METAL_THRESHOLD] = 0

    if not np.any(mask):
        return None

    # Pre-close: bridge narrow shaft gaps and label dropout (ball=3 ≈ 1.5mm at 0.5mm)
    print(f"    Bridging intra-bone gaps (ball radius=3)...")
    mask = sk_closing(mask, ball(3)).astype(np.uint8)

    # Multi-component retention: keep all fragments >= 3% of the largest component.
    # Prevents silently discarding real bone when segmentation has a gap.
    comp_arr = measure.label(mask)
    n_comps = comp_arr.max()
    if n_comps > 1:
        counts = np.bincount(comp_arr.flat)[1:]
        largest = counts.max()
        kept_ids = np.where(counts >= 0.03 * largest)[0] + 1
        print(f"    Retained {len(kept_ids)}/{n_comps} components (>= 3% of {largest:,} voxels)")
        mask = np.isin(comp_arr, kept_ids).astype(np.uint8)

    # Internal void closing (marrow spaces)
    mask = sk_closing(mask, ball(1)).astype(np.uint8)

    if not np.any(mask):
        return None

    # Marching Cubes
    print(f"    Marching Cubes...")
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)

    # Voxel → LPS mm
    world_verts = (affine[:3, :3] @ verts.T).T + affine[:3, 3]

    # PyMeshLab clinical refinement
    print(f"    PyMeshLab refinement...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(world_verts, faces))

    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")

    try:
        voxel_size = np.cbrt(np.abs(np.linalg.det(affine[:3, :3])))
        max_edges = int(MAX_HOLE_DIAMETER_MM / voxel_size)
        ms.apply_filter("meshing_close_holes", maxholesize=max_edges)
    except Exception as e:
        print(f"    Warning: Hole filling skipped ({e})")

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
        return img.get_fdata().astype(np.int32), img.affine

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
        _export_bone(volume_name, bone_name, available_mask, available_affine, has_metal)
        return

    # ── Both bones found — export independently ───────────────────────────────
    raw_data = None
    if has_metal:
        raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
        if raw_path.exists():
            print(f"  Loading raw CT for metal filtering...")
            raw_data = nib.load(str(raw_path)).get_fdata()

    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)

    for bone_name, mask, affine in [
        ("femur", femur_mask, femur_affine),
        ("tibia", tibia_mask, tibia_affine),
    ]:
        print(f"\nProcessing {bone_name.upper()} ({int(np.sum(mask)):,} voxels)...")
        mesh = extract_mesh(mask, affine, raw_data=raw_data, has_metal=has_metal)
        if mesh is not None:
            out_path = output_dir / f"{volume_name}_{bone_name}_full.stl"
            mesh.export(str(out_path))
            print(f"  [OK] {out_path.name}  ({len(mesh.faces):,} faces)")
        else:
            print(f"  [WARN] No mesh produced for {bone_name}.")


def _export_bone(volume_name, bone_name, mask, affine, has_metal):
    """Extract and export a single bone mesh."""
    raw_data = None
    if has_metal:
        raw_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
        if raw_path.exists():
            raw_data = nib.load(str(raw_path)).get_fdata()

    output_dir = DATA / "meshes"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {bone_name.upper()} ({int(np.sum(mask)):,} voxels)...")
    mesh = extract_mesh(mask, affine, raw_data=raw_data, has_metal=has_metal)
    if mesh is not None:
        out_path = output_dir / f"{volume_name}_{bone_name}_full.stl"
        mesh.export(str(out_path))
        print(f"  [OK] {out_path.name}  ({len(mesh.faces):,} faces)")
    else:
        print(f"  [WARN] No mesh produced for {bone_name}.")


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
