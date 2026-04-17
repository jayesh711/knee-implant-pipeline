"""
Diagnostic: Inspect AB_72_Y_Male-Left_Knee segmentation labels
Run from project root: python scratch/diagnose_left_knee_seg.py
"""
import nibabel as nib
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PATIENT = "AB_72_Y_Male-Left_Knee"

seg_path = DATA / "segmentations" / "phase1" / f"{PATIENT}.nii"
raw_path = DATA / "NIfTI" / f"{PATIENT}_raw.nii.gz"

print(f"Loading segmentation: {seg_path}")
img = nib.load(str(seg_path))
data = img.get_fdata()
affine = img.affine

print(f"\nVolume shape : {data.shape}")
print(f"Affine diagonal  : {np.diag(affine[:3,:3]).round(3)}")
print(f"Z-direction sign : {np.sign(affine[2,2])}  (+1 = Z increases superiorly in LPS)")
print(f"Origin (world mm): {affine[:3,3].round(1)}")

zooms = img.header.get_zooms()
voxel_vol_cc = float(np.prod(zooms)) / 1000.0
print(f"Voxel volume : {voxel_vol_cc:.6f} cc\n")

labels, counts = np.unique(data, return_counts=True)
labels = labels.astype(int)

# Candidate lists used in the pipeline
femur_candidates = [13, 14, 76, 77, 75, 44, 43, 2, 1, 25, 24]
tibia_candidates = [2, 46, 45, 4, 3, 27, 26, 80, 81]

print(f"{'Label':<8} {'Voxels':<10} {'Vol(cc)':<10} {'Z-vox(mean)':<14} {'Z-world(mm)':<14} {'Tag'}")
print("-" * 72)
for lbl, cnt in zip(labels, counts):
    if lbl == 0:
        continue
    coords = np.argwhere(data == lbl)
    z_vox = coords[:, 2].mean()
    z_world = affine[2, 2] * z_vox + affine[2, 3]
    vol = cnt * voxel_vol_cc
    tag = []
    if lbl in femur_candidates:
        tag.append("FEMUR-candidate")
    if lbl in tibia_candidates:
        tag.append("TIBIA-candidate")
    print(f"{lbl:<8} {cnt:<10} {vol:<10.2f} {z_vox:<14.1f} {z_world:<14.1f} {', '.join(tag)}")

# --- Show what the pipeline actually picks ---
print("\n=== Pipeline label selection ===")
unique_labels = set(labels.tolist())

def get_best_label(candidates, data, unique_labels):
    valid = [c for c in candidates if c in unique_labels]
    if not valid:
        return None, 0
    best = max(valid, key=lambda c: np.sum(data == c))
    return best, int(np.sum(data == best))

actual_femur, fvox = get_best_label(femur_candidates, data, unique_labels)
actual_tibia, tvox = get_best_label(tibia_candidates, data, unique_labels)

print(f"  Femur label picked : {actual_femur}  ({fvox} voxels, {fvox*voxel_vol_cc:.1f} cc)")
print(f"  Tibia label picked : {actual_tibia}  ({tvox} voxels, {tvox*voxel_vol_cc:.1f} cc)")

if actual_femur == actual_tibia:
    print("  *** WARNING: Same label picked for both femur AND tibia! ***")

if actual_femur is not None and actual_tibia is not None:
    f_z = np.mean(np.argwhere(data == actual_femur)[:, 2])
    t_z = np.mean(np.argwhere(data == actual_tibia)[:, 2])
    z_dir = np.sign(affine[2, 2])
    is_anatomical = (t_z < f_z) if z_dir > 0 else (t_z > f_z)
    print(f"\n  Femur Z-centroid (vox): {f_z:.1f}")
    print(f"  Tibia Z-centroid (vox): {t_z:.1f}")
    print(f"  Z-direction: {'+' if z_dir > 0 else '-'}  (Z increases {'superiorly' if z_dir > 0 else 'inferiorly'})")
    print(f"  Anatomically correct order: {'YES' if is_anatomical else 'NO -- INVERTED!'}")

# --- Canal script label selection ---
print("\n=== Canal script label selection ===")
femur_canal_labels = [76, 75, 44, 25, 24, 94, 93]
tibia_canal_labels = [81, 80, 46, 45, 4, 3, 27, 26]

# Canal script samples at stride 5 — replicate that
sample = data[::5, ::5, ::5]
present = set(np.unique(sample).astype(int).tolist())

valid_f = [l for l in femur_canal_labels if l in present]
valid_t = [l for l in tibia_canal_labels if l in present]

canal_femur = max(valid_f, key=lambda l: np.sum(sample == l)) if valid_f else None
canal_tibia = max(valid_t, key=lambda l: np.sum(sample == l)) if valid_t else None

print(f"  Canal femur label : {canal_femur}  (valid candidates present: {valid_f})")
print(f"  Canal tibia label : {canal_tibia}  (valid candidates present: {valid_t})")

if canal_tibia is None:
    print("  *** Tibia label NOT FOUND in canal candidates — HU fallback will trigger ***")

# --- Compare with right knee ---
print("\n=== Comparison: Right Knee labels ===")
right_seg = DATA / "segmentations" / "phase1" / "AB_72_Y_Male-Right_Knee.nii"
if right_seg.exists():
    rimg = nib.load(str(right_seg))
    rdata = rimg.get_fdata()
    rlabels = set(np.unique(rdata).astype(int).tolist())
    rvalid_f = [c for c in femur_candidates if c in rlabels]
    rvalid_t = [c for c in tibia_candidates if c in rlabels]
    r_femur = max(rvalid_f, key=lambda c: np.sum(rdata == c)) if rvalid_f else None
    r_tibia = max(rvalid_t, key=lambda c: np.sum(rdata == c)) if rvalid_t else None
    rf_vox = int(np.sum(rdata == r_femur)) if r_femur else 0
    rt_vox = int(np.sum(rdata == r_tibia)) if r_tibia else 0
    print(f"  Right femur label: {r_femur}  ({rf_vox} voxels, {rf_vox*voxel_vol_cc:.1f} cc)")
    print(f"  Right tibia label: {r_tibia}  ({rt_vox} voxels, {rt_vox*voxel_vol_cc:.1f} cc)")
    print(f"\n  Labels in right but NOT left: {rlabels - unique_labels - {0}}")
    print(f"  Labels in left  but NOT right: {unique_labels - rlabels - {0}}")
else:
    print("  Right knee segmentation not found for comparison.")
