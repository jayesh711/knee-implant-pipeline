"""One-off diagnostic: analyzes tibia segmentation component gap size."""
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.morphology import closing as sk_closing, ball

seg = nib.load("data/segmentations/phase1/AB_72Y_Male_Left_Knee.nii")
seg_data = seg.get_fdata().astype(np.int32)
tibia_mask = (seg_data == 2).astype(np.uint8)
zooms = seg.header.get_zooms()
print(f"Voxel size: {zooms}, tibia voxels: {tibia_mask.sum():,}")

def analyze_components(mask, label):
    lab, n = ndimage.label(mask)
    if n == 0:
        print(f"{label}: no components")
        return
    counts = np.array([np.sum(lab == i) for i in range(1, n + 1)])
    print(f"{label}: {n} components, sizes: {sorted(counts, reverse=True)[:5]}")
    sorted_ids = np.argsort(counts)[::-1]
    for i in range(min(3, n)):
        cid = sorted_ids[i] + 1
        coords = np.where(lab == cid)
        print(f"  Comp {i+1} ({counts[sorted_ids[i]]:,}v): "
              f"X={coords[0].min()}-{coords[0].max()}, "
              f"Y={coords[1].min()}-{coords[1].max()}, "
              f"Z={coords[2].min()}-{coords[2].max()}")

analyze_components(tibia_mask, "RAW")

print("\nApplying ball(7) closing...")
closed7 = sk_closing(tibia_mask, ball(7)).astype(np.uint8)
analyze_components(closed7, "ball(7)")

print("\nApplying ball(15) closing...")
closed15 = sk_closing(tibia_mask, ball(15)).astype(np.uint8)
analyze_components(closed15, "ball(15)")
