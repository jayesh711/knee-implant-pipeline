
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_metal_slices(volume_name):
    raw_vol_path = Path("data/NIfTI") / f"{volume_name}_raw.nii.gz"
    if not raw_vol_path.exists():
        raw_vol_path = Path("data/NIfTI") / f"{volume_name}.nii.gz"
    
    img = nib.load(str(raw_vol_path))
    data = img.get_fdata()
    
    # Metal threshold
    metal_mask = (data > 2000)
    
    # Find Z-range of metal
    z_indices = np.argwhere(np.any(metal_mask, axis=(0,1))).flatten()
    if len(z_indices) == 0:
        print("No metal found above 2000 HU.")
        return
    
    z_min, z_max = z_indices[0], z_indices[-1]
    z_mid = (z_min + z_max) // 2
    
    # Slices to check: top of metal, middle, bottom
    slices = [z_min + 10, z_mid, z_max - 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, z in enumerate(slices):
        if z < 0 or z >= data.shape[2]: continue
        
        # Show a crop around the center of the image
        center_x, center_y = data.shape[0]//2, data.shape[1]//2
        crop = 200
        slice_data = data[center_x-crop:center_x+crop, center_y-crop:center_y+crop, z]
        
        axes[i].imshow(slice_data, cmap='gray', vmin=-100, vmax=1000)
        axes[i].set_title(f"Z={z} (HU Range -100 to 1000)")
        
        # Overlay metal in red
        metal_slice = metal_mask[center_x-crop:center_x+crop, center_y-crop:center_y+crop, z]
        axes[i].imshow(np.ma.masked_where(~metal_slice, metal_slice), cmap='Reds', alpha=0.5)
        
    plt.tight_layout()
    output_path = Path("scratch") / f"{volume_name}_metal_check.png"
    plt.savefig(str(output_path))
    print(f"Metal check image saved to {output_path}")

if __name__ == "__main__":
    generate_metal_slices("S0001")
