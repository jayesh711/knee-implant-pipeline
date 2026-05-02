import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import onnxruntime as ort
from pathlib import Path
from scipy.ndimage import generate_binary_structure, binary_closing as scipy_binary_closing

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import DATA, HU_BONE_MIN

# JPlanner Specific Constants
STAGE1_MODEL_PATH = BASE_DIR / "models" / "jplanner_tka" / "stage1" / "model.onnx"
STAGE2_MODEL_PATH = BASE_DIR / "models" / "jplanner_tka" / "stage2" / "model.onnx"
INPUT_SHAPE = (192, 112, 112)  # (Depth, Height, Width)
CLIP_RANGE = (-200, 1000)

def preprocess_image(itk_image, target_shape=INPUT_SHAPE, clip_range=CLIP_RANGE):
    """Clip, Normalize, and Resize image for model input."""
    # 1. Clip HU
    clamped = sitk.Clamp(itk_image, lowerBound=clip_range[0], upperBound=clip_range[1])
    
    # 2. Resize to Target Shape
    # Note: SITK uses (W, H, D) while target_shape is (D, H, W)
    original_size = clamped.GetSize()
    original_spacing = clamped.GetSpacing()
    
    # Target spacing to fit target shape
    target_spacing = [
        original_spacing[0] * (original_size[0] / target_shape[2]), # W
        original_spacing[1] * (original_size[1] / target_shape[1]), # H
        original_spacing[2] * (original_size[2] / target_shape[0])  # D
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((target_shape[2], target_shape[1], target_shape[0]))
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputDirection(clamped.GetDirection())
    resampler.SetOutputOrigin(clamped.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resized = resampler.Execute(clamped)
    
    # 3. Normalize to [0, 1]
    arr = sitk.GetArrayFromImage(resized).astype(np.float32)
    arr = (arr - clip_range[0]) / (clip_range[1] - clip_range[0])
    arr = np.clip(arr, 0, 1)
    
    # Add batch and channel dimensions: [1, 1, D, H, W]
    return arr[np.newaxis, np.newaxis, ...], resized

def run_inference(session, input_data):
    """Run ONNX inference and return argmax labels."""
    # Input name usually 'input.1' based on config.pbtxt
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    # outputs[0] shape: [1, 3, 192, 112, 112]
    # Channels: 0=BG, 1=Femur, 2=Tibia (Standard JPlanner)
    softmax_out = outputs[0][0]
    labels = np.argmax(softmax_out, axis=0).astype(np.uint8)
    return labels

def get_bounding_box(mask, padding_mm=25.0, spacing=(1.0, 1.0, 1.0)):
    """Calculate bounding box of mask in physical space. Z-axis is never cropped to prevent truncation."""
    coords = np.argwhere(mask > 0) # (z, y, x)
    if coords.size == 0:
        return None
    
    y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
    y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
    
    # Convert padding mm to voxel counts
    pad_y = int(np.ceil(padding_mm / spacing[1]))
    pad_x = int(np.ceil(padding_mm / spacing[0]))
    
    # Return (start_indices, sizes)
    # IMPORTANT: We use the FULL Z-AXIS (0 to depth) to avoid truncation
    start = [
        int(max(0, x_min - pad_x)),
        int(max(0, y_min - pad_y)),
        0 # Start at bottom of scan
    ]
    size = [
        int(min(mask.shape[2] - start[0], (x_max - x_min) + 2*pad_x)),
        int(min(mask.shape[1] - start[1], (y_max - y_min) + 2*pad_y)),
        int(mask.shape[0]) # Use full height
    ]
    
    return start, size

def segment_jplanner(volume_name):
    print(f"\n--- JPlanner-A Segmentation Engine: {volume_name} ---")
    input_path = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    output_path = DATA / "segmentations" / "phase1" / f"{volume_name}_jplanner.nii.gz"
    
    if not input_path.exists():
        print(f"  [ERROR] Input {input_path} not found.")
        return

    # 1. Load Image
    img = sitk.ReadImage(str(input_path))
    print(f"  Volume Loaded: {img.GetSize()} voxels @ {img.GetSpacing()} mm")

    # Hardware acceleration check
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"  Initialzing ONNX Runtime (Providers: {providers})...")

    # 2. Stage 1: Coarse Localization
    print("  [STEP 1] Stage 1: Global Localization...")
    try:
        sess1 = ort.InferenceSession(str(STAGE1_MODEL_PATH), providers=providers)
        input1, resized1 = preprocess_image(img)
        labels1 = run_inference(sess1, input1)
        
        # Upsample Stage 1 mask to original resolution to find ROI
        mask1_itk = sitk.GetImageFromArray(labels1)
        mask1_itk.CopyInformation(resized1)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mask1_full = resampler.Execute(mask1_itk)
    except Exception as e:
        print(f"  [CRITICAL] Stage 1 Inference failed: {e}")
        return

    # 3. ROI Extraction
    print("  [STEP 2] ROI Extraction...")
    mask1_arr = sitk.GetArrayFromImage(mask1_full)
    bbox = get_bounding_box(mask1_arr, padding_mm=25.0, spacing=img.GetSpacing())
    
    if bbox is None:
        print("  [ERROR] Stage 1 failed to localize bone. Volume may be empty or non-CT.")
        return
    
    start, size = bbox
    roi_img = sitk.RegionOfInterest(img, size, start)
    print(f"    ROI Bbox: Start={start}, Size={size}")

    # 4. Stage 2: Refined Segmentation
    print("  [STEP 3] Stage 2: ROI-Based Refinement...")
    try:
        sess2 = ort.InferenceSession(str(STAGE2_MODEL_PATH), providers=providers)
        input2, resized2 = preprocess_image(roi_img)
        
        # Run inference and get softmax probabilities
        input_name = sess2.get_inputs()[0].name
        outputs = sess2.run(None, {input_name: input2})
        softmax_out = outputs[0][0] # [3, 192, 112, 112]
        
        # Upsample Softmax to ROI Resolution (Smooth transitions)
        print("    Upsampling softmax probabilities for smooth boundaries...")
        roi_size = roi_img.GetSize()
        full_softmax = []
        for i in range(softmax_out.shape[0]):
            prob_itk = sitk.GetImageFromArray(softmax_out[i])
            prob_itk.CopyInformation(resized2)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(roi_size)
            resampler.SetOutputSpacing(roi_img.GetSpacing())
            resampler.SetOutputOrigin(roi_img.GetOrigin())
            resampler.SetOutputDirection(roi_img.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            full_softmax.append(sitk.GetArrayFromImage(resampler.Execute(prob_itk)))
            
        full_softmax = np.stack(full_softmax, axis=0)
        labels_roi = np.argmax(full_softmax, axis=0).astype(np.uint8)
        
        # 5. Global Reconstruction (Ensemble Stage 1 + Stage 2)
        print("  [STEP 4] Global Reconstruction (Ensemble)...")
        # Start with Stage 1 (Full bone, coarse)
        final_mask_arr = sitk.GetArrayFromImage(mask1_full)
        
        z_start, y_start, x_start = start[2], start[1], start[0]
        z_size, y_size, x_size = labels_roi.shape
        
        # --- CRITICAL FIX: MAXIMUM ENSEMBLE ---
        # Instead of overwriting (which erases the shaft), we use maximum.
        # This keeps the Stage 1 shaft while adding the Stage 2 knee detail.
        roi_target = final_mask_arr[z_start:z_start+z_size, y_start:y_start+y_size, x_start:x_start+x_size]
        final_mask_arr[z_start:z_start+z_size, y_start:y_start+y_size, x_start:x_start+x_size] = np.maximum(roi_target, labels_roi)
        
        # 6. Anatomical Bridging (Connect shaft fragments)
        print("  [STEP 5] Anatomical Bridging (Connecting fragments)...")
        # Perform a vertical-heavy closing to bridge gaps in the shaft
        for label in [1, 2]: # Femur, Tibia
            bone_mask = (final_mask_arr == label).astype(np.uint8)
            if not np.any(bone_mask): continue
            
            # Bridge up to 30mm gaps in the shaft
            struct = generate_binary_structure(3, 1)
            # Expand more in Z
            bone_mask = scipy_binary_closing(bone_mask, iterations=3, structure=struct)
            final_mask_arr[bone_mask > 0] = label

        final_itk = sitk.GetImageFromArray(final_mask_arr)
        final_itk.CopyInformation(img)
        
        # Save output
        os.makedirs(output_path.parent, exist_ok=True)
        sitk.WriteImage(final_itk, str(output_path))
        
        # Print stats
        counts = np.bincount(final_mask_arr.flat)
        femur_v = counts[1] if len(counts) > 1 else 0
        tibia_v = counts[2] if len(counts) > 2 else 0
        print(f"    Voxel Counts: Femur={femur_v:,}, Tibia={tibia_v:,}")
        print(f"  [SUCCESS] JPlanner-A segmentation saved to: {output_path.name}")
        
    except Exception as e:
        print(f"  [ERROR] Segmentation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JPlanner-A TKA segmentation pipeline")
    parser.add_argument("--name", type=str, default="AB_72Y_Male_Left", help="Volume name")
    args = parser.parse_args()
    
    segment_jplanner(args.name)
