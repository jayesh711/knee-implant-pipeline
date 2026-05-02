import os
import sys
import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from config import (
    DATA, 
    DEFAULT_SPACING, 
    DEFAULT_ORIENTATION, 
    HU_WINDOW_LOW, 
    HU_WINDOW_HIGH, 
    HU_BIN_WIDTH, 
    NORM_METHOD
)

def resample_image(itk_image, out_spacing=DEFAULT_SPACING, is_label=False):
    """Resample image to isotropic spacing."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_size = [
        int(round(original_size[i] * (original_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def apply_clinical_normalization(image, method=NORM_METHOD):
    """Apply global intensity normalization."""
    if method == "none":
        return image
    
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    
    if method == "zscore":
        mean = np.mean(arr)
        std = np.std(arr)
        arr = (arr - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = np.min(arr)
        max_val = np.max(arr)
        arr = (arr - min_val) / (max_val - min_val + 1e-8)
        
    normalized_image = sitk.GetImageFromArray(arr)
    normalized_image.CopyInformation(image)
    return normalized_image

def apply_intensity_discretization(image, bin_width=HU_BIN_WIDTH):
    """Apply intensity discretization (binning) to reduce noise."""
    if bin_width <= 0:
        return image
    
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    # Binning in-place to avoid peak memory from temporary arrays
    arr /= bin_width
    np.floor(arr, out=arr)
    arr *= bin_width
    
    discretized_image = sitk.GetImageFromArray(arr)
    discretized_image.CopyInformation(image)
    return discretized_image

def ingest_series(dicom_dir, volume_name):
    print(f"--- Ingesting: {dicom_dir} (Name: {volume_name}) ---")
    
    # 1. Load DICOM
    reader = sitk.ImageSeriesReader()
    
    # Get all series IDs in the directory
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    
    if not series_ids:
        # Recursive search for nested structures
        print("  Direct folder contains no series. Searching subdirectories...")
        all_series = []
        for root, dirs, files in os.walk(dicom_dir):
            s_ids = reader.GetGDCMSeriesIDs(root)
            for s_id in s_ids:
                s_files = reader.GetGDCMSeriesFileNames(root, s_id)
                all_series.append((s_id, len(s_files), s_files, root))
        
        if all_series:
            # Pick the largest series (to avoid localizers/scouts)
            all_series.sort(key=lambda x: x[1], reverse=True)
            best_id, best_count, best_names, best_dir = all_series[0]
            print(f"  Found largest series '{best_id}' in: {best_dir} ({best_count} slices)")
            dicom_names = best_names
        else:
            print(f"Error: No DICOM series found in {dicom_dir} or its subdirectories.")
            return
    else:
        # Multiple series in the direct folder - pick the one with most files
        series_counts = []
        for s_id in series_ids:
            s_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), s_id)
            series_counts.append((s_id, len(s_files), s_files))
        
        series_counts.sort(key=lambda x: x[1], reverse=True)
        best_id, best_count, dicom_names = series_counts[0]
        print(f"  Selected largest series '{best_id}' from {len(series_ids)} candidates ({best_count} slices)")
    
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    
    # 2. Reorient to standard (LPS)
    print(f"Original Orientation: {sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())}")
    img = sitk.DICOMOrient(img, DEFAULT_ORIENTATION)
    
    # 3. HU Windowing (Clipping)
    print(f"Applying HU Windowing: [{HU_WINDOW_LOW}, {HU_WINDOW_HIGH}]")
    img = sitk.Clamp(img, lowerBound=HU_WINDOW_LOW, upperBound=HU_WINDOW_HIGH)
    
    # 4. Resampling to Isotropic (B-spline) OR Native Resolution
    print(f"Original Spacing: {img.GetSpacing()}")
    if not getattr(args, 'pure', False):
        print(f"Target Spacing:   {DEFAULT_SPACING} (B-spline)")
        resampled_image = resample_image(img, out_spacing=DEFAULT_SPACING)
        
        # 5. Clinical Denoising: Anisotropic Diffusion
        print("Applying Gradient Anisotropic Diffusion denoising...")
        resampled_image = sitk.Cast(resampled_image, sitk.sitkFloat32)
        denoiser = sitk.GradientAnisotropicDiffusionImageFilter()
        denoiser.SetNumberOfIterations(5)
        denoiser.SetTimeStep(0.03)
        denoiser.SetConductanceParameter(3.0)
        denoised_image = denoiser.Execute(resampled_image)
        
        # 6. Intensity Discretization
        print(f"Applying Intensity Discretization (Bin Width: {HU_BIN_WIDTH} HU)...")
        discretized_image = apply_intensity_discretization(denoised_image, bin_width=HU_BIN_WIDTH)
    else:
        print("  [PURE] Skipping resampling, denoising, and discretization for high-fidelity native resolution...")
        resampled_image = img
        discretized_image = resampled_image
    
    # 7. Global Normalization
    print(f"Applying Global Normalization ({NORM_METHOD})...")
    normalized_image = apply_clinical_normalization(discretized_image, method=NORM_METHOD)
    
    # Save Preprocessed volume
    OUTPUT_DIR = DATA / "NIfTI"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Raw resampled (but windowed) - for traditional AI
    raw_path = OUTPUT_DIR / f"{volume_name}_raw.nii.gz"
    sitk.WriteImage(resampled_image, str(raw_path))
    print(f"Saved RAW resampled volume to: {raw_path}")
    
    # Prepped (Denoised, Discretized, Normalized) - for clinical grade pipeline
    prepped_path = OUTPUT_DIR / f"{volume_name}_prepped.nii.gz"
    sitk.WriteImage(normalized_image, str(prepped_path))
    print(f"Saved PREPPED refined volume to: {prepped_path}")
    
    print(f"Final Size: {normalized_image.GetSize()}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DICOM series to NIfTI with clinical preprocessing")
    parser.add_argument("input_dir", type=str, help="Path to DICOM directory")
    parser.add_argument("--name", type=str, default=None, help="Output filename (optional)")
    parser.add_argument("--pure", action="store_true", help="Skip clinical prep (denoising/discretization)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Path {input_path} does not exist.")
        sys.exit(1)
        
    output_name = args.name if args.name else input_path.name
    
    ingest_series(input_path, output_name)
