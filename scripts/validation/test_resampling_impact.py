"""
Phase 1 Validation: Test whether 0.5mm resampling hurts TotalSegmentator accuracy.

Compares TotalSegmentator output on:
  1. Current _raw.nii.gz (already resampled to 0.5mm isotropic)
  2. A freshly saved original-resolution NIfTI (only HU-windowed + LPS-reoriented)

If Dice > 0.95 between the two segmentations, resampling is NOT significantly
hurting segmentation quality, and Phase 1 ingest restructure can be skipped.

Usage:
    python scripts/validation/test_resampling_impact.py <dicom_path> --name <patient_name>
"""

import sys
import os
import argparse
import subprocess
import shutil
import tempfile
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATA, DEFAULT_ORIENTATION, HU_WINDOW_LOW, HU_WINDOW_HIGH


def save_original_resolution_nifti(dicom_dir, output_path):
    """
    Save a NIfTI at the DICOM's original resolution.
    Only applies HU windowing and LPS reorientation -- NO resampling.
    """
    reader = sitk.ImageSeriesReader()
    
    # Find the best series
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        # Recursive search
        all_series = []
        for root, dirs, files in os.walk(dicom_dir):
            s_ids = reader.GetGDCMSeriesIDs(root)
            for s_id in s_ids:
                s_files = reader.GetGDCMSeriesFileNames(root, s_id)
                all_series.append((s_id, len(s_files), s_files))
        
        if not all_series:
            print(f"Error: No DICOM series found in {dicom_dir}")
            return None
        all_series.sort(key=lambda x: x[1], reverse=True)
        dicom_names = all_series[0][2]
    else:
        series_counts = []
        for s_id in series_ids:
            s_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), s_id)
            series_counts.append((s_id, len(s_files), s_files))
        series_counts.sort(key=lambda x: x[1], reverse=True)
        dicom_names = series_counts[0][2]
    
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    
    # Reorient to LPS
    img = sitk.DICOMOrient(img, DEFAULT_ORIENTATION)
    
    # HU Windowing only
    img = sitk.Clamp(img, lowerBound=HU_WINDOW_LOW, upperBound=HU_WINDOW_HIGH)
    
    print(f"  Original spacing: {img.GetSpacing()}")
    print(f"  Original size:    {img.GetSize()}")
    
    sitk.WriteImage(img, str(output_path))
    print(f"  Saved original-resolution NIfTI: {output_path}")
    return output_path


def run_totalseg_on(input_path, output_path):
    """Run TotalSegmentator on a NIfTI file."""
    from scripts.phase1 import _find_totalseg_exe  # noqa -- won't work due to 02_ prefix
    
    # Find TotalSegmentator executable manually
    totalseg_exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
    if not totalseg_exe:
        scripts_dir = Path(sys.executable).parent / "Scripts"
        for name in ["TotalSegmentator.exe", "totalsegmentator.exe"]:
            candidate = scripts_dir / name
            if candidate.exists():
                totalseg_exe = str(candidate)
                break
    
    if not totalseg_exe:
        print("Error: TotalSegmentator executable not found!")
        return False
    
    print(f"  Running TotalSegmentator on {input_path.name}...")
    try:
        subprocess.run([
            totalseg_exe,
            "-i", str(input_path),
            "-o", str(output_path),
            "--ml",
            "--task", "total",
            "--quiet"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  TotalSegmentator failed: {e}")
        return False


def compute_dice(seg1_data, seg2_data, label_id):
    """Compute Dice coefficient for a specific label."""
    mask1 = (seg1_data == label_id)
    mask2 = (seg2_data == label_id)
    
    intersection = np.sum(mask1 & mask2)
    vol1 = np.sum(mask1)
    vol2 = np.sum(mask2)
    
    if vol1 + vol2 == 0:
        return 1.0  # Both empty = perfect agreement
    
    return 2.0 * intersection / (vol1 + vol2)


def main():
    parser = argparse.ArgumentParser(description="Test if 0.5mm resampling hurts TotalSegmentator")
    parser.add_argument("dicom_path", type=str, help="Path to DICOM directory")
    parser.add_argument("--name", type=str, required=True, help="Patient name")
    args = parser.parse_args()
    
    dicom_path = Path(args.dicom_path)
    name = args.name
    
    print(f"\n{'='*60}")
    print(f" PHASE 1 VALIDATION: Resampling Impact Test")
    print(f" Patient: {name}")
    print(f"{'='*60}")
    
    # 1. The existing _raw.nii.gz (already resampled to 0.5mm)
    resampled_path = DATA / "NIfTI" / f"{name}_raw.nii.gz"
    if not resampled_path.exists():
        print(f"Error: Resampled volume not found: {resampled_path}")
        print(f"Run the pipeline first: python -m scripts.run_patient {dicom_path} --name {name}")
        return
    
    # 2. Save original-resolution NIfTI
    print(f"\n--- Step 1: Saving original-resolution NIfTI ---")
    orig_path = DATA / "NIfTI" / f"{name}_original_test.nii.gz"
    if not save_original_resolution_nifti(dicom_path, orig_path):
        return
    
    # 3. Run TotalSegmentator on original-resolution
    print(f"\n--- Step 2: Segmenting original-resolution volume ---")
    seg_orig_path = DATA / "segmentations" / "phase1" / f"{name}_origtest_total.nii.gz"
    if not run_totalseg_on(orig_path, seg_orig_path):
        return
    
    # 4. Load existing segmentation (from resampled input)
    seg_resampled_path = DATA / "segmentations" / "phase1" / f"{name}_total.nii.gz"
    if not seg_resampled_path.exists():
        print(f"Error: Resampled segmentation not found: {seg_resampled_path}")
        return
    
    # 5. Compare via Dice
    print(f"\n--- Step 3: Computing Dice Scores ---")
    
    seg_resamp = nib.load(str(seg_resampled_path))
    seg_orig = nib.load(str(seg_orig_path))
    
    data_resamp = seg_resamp.get_fdata()
    data_orig = seg_orig.get_fdata()
    
    # If shapes differ (they will), resample one to match the other using
    # nearest-neighbor interpolation on the label volume
    if data_resamp.shape != data_orig.shape:
        print(f"  Shapes differ: resampled={data_resamp.shape}, original={data_orig.shape}")
        print(f"  Resampling original seg to resampled grid (nearest-neighbor)...")
        
        # Use SimpleITK for label resampling -- MUST be nearest neighbor
        seg_orig_sitk = sitk.ReadImage(str(seg_orig_path))
        seg_resamp_sitk = sitk.ReadImage(str(seg_resampled_path))
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(seg_resamp_sitk.GetSpacing())
        resample.SetSize(seg_resamp_sitk.GetSize())
        resample.SetOutputDirection(seg_resamp_sitk.GetDirection())
        resample.SetOutputOrigin(seg_resamp_sitk.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)
        # NON-NEGOTIABLE: nearest-neighbor for label data
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        
        seg_orig_resampled = resample.Execute(seg_orig_sitk)
        data_orig = sitk.GetArrayFromImage(seg_orig_resampled)
        data_resamp = sitk.GetArrayFromImage(seg_resamp_sitk)
    
    # Femur labels (total task)
    femur_labels = [75, 76]
    # Find which femur label is present
    present_labels = set(np.unique(data_resamp[data_resamp > 0]).astype(int))
    
    print(f"\n  Labels present in resampled seg: {sorted(present_labels)}")
    
    dice_results = {}
    for label_id in sorted(present_labels):
        dice = compute_dice(data_resamp, data_orig, label_id)
        dice_results[label_id] = dice
        label_name = {75: "femur_left", 76: "femur_right"}.get(label_id, f"label_{label_id}")
        print(f"    Label {label_id:3d} ({label_name:15s}): Dice = {dice:.4f}")
    
    # Overall assessment
    bone_labels = [l for l in [75, 76] if l in present_labels]
    if bone_labels:
        avg_bone_dice = np.mean([dice_results[l] for l in bone_labels])
        print(f"\n  Average Bone Dice: {avg_bone_dice:.4f}")
        
        if avg_bone_dice > 0.95:
            print(f"\n  VERDICT: Dice > 0.95 -- Resampling does NOT significantly hurt segmentation.")
            print(f"  RECOMMENDATION: Phase 1 ingest restructure is LOW PRIORITY. Skip it.")
        else:
            print(f"\n  VERDICT: Dice <= 0.95 -- Resampling IS hurting segmentation quality.")
            print(f"  RECOMMENDATION: Proceed with Phase 1 to save original-resolution NIfTI.")
    
    # Cleanup test files
    print(f"\n--- Cleanup ---")
    for p in [orig_path, seg_orig_path]:
        if p.exists():
            os.remove(p)
            print(f"  Removed: {p.name}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
