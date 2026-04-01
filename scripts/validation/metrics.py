import numpy as np
import nibabel as nib
import torch
import argparse
from pathlib import Path
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import EnsureType

def compute_metrics(gt_path, pred_path, label_id=1):
    """Compute DSC, ASD, and HD95 using MONAI metrics."""
    print(f"Loading GT: {gt_path}")
    print(f"Loading Pred: {pred_path}")
    
    gt_img = nib.load(str(gt_path))
    pred_img = nib.load(str(pred_path))
    
    gt_data = gt_img.get_fdata()
    pred_data = pred_img.get_fdata()
    
    # Binarize for the specific label
    gt_mask = (gt_data == label_id).astype(np.float32)
    pred_mask = (pred_data == label_id).astype(np.float32)
    
    # Add batch and channel dimensions for MONAI [B, C, H, W, D]
    gt_tensor = torch.from_numpy(gt_mask)[None, None, ...]
    pred_tensor = torch.from_numpy(pred_mask)[None, None, ...]
    
    # 1. Dice Similarity Coefficient
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=pred_tensor, y=gt_tensor)
    dice_score = dice_metric.aggregate().item()
    
    # 2. Hausdorff Distance (95th percentile)
    hd95_metric = HausdorffDistanceMetric(percentile=95, reduction="mean")
    hd95_metric(y_pred=pred_tensor, y=gt_tensor)
    hd95_score = hd95_metric.aggregate().item()
    
    # 3. Average Surface Distance
    asd_metric = SurfaceDistanceMetric(symmetric=True, reduction="mean")
    asd_metric(y_pred=pred_tensor, y=gt_tensor)
    asd_score = asd_metric.aggregate().item()
    
    return {
        "Dice": dice_score,
        "HD95": hd95_score,
        "ASD": asd_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute clinical segmentation metrics")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth NIfTI")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted NIfTI")
    parser.add_argument("--label", type=int, default=1, help="Label ID to evaluate")
    
    args = parser.parse_args()
    
    results = compute_metrics(args.gt, args.pred, args.label)
    
    print("\n" + "="*30)
    print(" CLINICAL VALIDATION RESULTS")
    print("="*30)
    print(f"Dice Score: {results['Dice']:.4f} (Target: >0.98)")
    print(f"HD95:       {results['HD95']:.4f} mm (Target: <1.0mm)")
    print(f"ASD:        {results['ASD']:.4f} mm (Target: <0.5mm)")
    print("="*30)
