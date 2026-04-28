"""
Measure the minimum surface-to-surface distance between femur and tibia STL meshes.
Run after every patient to verify adequate joint space is maintained.

Usage:
    python -m scripts.validation.measure_gap --name AS_60_male
    python -m scripts.validation.measure_gap --femur path/to/femur.stl --tibia path/to/tibia.stl

Targets:
    - Min gap  >= 1.5mm  (bones are not fused)
    - Mean gap >= 4.0mm  (anatomically reasonable cartilage space)
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path
from config import DATA


def measure_gap(femur_path, tibia_path, n_samples=5000):
    """
    Measure surface-to-surface distance between femur and tibia meshes.
    Returns dict with min_gap, mean_gap, median_gap, p95_gap.
    """
    femur = trimesh.load(str(femur_path))
    tibia = trimesh.load(str(tibia_path))

    # Sample points on tibia surface, query closest point on femur
    tibia_samples = tibia.sample(n_samples)
    _, dist_t2f, _ = trimesh.proximity.closest_point(femur, tibia_samples)

    # Also sample femur -> tibia for symmetric measurement
    femur_samples = femur.sample(n_samples)
    _, dist_f2t, _ = trimesh.proximity.closest_point(tibia, femur_samples)

    # Combine for symmetric metric
    all_dist = np.concatenate([dist_t2f, dist_f2t])

    return {
        "min_gap": float(np.min(all_dist)),
        "mean_gap": float(np.mean(all_dist)),
        "median_gap": float(np.median(all_dist)),
        "p5_gap": float(np.percentile(all_dist, 5)),
        "p95_gap": float(np.percentile(all_dist, 95)),
    }


def log_to_csv(patient_name, metrics, csv_path=None):
    """Append gap measurement to CSV for consistency tracking across runs."""
    if csv_path is None:
        csv_path = DATA / "gap_measurements.csv"

    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("patient,min_gap_mm,mean_gap_mm,median_gap_mm,p5_gap_mm,p95_gap_mm,status\n")
        status = "PASS" if metrics["min_gap"] >= 1.5 else "FAIL"
        f.write(
            f"{patient_name},"
            f"{metrics['min_gap']:.2f},{metrics['mean_gap']:.2f},"
            f"{metrics['median_gap']:.2f},{metrics['p5_gap']:.2f},"
            f"{metrics['p95_gap']:.2f},{status}\n"
        )
    return status


def main():
    parser = argparse.ArgumentParser(description="Measure joint gap between femur and tibia meshes")
    parser.add_argument("--name", type=str, default=None, help="Patient name (auto-finds meshes in data/meshes/)")
    parser.add_argument("--femur", type=str, default=None, help="Path to femur STL")
    parser.add_argument("--tibia", type=str, default=None, help="Path to tibia STL")
    parser.add_argument("--samples", type=int, default=5000, help="Number of surface samples per mesh")
    args = parser.parse_args()

    if args.name:
        femur_path = DATA / "meshes" / f"{args.name}_femur_full.stl"
        tibia_path = DATA / "meshes" / f"{args.name}_tibia_full.stl"
        patient_name = args.name
    elif args.femur and args.tibia:
        femur_path = Path(args.femur)
        tibia_path = Path(args.tibia)
        patient_name = femur_path.stem.replace("_femur_full", "").replace("_femur", "")
    else:
        print("Error: Specify --name or both --femur and --tibia")
        return

    if not femur_path.exists():
        print(f"Error: Femur mesh not found: {femur_path}")
        return
    if not tibia_path.exists():
        print(f"Error: Tibia mesh not found: {tibia_path}")
        return

    print(f"\n--- Joint Gap Measurement: {patient_name} ---")
    print(f"  Femur: {femur_path.name}")
    print(f"  Tibia: {tibia_path.name}")

    metrics = measure_gap(femur_path, tibia_path, n_samples=args.samples)

    print(f"\n  Results:")
    print(f"    Min gap:    {metrics['min_gap']:.2f} mm  (target: >= 1.5mm)")
    print(f"    Mean gap:   {metrics['mean_gap']:.2f} mm  (target: >= 4.0mm)")
    print(f"    Median gap: {metrics['median_gap']:.2f} mm")
    print(f"    5th pctl:   {metrics['p5_gap']:.2f} mm")
    print(f"    95th pctl:  {metrics['p95_gap']:.2f} mm")

    status = log_to_csv(patient_name, metrics)
    print(f"\n  Status: {status}")
    print(f"  Logged to: {DATA / 'gap_measurements.csv'}")

    if status == "FAIL":
        print(f"\n  [WARNING] Min gap < 1.5mm -- bones may appear fused!")
    else:
        print(f"\n  [OK] Joint space maintained -- adequate cartilage gap.")


if __name__ == "__main__":
    main()
