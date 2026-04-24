import subprocess
import sys
import os
import shutil
from pathlib import Path
from config import DATA, TOTALSEG_LICENSE_KEY

def _setup_license():
    """Activate TotalSegmentator license key if configured (one-time setup)."""
    if not TOTALSEG_LICENSE_KEY:
        print("  No license key configured — using free tasks only.")
        print("  (Set TOTALSEG_LICENSE_KEY in .env to unlock appendicular_bones, tissue_types, etc.)")
        return False

    # Find the totalseg_set_license executable
    license_exe = shutil.which("totalseg_set_license")
    if not license_exe:
        # Try alongside python executable
        scripts_dir = Path(sys.executable).parent / "Scripts"
        license_exe = scripts_dir / "totalseg_set_license.exe"
        if not license_exe.exists():
            print(f"  Warning: totalseg_set_license not found. Cannot activate license.")
            return False
        license_exe = str(license_exe)

    print(f"  Activating TotalSegmentator license...")
    try:
        subprocess.run(
            [str(license_exe), "-l", TOTALSEG_LICENSE_KEY],
            check=True, capture_output=True, text=True
        )
        print(f"  [OK] License activated successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: License activation failed ({e.stderr.strip()}). Using free tasks.")
        return False

def _find_totalseg_exe():
    """Find the TotalSegmentator executable dynamically."""
    # 1. Try shutil.which (checks PATH)
    exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
    if exe:
        return exe

    # 2. Try alongside current Python's Scripts folder
    scripts_dir = Path(sys.executable).parent / "Scripts"
    for name in ["TotalSegmentator.exe", "totalsegmentator.exe"]:
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    # 3. Hardcoded fallback (legacy)
    legacy = Path(r"C:\Users\dongr\anaconda3\envs\knee-pipeline\Scripts\totalsegmentator.exe")
    if legacy.exists():
        return str(legacy)

    return None

def run_segmentation(volume_name="S0001"):
    input_file = DATA / "NIfTI" / f"{volume_name}_raw.nii.gz"
    output_dir = DATA / "segmentations" / "phase1" / volume_name

    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find TotalSegmentator executable
    totalseg_exe = _find_totalseg_exe()
    if not totalseg_exe:
        print("Error: TotalSegmentator executable not found!")
        print("  Install it with: pip install TotalSegmentator")
        return

    print(f"--- Starting TotalSegmentator for {volume_name} ---")
    print(f"Input:  {input_file}")
    print(f"Executable: {totalseg_exe}")

    # Activate license if available
    has_license = _setup_license()

    # Strategy: If licensed, try the premium 'appendicular_bones' task first
    # (higher resolution for knee/limb bones). Fallback to 'total' if it fails.
    if has_license:
        tasks_to_try = [
            ("appendicular_bones", "Premium Appendicular Bones (licensed)"),
            ("total", "Standard Total Segmentation (free fallback)")
        ]
    else:
        tasks_to_try = [
            ("total", "Standard Total Segmentation (free)")
        ]

    completed_tasks = []
    for task_name, task_desc in tasks_to_try:
        print(f"\n  Running: {task_desc} (task={task_name})...")
        
        # Determine output filename for this task
        # If task is 'total', save as volume_name_total.nii.gz
        # Otherwise save as volume_name.nii.gz (primary)
        suffix = "_total" if task_name == "total" else ""
        task_output_file = output_dir.parent / f"{volume_name}{suffix}.nii.gz"
        
        try:
            subprocess.run([
                totalseg_exe,
                "-i", str(input_file),
                "-o", str(task_output_file), # TS can take a file path with --ml
                "--ml",
                "--task", task_name,
                "--quiet"
            ], check=True)
            print(f"  [OK] Successfully completed segmentation with task '{task_name}'")
            completed_tasks.append(task_name)
        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] Task '{task_name}' failed: {e}")
            continue
        except Exception as e:
            print(f"  An unexpected error occurred during task '{task_name}': {e}")
            continue

    if not completed_tasks:
        print(f"Error: All segmentation tasks failed for {volume_name}.")
    else:
        print(f"\n--- Segmentation Complete for {volume_name} (Tasks: {', '.join(completed_tasks)}) ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TotalSegmentator on NIfTI volume")
    parser.add_argument("--name", type=str, default="S0001", help="Volume name to segment")

    args = parser.parse_args()
    run_segmentation(args.name)
