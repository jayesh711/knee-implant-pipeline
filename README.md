# Knee Implant Pipeline

Automated clinical pipeline that converts raw knee CT scans (DICOM) into clean, validated 3D bone meshes (STL) for orthopedic surgery planning.

## What It Does

Processes a patient CT scan end-to-end:

1. Ingests DICOM → NIfTI with HU windowing, denoising, and isotropic resampling
2. Runs AI segmentation (TotalSegmentator v2) to isolate femur and tibia
3. Extracts and refines 3D surface meshes (Marching Cubes + Taubin smoothing + QEM decimation)
4. Optionally measures medullary canal centerlines and diameters
5. Generates an HU-threshold baseline mesh for validation
6. Launches 3D Slicer for clinical review

---

## Directory Structure

```
knee-implant-pipeline/
├── config.py                       # All HU thresholds, mesh settings, paths
├── .env                            # License keys and environment-specific paths
├── conda_env.yaml                  # Conda environment definition
├── requirements.txt                # pip dependencies
│
├── scripts/
│   ├── run_patient.py              # Entry point — single patient
│   ├── batch_process.py            # Entry point — multiple patients
│   ├── ingest_dicom.py             # Phase 0: DICOM → NIfTI preprocessing
│   ├── reconstruct_ground_truth.py # HU-based validation baseline mesh
│   ├── visualize_results.py        # 3D Slicer / PyVista visualization
│   │
│   ├── phase1/
│   │   ├── 02_segment.py               # AI segmentation (TotalSegmentator v2)
│   │   ├── 03_extract_and_mesh.py      # Bone extraction & mesh generation
│   │   ├── verify_labels.py            # Post-segmentation label validation
│   │   └── advanced_femur_reconstruct.py  # Fallback reconstruction when AI fails
│   │
│   ├── canal/
│   │   └── canal_measurement.py    # Medullary canal centerline & diameter
│   │
│   ├── mesh/
│   │   └── mesh_quality.py         # Standalone mesh quality checker (CLI)
│   │
│   ├── validation/
│   │   ├── metrics.py              # Dice, Hausdorff, and other accuracy metrics
│   │   └── validate_comparison.py  # Compare pipeline output vs reference STLs
│   │
│   └── utils/
│       └── io.py                   # Shared I/O helpers
│
├── models/                         # Pre-trained AI model weights (not committed)
│   ├── totalsegmentator/           # TotalSegmentator v2 weights
│   ├── VISTA/                      # NVIDIA VISTA 2D/3D (future use)
│   └── NV-Segment-CTMR/           # NV-Segment framework (future use)
│
└── data/                           # Patient data (not committed)
    ├── DICOM/                      # Raw DICOM scans, one subfolder per patient
    ├── NIfTI/                      # Preprocessed volumetric files
    ├── segmentations/phase1/       # TotalSegmentator output label maps
    ├── meshes/                     # Final STL outputs
    └── canal/                      # Canal centerline reports and skeletons
```

---

## Setup

**Prerequisites:** Conda, CUDA-capable GPU (8 GB+ VRAM recommended)

```bash
conda env create -f conda_env.yaml
conda activate knee-pipeline
```

Copy `.env` and fill in your values:

```ini
TOTALSEG_LICENSE_KEY=your_key_here
SLICER_PATH=C:\path\to\Slicer.exe
```

The TotalSegmentator license is required to use the `appendicular_bones` task (tibia with high accuracy). Without it the pipeline falls back to the unlicensed `total` task.

---

## Running the Pipeline

**Single patient (standard):**
```bash
python -m scripts.run_patient "data/DICOM/PatientFolder" --name PatientName
```

**With medullary canal analysis:**
```bash
python -m scripts.run_patient "data/DICOM/PatientFolder" --name PatientName --canal
```

**Post-op patient with metal implants** (enables HU-based hardware filter):
```bash
python -m scripts.run_patient "data/DICOM/PatientFolder" --name PatientName --has-metal
```

**Batch processing** (scans all DICOM subfolders):
```bash
python -m scripts.batch_process
```

**Visualize results in 3D Slicer:**
```bash
python -m scripts.visualize_results --name PatientName
```

**Check mesh quality:**
```bash
python -m scripts.mesh.mesh_quality --mesh_path data/meshes/PatientName_femur_full.stl
```

---

## Pipeline Phases

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| 0 – Ingest | `ingest_dicom.py` | DICOM folder | `data/NIfTI/{name}_raw.nii.gz` |
| 1 – Segment | `phase1/02_segment.py` | Raw NIfTI | `data/segmentations/phase1/{name}.nii` |
| 2 – Mesh | `phase1/03_extract_and_mesh.py` | Segmentation | `data/meshes/{name}_{bone}_full.stl` |
| 3 – Canal *(opt-in)* | `canal/canal_measurement.py` | Segmentation | `data/canal/{name}/` |
| 4 – Ground Truth | `reconstruct_ground_truth.py` | Raw NIfTI | `data/meshes/{name}_ground_truth.stl` |

---

## Key Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HU_METAL_MIN` | 2500 HU | Hardware filter threshold — opt-in via `--has-metal` |
| `MAX_TRIANGLES` | 500,000 | Maximum faces per output mesh |
| `SMOOTH_ITERS` | 5 | Taubin smoothing iterations |
| `TAUBIN_LAMBDA` / `TAUBIN_MU` | 0.5 / -0.53 | Taubin smoothing parameters |
| `DEFAULT_SPACING` | 0.5 mm | Isotropic resampling resolution |
| `MAX_HOLE_DIAMETER_MM` | 10.0 mm | Largest hole PyMeshLab will fill |

---

## Segmentation Strategy

The pipeline runs TotalSegmentator v2 with two complementary tasks and merges the results:

- **`appendicular_bones` task** — best source for tibia (label 2), requires license
- **`total` task** — best source for femur (labels 75 = left, 76 = right), full shaft coverage

Each bone is extracted from its most reliable source. If only one task is available, the pipeline degrades gracefully and exports whichever bones it can find.

---

## Dependencies

- Python 3.10, conda env `knee-pipeline`
- TotalSegmentator 2.x (license required for `appendicular_bones` task)
- PyMeshLab 2025.7 — Taubin smoothing, QEM decimation, hole filling
- nibabel, SimpleITK — medical image I/O
- scikit-image — Marching Cubes, morphological operations
- trimesh — mesh loading and inspection
- PyTorch CUDA 11.8, MONAI 1.4.0
- 3D Slicer 5.10+ — primary visualization (PyVista used as fallback)
