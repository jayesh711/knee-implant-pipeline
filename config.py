import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"

# Voxel/Segmentation Settings
LABEL_FEMUR = 24
LABEL_TIBIA = 25
MIN_BONE_VOLUME_CC = 100.0  # Threshold for bone detection

# HU thresholds (Clinical CT Values)
HU_CORTICAL_MIN  = 700
HU_SPONGY_MAX    = 300
HU_BONE_MIN      = 250      # Finalized threshold for clinical-grade bone surface precision
HU_BONE_MAX      = 2500     # Maximum bone HU for segmentation
HU_CANAL_MAX     = 400
HU_METAL_MIN     = 2500     # Surgical hardware threshold

# Preprocessing: Phase 1 Foundations
HU_WINDOW_LOW    = -200
HU_WINDOW_HIGH   = 3071
HU_BIN_WIDTH     = 25       # 25 HU steps
NORM_METHOD      = "zscore"

# Mesh Processing Settings
MAX_TRIANGLES    = 2_000_000  # Matched to JPlanner-A high-fidelity output
SMOOTH_ITERS     = 50         # Increased for organic surface finish
MESH_SIGMA       = 0.4        # Preserve more anatomical detail
REMESHER_TARGET_LEN = 0.3     # Match CT resolution (approx. 0.3mm - 0.5mm)
TAUBIN_LAMBDA    = 0.5
TAUBIN_MU        = -0.53
MAX_HOLE_DIAMETER_MM = 10.0   # Preserve intercondylar notch while closing small gaps
FEMUR_HEAD_RATIO_MIN = 1.5
FEMUR_HEAD_RATIO_MAX = 1.8
SHAFT_BRIDGE_MARGIN_MM = 15.0 # How much to overlap the bridge with the fragments

# Adaptive Morphological Closing (physical mm, not voxel counts)
CLOSING_MM_FEMUR = 1.5       # Bridge intra-bone gaps without crossing joint (Fix #2)
CLOSING_MM_TIBIA = 1.5       # Match femur for consistent gap maintenance


# Joint Gap Enforcement
JOINT_GAP_MM     = 3.0       # Minimum maintained gap between femur/tibia surfaces

# Component Filtering
COMPONENT_MIN_PCT = 5.0      # Keep components >= 5% of largest (was 1% — too noisy)

# Preprocessing Settings
DEFAULT_SPACING     = (0.5, 0.5, 0.5)  # 0.5mm isotropic
DEFAULT_ORIENTATION = "LPS"

# TotalSegmentator License Key
TOTALSEG_LICENSE_KEY = os.getenv("TOTALSEG_LICENSE_KEY", "")

# 3D Slicer Path for visualization
SLICER_PATH = os.getenv("SLICER_PATH", r"D:\3D Slicer 5.10.0\Slicer.exe")
