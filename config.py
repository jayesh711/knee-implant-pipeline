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
HU_CANAL_MAX     = 400
HU_METAL_MIN     = 2500     # Surgical hardware threshold

# Preprocessing: Phase 1 Foundations
HU_WINDOW_LOW    = -200
HU_WINDOW_HIGH   = 3000
HU_BIN_WIDTH     = 25       # 25 HU steps
NORM_METHOD      = "zscore"

# Mesh Processing Settings
MAX_TRIANGLES    = 500_000  # High res for full bones
SMOOTH_ITERS     = 10       # Light Taubin smoothing
TAUBIN_LAMBDA    = 0.5
TAUBIN_MU        = -0.53
MAX_HOLE_DIAMETER_MM = 10.0

# Preprocessing Settings
DEFAULT_SPACING     = (0.5, 0.5, 0.5)  # 0.5mm isotropic
DEFAULT_ORIENTATION = "LPS"

# TotalSegmentator License Key
TOTALSEG_LICENSE_KEY = os.getenv("TOTALSEG_LICENSE_KEY", "")

# 3D Slicer Path for visualization
SLICER_PATH = os.getenv("SLICER_PATH", r"D:\3D Slicer 5.10.0\Slicer.exe")
