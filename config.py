import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT        = Path(os.getenv("PROJECT_ROOT", "."))
DATA        = ROOT / "data"
MODELS      = ROOT / "models"
NV_SEG_DIR  = ROOT / "models/NV-Segment-CTMR/NV-Segment-CT"
NV_WEIGHTS  = ROOT / "models/weights/NV-Segment-CT"
OUTPUTS     = DATA / "outputs"
LOGS        = ROOT / "logs"

# Segmentation label IDs (verify these against label_dict.json after first run)
LABEL_FEMUR = 24
LABEL_TIBIA = 25

# HU thresholds
HU_CORTICAL_MIN  = 700
HU_CORTICAL_MAX  = 1800
HU_CANAL_MIN     = -100
HU_CANAL_MAX     = 400
HU_METAL_MIN     = 2500   # Phase 2: implant isolation
HU_METAL_ARTIFACT = 3000

# Preprocessing: Phase 1 Foundations
HU_WINDOW_LOW    = -200
HU_WINDOW_HIGH   = 3000
HU_BIN_WIDTH     = 25
NORM_METHOD      = "zscore"  # choices: "zscore", "minmax", "none"

# Mesh settings
MAX_TRIANGLES    = 150_000
SMOOTH_ITERS     = 10

# Preprocessing Settings
DEFAULT_SPACING     = (0.5, 0.5, 0.5)  # 0.5mm isotropic
DEFAULT_ORIENTATION = "LPS"             # Standard medical orientation
