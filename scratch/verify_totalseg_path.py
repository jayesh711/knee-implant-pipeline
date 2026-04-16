
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to ensure config is loadable
sys.path.append(os.getcwd())

# Load environment
from config import MODELS
print(f"Project MODELS dir: {MODELS}")

# Manually trigger load_dotenv if not already done in config
load_dotenv()

# Set env var for this process (though it should be loaded from .env)
# The library checks os.environ['TOTALSEG_WEIGHTS_PATH']
weights_path = os.getenv('TOTALSEG_WEIGHTS_PATH')
print(f"TOTALSEG_WEIGHTS_PATH from env: {weights_path}")

try:
    from totalsegmentator.libs import get_weights_dir
    actual_dir = get_weights_dir()
    print(f"TotalSegmentator is using: {actual_dir}")
    
    if str(actual_dir).replace('\\', '/').endswith('models/totalsegmentator'):
        print("SUCCESS: TotalSegmentator is picking up the project-local models!")
    else:
        print("WARNING: TotalSegmentator is still using the default path or another path.")
except ImportError:
    print("TotalSegmentator library not found in this environment.")
except Exception as e:
    print(f"Error checking weights: {e}")
