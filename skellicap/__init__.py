import os

from .pose_tracker import PoseTracker
from .pose_analyzer import PoseAnalyzer

# Load version from VERSION file
try:
    version_path = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_path, 'r') as f:
        __version__ = f.read().strip()
except Exception:
    __version__ = "unknown"
