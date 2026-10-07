"""
EYEON - BoT-SORT Tracker Module

Convenience wrapper around MultiObjectTracker configured for BoT-SORT tracking.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tracking.tracker import BoTSORTTracker, process_video_tracking

__all__ = ["BoTSORTTracker", "process_video_tracking"]

if __name__ == "__main__":
    process_video_tracking(tracker_type="botsort")