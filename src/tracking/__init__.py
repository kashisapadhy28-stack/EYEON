"""
Tracking Module for EYEON.
"""

from src.tracking.tracker import (
    MultiObjectTracker,
    ByteTracker,
    BoTSORTTracker,
    get_track_color,
    process_video_tracking,
)

__all__ = [
    "MultiObjectTracker",
    "ByteTracker",
    "BoTSORTTracker",
    "get_track_color",
    "process_video_tracking",
]
