"""
Detection and Video Input Module for EYEON.
"""

from src.detection.video_input import (
    VideoStream,
    FramePacket,
    process_video,
)

from src.detection.detector import (
    ObjectDetector,
    Detection,
    process_video_detection,
)

__all__ = [
    "VideoStream",
    "FramePacket",
    "process_video",
    "ObjectDetector",
    "Detection",
    "process_video_detection",
]
