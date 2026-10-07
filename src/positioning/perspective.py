"""
EYEON - Perspective Transformation Module

Transforms 2D frame image coordinates (e.g. foot-points) to a top-down
bird's-eye view plane or real-world ground coordinates using planar Homography.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


class PerspectiveTransformer:
    """
    Computes homography matrix between camera image plane and top-down ground plane.
    """

    def __init__(
        self,
        src_points: Optional[List[Tuple[float, float]]] = None,
        dst_points: Optional[List[Tuple[float, float]]] = None,
        target_size: Tuple[int, int] = (600, 600),
    ) -> None:
        """
        Args:
            src_points: 4 points in image space [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            dst_points: 4 points in top-down bird's-eye space
            target_size: (width, height) of top-down bird's-eye view canvas
        """
        self.target_w, self.target_h = target_size

        if src_points is None:
            # Default camera trapezoid ROI
            src_points = [(100, 100), (540, 100), (600, 350), (40, 350)]

        if dst_points is None:
            # Default top-down rectangle
            dst_points = [
                (50, 50),
                (self.target_w - 50, 50),
                (self.target_w - 50, self.target_h - 50),
                (50, self.target_h - 50),
            ]

        self.src_pts = np.float32(src_points)
        self.dst_pts = np.float32(dst_points)

        # Compute Homography matrix H and its inverse H_inv
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        self.H_inv, _ = cv2.findHomography(self.dst_pts, self.src_pts)

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transforms a single (x, y) image point to (x_top, y_top) in top-down plane.
        """
        pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
        warped = np.dot(self.H, pt)
        if warped[2] != 0:
            warped = warped / warped[2]
        return (float(warped[0]), float(warped[1]))

    def transform_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transforms a list of (x, y) points.
        """
        return [self.transform_point(pt) for pt in points]

    def generate_birds_eye_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Warps the input camera frame into a top-down bird's-eye view frame.
        """
        return cv2.warpPerspective(frame, self.H, (self.target_w, self.target_h))
