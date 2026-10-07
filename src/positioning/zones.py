"""
EYEON - Phase 5: Positioning & Zones

ZoneManager:
  - Loads polygon zones from config.yaml (zones.list)
  - Computes foot-point (bottom-center) for every tracked bbox
  - Tests zone membership using cv2.pointPolygonTest (no Shapely needed)
  - Tracks per-(track_id, zone_name) dwell time
  - Emits ENTERED / INSIDE / EXITED events each frame
  - Scales config polygons from config resolution to working resolution

ZoneEvent:
  - Lightweight dataclass for per-frame zone transition events

ZoneInfo:
  - Per-track result returned by ZoneManager.update()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ZoneEventType(Enum):
    ENTERED = auto()
    INSIDE  = auto()
    EXITED  = auto()


@dataclass
class ZoneEvent:
    """A single zone transition event for one track."""
    track_id:   int
    zone_name:  str
    event_type: ZoneEventType
    timestamp:  float  # seconds


@dataclass
class ZoneInfo:
    """
    Full zone context for one track in a given frame.

    Attributes:
        track_id:      The tracker's assigned ID.
        foot_point:    (x, y) ground-contact pixel in the working frame.
        current_zones: Names of zones the foot-point is currently inside.
        zone_weight:   Highest zone weight among occupied zones (for risk engine).
        dwell_seconds: How many seconds this track has been in each zone.
        events:        ENTERED / EXITED events fired this frame.
    """
    track_id:      int
    foot_point:    Tuple[int, int]
    current_zones: List[str]            = field(default_factory=list)
    zone_weight:   float                = 0.0
    dwell_seconds: Dict[str, float]     = field(default_factory=dict)
    events:        List[ZoneEvent]      = field(default_factory=list)


@dataclass
class _Zone:
    """Internal zone record."""
    name:    str
    weight:  float
    contour: np.ndarray   # shape (N, 1, 2), int32 — cv2-compatible


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class ZoneManager:
    """
    Manages polygon zone definitions and per-track zone state.

    Usage:
        zm = ZoneManager(frame_width=640, frame_height=360)
        zone_info = zm.update(tracks, timestamp=t)
        # zone_info: Dict[track_id, ZoneInfo]
    """

    # Reference resolution in which config polygons were drawn.
    # Change this if your config.yaml polygons use a different coordinate space.
    CONFIG_REF_W: int = 1280
    CONFIG_REF_H: int = 720

    def __init__(
        self,
        frame_width:  int = 640,
        frame_height: int = 360,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.frame_width  = frame_width
        self.frame_height = frame_height

        cfg = config or load_config()
        self._zones: List[_Zone] = self._load_zones(cfg)

        # Per-(track_id, zone_name) entry timestamp
        self._entry_timestamps: Dict[Tuple[int, str], float] = {}

        # Zones occupied last frame: Dict[track_id → set of zone names]
        self._prev_zones: Dict[int, set] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks:    List[Dict[str, Any]],
        timestamp: float = 0.0,
    ) -> Dict[int, ZoneInfo]:
        """
        Process one frame's tracks and return zone info for every track.

        Args:
            tracks:    List of track dicts with keys 'track_id', 'bbox', 'class'.
            timestamp: Current video timestamp in seconds.

        Returns:
            Dict mapping track_id → ZoneInfo.
        """
        results: Dict[int, ZoneInfo] = {}

        current_ids = {t["track_id"] for t in tracks}

        # Expire dwell timers for tracks that disappeared
        for (tid, zn) in list(self._entry_timestamps.keys()):
            if tid not in current_ids:
                del self._entry_timestamps[(tid, zn)]
        for tid in list(self._prev_zones.keys()):
            if tid not in current_ids:
                del self._prev_zones[tid]

        for track in tracks:
            tid  = track["track_id"]
            bbox = track["bbox"]   # [x1, y1, x2, y2]

            foot = self._foot_point(bbox)
            occupied = self._check_zones(foot)           # set of zone names
            prev     = self._prev_zones.get(tid, set())

            events: List[ZoneEvent] = []
            dwell:  Dict[str, float] = {}

            for zone_name in occupied:
                key = (tid, zone_name)

                if zone_name not in prev:
                    # ENTERED this frame
                    self._entry_timestamps[key] = timestamp
                    events.append(ZoneEvent(tid, zone_name, ZoneEventType.ENTERED, timestamp))
                else:
                    # INSIDE — accumulate dwell
                    entry_t = self._entry_timestamps.get(key, timestamp)
                    dwell[zone_name] = round(timestamp - entry_t, 2)
                    events.append(ZoneEvent(tid, zone_name, ZoneEventType.INSIDE, timestamp))

            for zone_name in prev - occupied:
                # EXITED this frame
                key = (tid, zone_name)
                self._entry_timestamps.pop(key, None)
                events.append(ZoneEvent(tid, zone_name, ZoneEventType.EXITED, timestamp))

            # Max weight of occupied zones (0.0 if outside all zones)
            max_weight = max(
                (z.weight for z in self._zones if z.name in occupied),
                default=0.0,
            )

            results[tid] = ZoneInfo(
                track_id      = tid,
                foot_point    = foot,
                current_zones = list(occupied),
                zone_weight   = max_weight,
                dwell_seconds = dwell,
                events        = events,
            )

            self._prev_zones[tid] = occupied

        return results

    @property
    def zones(self) -> List[_Zone]:
        """Expose zone list for drawing routines."""
        return self._zones

    def draw_zones(
        self,
        frame:      np.ndarray,
        alpha:      float = 0.25,
        font_scale: float = 0.50,
    ) -> None:
        """
        Draw filled semi-transparent zone polygons on *frame* in-place.

        Each zone gets a distinct colour from the palette below.
        Zone name is drawn at the centroid.
        """
        # Distinct soft colours per zone (BGR)
        palette = [
            (255, 140,  60),   # soft orange
            ( 60, 200, 255),   # cyan
            (120, 255, 120),   # lime
            (200,  80, 255),   # violet
            (255, 255,  80),   # yellow
        ]

        for idx, zone in enumerate(self._zones):
            color = palette[idx % len(palette)]

            # Filled overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone.contour], color)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Border
            cv2.polylines(frame, [zone.contour], isClosed=True, color=color, thickness=2)

            # Zone name at centroid
            moments = cv2.moments(zone.contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = int(zone.contour[:, 0, 0].mean()), int(zone.contour[:, 0, 1].mean())

            (tw, th), _ = cv2.getTextSize(zone.name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx, ty = cx - tw // 2, cy + th // 2

            # Pill background
            pad = 4
            sub = frame[ty - th - pad: ty + pad, tx - pad: tx + tw + pad]
            if sub.size > 0:
                bg = np.zeros_like(sub)
                bg[:] = (15, 15, 15)
                cv2.addWeighted(bg, 0.70, sub, 0.30, 0, sub)
                frame[ty - th - pad: ty + pad, tx - pad: tx + tw + pad] = sub

            cv2.putText(
                frame, zone.name,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_zones(self, cfg: Dict[str, Any]) -> List[_Zone]:
        """Parse zones from config and scale polygons to working resolution."""
        zones_cfg = cfg.get("zones", {})
        if not zones_cfg.get("enabled", True):
            return []

        zone_list = zones_cfg.get("list", [])
        loaded: List[_Zone] = []

        sx = self.frame_width  / self.CONFIG_REF_W
        sy = self.frame_height / self.CONFIG_REF_H

        for z in zone_list:
            raw_poly = z.get("polygon", [])
            if len(raw_poly) < 3:
                continue  # degenerate polygon, skip

            pts = np.array(
                [[int(p[0] * sx), int(p[1] * sy)] for p in raw_poly],
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            loaded.append(_Zone(
                name    = str(z.get("name", f"Zone{len(loaded)}")),
                weight  = float(z.get("weight", 0.5)),
                contour = pts,
            ))

        return loaded

    @staticmethod
    def _foot_point(bbox: List[int]) -> Tuple[int, int]:
        """
        Returns the ground-contact point of a bounding box.

        For standard CCTV (slightly downward angle):
            foot = bottom-center = ( (x1+x2)//2 , y2 )
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    def _check_zones(self, point: Tuple[int, int]) -> set:
        """
        Returns the set of zone names whose polygon contains *point*.

        Uses cv2.pointPolygonTest which returns >= 0 when on boundary or inside.
        """
        occupied = set()
        pt = (float(point[0]), float(point[1]))
        for zone in self._zones:
            result = cv2.pointPolygonTest(zone.contour, pt, measureDist=False)
            if result >= 0:
                occupied.add(zone.name)
        return occupied
