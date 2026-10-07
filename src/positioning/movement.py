"""
EYEON - Movement Analysis Module

Tracks velocity vectors, estimates speed, direction of movement, and detects loitering behavior.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class MovementState:
    """
    Stores movement metrics for a tracked object/person.
    """
    track_id: int
    current_pos: Tuple[int, int]
    speed_px_per_sec: float = 0.0
    direction_deg: float = 0.0
    is_loitering: bool = False
    loitering_seconds: float = 0.0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, timestamp)


class MovementAnalyzer:
    """
    Analyzes spatial movements, velocity vectors, and loitering durations.
    """

    def __init__(
        self,
        history_seconds: float = 3.0,
        loitering_radius_px: float = 40.0,
        loitering_threshold_sec: float = 10.0,
    ) -> None:
        self.history_seconds = history_seconds
        self.loitering_radius_px = loitering_radius_px
        self.loitering_threshold_sec = loitering_threshold_sec

        # {track_id: MovementState}
        self._tracks: Dict[int, MovementState] = {}

    def update(
        self,
        tracks: List[Dict[str, Any]],
        timestamp: float,
    ) -> Dict[int, MovementState]:
        """
        Updates movement state for current frame tracks.
        """
        current_ids = {t["track_id"] for t in tracks}

        # Expire missing tracks
        for tid in list(self._tracks.keys()):
            if tid not in current_ids:
                del self._tracks[tid]

        results: Dict[int, MovementState] = {}

        for track in tracks:
            tid = track["track_id"]
            bbox = track["bbox"]  # [x1, y1, x2, y2]
            foot_x = (bbox[0] + bbox[2]) / 2.0
            foot_y = float(bbox[3])

            if tid not in self._tracks:
                ms = MovementState(track_id=tid, current_pos=(int(foot_x), int(foot_y)))
                self._tracks[tid] = ms

            ms = self._tracks[tid]
            ms.current_pos = (int(foot_x), int(foot_y))
            ms.trajectory.append((foot_x, foot_y, timestamp))

            # Prune trajectory older than history window
            cutoff = timestamp - self.history_seconds
            ms.trajectory = [p for p in ms.trajectory if p[2] >= cutoff]

            # Calculate speed and direction over history window
            if len(ms.trajectory) >= 2:
                oldest_x, oldest_y, oldest_t = ms.trajectory[0]
                dt = timestamp - oldest_t

                if dt > 0.1:
                    dx = foot_x - oldest_x
                    dy = foot_y - oldest_y
                    dist = math.hypot(dx, dy)

                    ms.speed_px_per_sec = dist / dt
                    ms.direction_deg = math.degrees(math.atan2(dy, dx)) % 360.0

            # Calculate loitering
            # Check how long the position has stayed within loitering_radius_px
            if len(ms.trajectory) >= 2:
                # Find the earliest point in continuous trajectory within radius
                loiter_start_t = timestamp
                for tx, ty, tt in ms.trajectory:
                    d = math.hypot(foot_x - tx, foot_y - ty)
                    if d <= self.loitering_radius_px:
                        loiter_start_t = min(loiter_start_t, tt)
                    else:
                        loiter_start_t = timestamp
                        break

                ms.loitering_seconds = round(timestamp - loiter_start_t, 1)
                ms.is_loitering = ms.loitering_seconds >= self.loitering_threshold_sec

            results[tid] = ms

        return results
