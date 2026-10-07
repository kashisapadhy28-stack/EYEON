"""
EYEON - Phase 5: Positioning & Zones Video Pipeline

Extends Phase 4 AssociationTracker with ZoneManager:
  - Polygon zone overlays (semi-transparent filled)
  - Foot-point dot per person
  - Zone badge (name + dwell time) above person label
  - All Phase 4 intelligence retained: state machine, temporal voting, alerts
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.association.association import (
    PersonObjectIntelligence,
    ObjectState,
)
from src.association.association_video import AssociationTracker, _draw_label
from src.positioning.zones import ZoneManager, ZoneInfo, ZoneEventType
from src.utils.config_loader import load_config, get_path


# ---------------------------------------------------------------------------
# Full pipeline — extends AssociationTracker with zone awareness
# ---------------------------------------------------------------------------

def process_video(
    input_path:  Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    display:     bool = True,
    max_frames:  Optional[int] = None,
) -> None:
    """
    Phase 5 full video pipeline:
      Tracking → Association → Zone Analysis → Visualization

    Args:
        input_path:  Path to source video file.
        output_path: Optional path to save annotated output video.
        display:     Show live OpenCV window while processing.
        max_frames:  Stop after this many frames (None = full video).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        input_path = _ROOT / input_path
    if not input_path.exists():
        print(f"ERROR: Video not found at {input_path}")
        return

    # ------------------------------------------------------------------
    # Initialise tracker
    # ------------------------------------------------------------------
    tracker = AssociationTracker()
    cap     = cv2.VideoCapture(str(input_path))

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_h > orig_w:
        width, height = 360, 640
    else:
        width, height = 640, 360

    # ZoneManager — polygons scale to working resolution
    zone_manager = ZoneManager(frame_width=width, frame_height=height)

    # ------------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------------
    out = None
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    window_name = "EYEON - Phase 5 Positioning & Zones"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1024), min(height, 576))

    frame_count = 0
    start_time  = time.time()

    print("=" * 65)
    print("  EYEON Phase 5 - Positioning & Zone Intelligence")
    print(f"  Source:      {input_path.name}")
    print(f"  Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"  Zones:       {[z.name for z in zone_manager.zones]}")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count   += 1
            cur_timestamp  = frame_count / fps
            frame          = cv2.resize(frame, (width, height))

            # --------------------------------------------------------------
            # Phase 4: tracking + association
            # --------------------------------------------------------------
            tracks, associations, object_states = tracker.process_frame(
                frame, timestamp=cur_timestamp
            )

            # --------------------------------------------------------------
            # Phase 5: zone membership for every track
            # --------------------------------------------------------------
            zone_info: Dict[int, ZoneInfo] = zone_manager.update(
                tracks, timestamp=cur_timestamp
            )

            # --------------------------------------------------------------
            # Draw layer 0: zone polygons (bottom layer, drawn first)
            # --------------------------------------------------------------
            zone_manager.draw_zones(frame, alpha=0.18)

            # --------------------------------------------------------------
            # Draw layer 1: Person boxes + foot-point + zone badge
            # --------------------------------------------------------------
            for track in tracks:
                if track["class"] != "person":
                    continue

                x1, y1, x2, y2 = track["bbox"]
                pid = track["track_id"]
                color_p = (46, 204, 113)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color_p, 2)

                # Foot-point dot
                zi   = zone_info.get(pid)
                foot = zi.foot_point if zi else ((x1 + x2) // 2, y2)
                cv2.circle(frame, foot, 4, color_p, -1)
                cv2.circle(frame, foot, 4, (255, 255, 255), 1)

                # Person label
                label_y = max(y1 - 4, 36)
                _draw_label(frame, f"Person #{pid}", (x1, label_y), color_p, font_scale=0.46)

                # Zone badge: [ZoneName Xs] or [--]
                if zi and zi.current_zones:
                    zone_parts = []
                    for zn in zi.current_zones:
                        dwell = zi.dwell_seconds.get(zn, 0.0)
                        zone_parts.append(f"{zn} {dwell:.0f}s")
                    badge_text = "  ".join(zone_parts)
                    badge_color = (60, 200, 255)   # cyan
                else:
                    badge_text  = "--"
                    badge_color = (150, 150, 150)

                _draw_label(
                    frame, badge_text,
                    (x1, label_y - 20),
                    badge_color,
                    font_scale=0.38,
                )

            # --------------------------------------------------------------
            # Draw layer 2: Object boxes (state-aware, from Phase 4)
            # --------------------------------------------------------------
            unattended_count = 0
            for track in tracks:
                if track["class"] == "person":
                    continue

                oid  = track["track_id"]
                ocls = track["class"]
                x1, y1, x2, y2 = track["bbox"]

                state_rec      = object_states.get(oid, {})
                state          = state_rec.get("state", ObjectState.UNKNOWN)
                owner          = state_rec.get("confirmed_owner", None)
                unattended_sec = state_rec.get("unattended_seconds", 0.0)

                if state == ObjectState.CARRIED:
                    color = (0, 220, 255)
                    line1 = f"{ocls.capitalize()} #{oid}"
                    line2 = f"CARRIED  P#{owner}"
                elif state == ObjectState.DETACHED:
                    color = (0, 165, 255)
                    line1 = f"{ocls.capitalize()} #{oid}"
                    line2 = f"DETACHED  P#{owner}  {unattended_sec:.0f}s"
                elif state == ObjectState.UNATTENDED:
                    color = (0, 60, 255)
                    line1 = f"{ocls.capitalize()} #{oid}"
                    line2 = f"!! UNATTENDED  {unattended_sec:.0f}s"
                    unattended_count += 1
                else:
                    color = (180, 180, 180)
                    line1 = f"{ocls.capitalize()} #{oid}"
                    line2 = "DETECTED"

                thickness = 3 if state == ObjectState.UNATTENDED else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                line_h  = 18
                label_y = max(y1 - 4, line_h * 2 + 8)
                _draw_label(frame, line1, (x1, label_y - line_h), color, font_scale=0.42)
                _draw_label(frame, line2, (x1, label_y),          color, font_scale=0.40)

            # --------------------------------------------------------------
            # Draw layer 3: Association lines
            # --------------------------------------------------------------
            for assoc in associations:
                pid = assoc["person_track_id"]
                oid = assoc["object_track_id"]

                p_box = next(
                    (t["bbox"] for t in tracks if t["track_id"] == pid and t["class"] == "person"),
                    None,
                )
                o_box = next(
                    (t["bbox"] for t in tracks if t["track_id"] == oid),
                    None,
                )
                if p_box and o_box:
                    pc = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
                    oc = ((o_box[0] + o_box[2]) // 2, (o_box[1] + o_box[3]) // 2)
                    cv2.line(frame, pc, oc, (0, 230, 255), 1, cv2.LINE_AA)

            # --------------------------------------------------------------
            # Draw layer 4: HUD bar
            # --------------------------------------------------------------
            avg_fps = frame_count / (time.time() - start_time + 1e-6)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 32), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            hud = (
                f"EYEON P5  |  Frame {frame_count:04d}"
                f"  |  {avg_fps:.1f} FPS"
                f"  |  Tracks: {len(tracks)}"
                f"  |  Links: {len(associations)}"
            )
            cv2.putText(
                frame, hud, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (0, 240, 190), 1, cv2.LINE_AA,
            )

            # --------------------------------------------------------------
            # Draw layer 5: Unattended alert banner (bottom)
            # --------------------------------------------------------------
            if unattended_count > 0:
                b_overlay = frame.copy()
                cv2.rectangle(b_overlay, (0, height - 34), (width, height), (0, 0, 160), -1)
                cv2.addWeighted(b_overlay, 0.88, frame, 0.12, 0, frame)
                cv2.putText(
                    frame,
                    f"  WARNING: {unattended_count} UNATTENDED OBJECT(S) DETECTED!",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

            # --------------------------------------------------------------
            if out:
                out.write(frame)

            if display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    print("\nStopped by user.")
                    break

            if max_frames and frame_count >= max_frames:
                break

    finally:
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\nCompleted: {frame_count} frames processed.")
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    test_vid = Path("data/videos/test_crowded_scene.mp4")
    out_vid  = Path("data/processed/test_crowded_scene_phase5.mp4")
    process_video(test_vid, out_vid)
