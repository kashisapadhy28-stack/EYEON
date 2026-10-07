"""
EYEON - Phase 6: Risk Engine Video Pipeline

Full integrated pipeline:
  Phase 4  → Tracking + Association + Object States
  Phase 5  → Zone membership + Dwell time
  Phase 6  → Risk scoring + Alert state machine

Visual output per person:
  - Bounding box colour keyed to risk level:
      LOW    → green   (46, 204, 113)
      MEDIUM → amber   (0, 165, 255)
      HIGH   → red     (0, 60, 255)
  - Risk score badge: [0.73 HIGH]
  - Alert banner on HIGH alert_fired
  - Zone polygons (Phase 5, bottom layer)
  - Foot-point dots
  - Object state boxes (Phase 4)
  - Association lines
  - Reason tooltip on HIGH/MEDIUM (first reason string)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.alerts.alert_manager import AlertManager
from src.association.association import ObjectState
from src.association.association_video import AssociationTracker, _draw_label
from src.positioning.zones import ZoneManager, ZoneInfo
from src.risk.risk_engine import RiskEngine, RiskLevel, RiskRecord
from src.utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Risk level → visual colour (BGR)
# ---------------------------------------------------------------------------
_RISK_COLOR = {
    RiskLevel.LOW:    (46, 204, 113),   # green
    RiskLevel.MEDIUM: (0,  165, 255),   # amber
    RiskLevel.HIGH:   (0,   60, 255),   # red
}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path:  Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    display:     bool = True,
    max_frames:  Optional[int] = None,
) -> None:
    """
    Phase 6 full video pipeline:
      Tracking → Association → Zone Analysis → Risk Scoring → Visualization

    Args:
        input_path:  Source video file.
        output_path: Optional annotated output video path.
        display:     Show live OpenCV window.
        max_frames:  Stop after N frames (None = full video).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        input_path = _ROOT / input_path
    if not input_path.exists():
        print(f"ERROR: Video not found at {input_path}")
        return

    tracker = AssociationTracker()
    cap     = cv2.VideoCapture(str(input_path))

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_h > orig_w:
        width, height = 360, 640
    else:
        width, height = 640, 360

    zone_manager = ZoneManager(frame_width=width, frame_height=height)
    risk_engine  = RiskEngine()
    alert_manager = AlertManager()

    out = None
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    window_name = "EYEON - Phase 6 Risk Engine"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1024), min(height, 576))

    frame_count     = 0
    start_time      = time.time()
    total_alerts    = 0

    print("=" * 65)
    print("  EYEON Phase 6 - Risk Engine & Alert Pipeline")
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

            # ----------------------------------------------------------
            # Phase 4: Tracking + Association
            # ----------------------------------------------------------
            tracks, associations, object_states = tracker.process_frame(
                frame, timestamp=cur_timestamp
            )

            # ----------------------------------------------------------
            # Phase 5: Zone membership
            # ----------------------------------------------------------
            zone_info: Dict[int, ZoneInfo] = zone_manager.update(
                tracks, timestamp=cur_timestamp
            )

            # ----------------------------------------------------------
            # Phase 6: Risk scoring
            # ----------------------------------------------------------
            risk_records: Dict[int, RiskRecord] = risk_engine.update(
                tracks        = tracks,
                associations  = associations,
                object_states = object_states,
                zone_info     = zone_info,
                timestamp     = cur_timestamp,
            )

            # Process alerts (saves snapshots & logs to database)
            fired_alerts = alert_manager.process_risk_records(
                risk_records, frame=frame, timestamp=cur_timestamp
            )

            # Count new alerts this frame
            new_alerts = len(fired_alerts)
            total_alerts += new_alerts

            # ----------------------------------------------------------
            # Draw layer 0: Zone polygons
            # ----------------------------------------------------------
            zone_manager.draw_zones(frame, alpha=0.15)

            # ----------------------------------------------------------
            # Draw layer 1: Person boxes  (risk-level colour)
            # ----------------------------------------------------------
            for track in tracks:
                if track["class"] != "person":
                    continue

                pid          = track["track_id"]
                x1, y1, x2, y2 = track["bbox"]
                rec          = risk_records.get(pid)
                level        = rec.risk_level if rec else RiskLevel.LOW
                color        = _RISK_COLOR[level]
                score_str    = f"{rec.risk_score:.2f}" if rec else "?"

                # Box thickness scales with risk
                thickness = {RiskLevel.LOW: 2, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[level]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Foot-point dot
                zi   = zone_info.get(pid)
                foot = zi.foot_point if zi else ((x1 + x2) // 2, y2)
                cv2.circle(frame, foot, 4, color, -1)
                cv2.circle(frame, foot, 4, (255, 255, 255), 1)

                # Person + risk score label
                label_y = max(y1 - 4, 40)
                _draw_label(
                    frame,
                    f"P#{pid}  {score_str} {level.value}",
                    (x1, label_y),
                    color,
                    font_scale=0.44,
                )

                # Zone badge
                if zi and zi.current_zones:
                    zone_parts = [
                        f"{zn} {zi.dwell_seconds.get(zn, 0):.0f}s"
                        for zn in zi.current_zones
                    ]
                    _draw_label(
                        frame,
                        "  ".join(zone_parts),
                        (x1, label_y - 20),
                        (60, 200, 255),
                        font_scale=0.36,
                    )

                # First reason string for MEDIUM / HIGH
                if rec and level in (RiskLevel.MEDIUM, RiskLevel.HIGH) and rec.reasons:
                    first_reason = rec.reasons[0][:48]   # truncate long reasons
                    _draw_label(
                        frame,
                        first_reason,
                        (x1, label_y + 18),
                        color,
                        font_scale=0.33,
                    )

                # Flashing HIGH alert ring
                if rec and rec.alert_fired:
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                                  (0, 0, 255), 2)

            # ----------------------------------------------------------
            # Draw layer 2: Object boxes (state-aware)
            # ----------------------------------------------------------
            unattended_count = 0
            for track in tracks:
                if track["class"] == "person":
                    continue

                oid  = track["track_id"]
                ocls = track["class"]
                x1, y1, x2, y2 = track["bbox"]
                state_rec       = object_states.get(oid, {})
                state           = state_rec.get("state", ObjectState.UNKNOWN)
                owner           = state_rec.get("confirmed_owner", None)
                secs            = state_rec.get("unattended_seconds", 0.0)

                if state == ObjectState.CARRIED:
                    color = (0, 220, 255)
                    line1, line2 = f"{ocls.capitalize()} #{oid}", f"CARRIED  P#{owner}"
                elif state == ObjectState.DETACHED:
                    color = (0, 165, 255)
                    line1, line2 = f"{ocls.capitalize()} #{oid}", f"DETACHED  P#{owner}  {secs:.0f}s"
                elif state == ObjectState.UNATTENDED:
                    color = (0, 60, 255)
                    line1, line2 = f"{ocls.capitalize()} #{oid}", f"!! UNATTENDED  {secs:.0f}s"
                    unattended_count += 1
                else:
                    color = (180, 180, 180)
                    line1, line2 = f"{ocls.capitalize()} #{oid}", "DETECTED"

                thick = 3 if state == ObjectState.UNATTENDED else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                lh = 18
                ly = max(y1 - 4, lh * 2 + 8)
                _draw_label(frame, line1, (x1, ly - lh), color, font_scale=0.42)
                _draw_label(frame, line2, (x1, ly),      color, font_scale=0.40)

            # ----------------------------------------------------------
            # Draw layer 3: Association lines
            # ----------------------------------------------------------
            for assoc in associations:
                pid   = assoc["person_track_id"]
                oid   = assoc["object_track_id"]
                p_box = next((t["bbox"] for t in tracks
                              if t["track_id"] == pid and t["class"] == "person"), None)
                o_box = next((t["bbox"] for t in tracks
                              if t["track_id"] == oid), None)
                if p_box and o_box:
                    pc = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
                    oc = ((o_box[0] + o_box[2]) // 2, (o_box[1] + o_box[3]) // 2)
                    cv2.line(frame, pc, oc, (0, 230, 255), 1, cv2.LINE_AA)

            # ----------------------------------------------------------
            # Draw layer 4: HUD bar
            # ----------------------------------------------------------
            avg_fps = frame_count / (time.time() - start_time + 1e-6)

            # Count risk levels
            n_high   = sum(1 for r in risk_records.values() if r.risk_level == RiskLevel.HIGH)
            n_medium = sum(1 for r in risk_records.values() if r.risk_level == RiskLevel.MEDIUM)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 32), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            hud = (
                f"EYEON P6  |  F{frame_count:04d}"
                f"  |  {avg_fps:.1f} FPS"
                f"  |  H:{n_high} M:{n_medium}"
                f"  |  Alerts: {total_alerts}"
            )
            cv2.putText(frame, hud, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (0, 240, 190), 1, cv2.LINE_AA)

            # ----------------------------------------------------------
            # Draw layer 5: Alert banners
            # ----------------------------------------------------------
            if new_alerts > 0:
                b = frame.copy()
                cv2.rectangle(b, (0, height - 34), (width, height), (0, 0, 140), -1)
                cv2.addWeighted(b, 0.88, frame, 0.12, 0, frame)
                cv2.putText(
                    frame,
                    f"  !! RISK ALERT: {new_alerts} HIGH-RISK PERSON(S) FLAGGED !!",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
            elif unattended_count > 0:
                b = frame.copy()
                cv2.rectangle(b, (0, height - 34), (width, height), (0, 0, 160), -1)
                cv2.addWeighted(b, 0.88, frame, 0.12, 0, frame)
                cv2.putText(
                    frame,
                    f"  WARNING: {unattended_count} UNATTENDED OBJECT(S) DETECTED!",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

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
    print(f"Total alerts fired: {total_alerts}")
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    test_vid = Path("data/videos/test_crowded_scene.mp4")
    out_vid  = Path("data/processed/test_crowded_scene_phase6.mp4")
    process_video(test_vid, out_vid)
