"""
EYEON - Phase 4: Person-Object Association Video Pipeline

Processes video streams through the complete Phase 4 Intelligence Engine:
- Multi-Object Tracking with Anti-Flicker & EMA Smoothing
- Spatial & Motion Association
- Temporal Ownership Voting (15-frame rolling window)
- Object Lifecycle States (CARRIED, DETACHED, UNATTENDED)
- Visual alert indicators for unattended/abandoned luggage
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.association.association import (
    PersonObjectAssociator,
    TemporalOwnershipTracker,
    ObjectStateManager,
    PersonObjectIntelligence,
    ObjectState,
)
from src.utils.config_loader import load_config, get_path


class AssociationTracker:
    """
    Combines YOLO BoT-SORT tracking with PersonObjectIntelligence.
    """

    def __init__(
        self,
        model_path: str = "models/yolo11n.pt",
        grace_frames: int = 8,
        smooth_alpha: float = 0.70,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            model_file = _ROOT / model_path
            if not model_file.exists():
                model_file = _ROOT / "yolo11n.pt"

        self.model = YOLO(str(model_file))
        self.intelligence = PersonObjectIntelligence()

        self.grace_frames = grace_frames
        self.smooth_alpha = smooth_alpha

        # Filter target classes
        self.target_names = {
            "person", "backpack", "handbag", "suitcase",
            "cell phone", "cellphone", "laptop", "bottle"
        }
        self.target_class_ids = []
        for cid, name in self.model.names.items():
            if name.lower() in self.target_names or name.lower().replace(" ", "") in self.target_names:
                self.target_class_ids.append(cid)

        # Track caches
        self.track_cache: Dict[int, Dict[str, Any]] = {}
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.prev_person_centers: Dict[int, Tuple[int, int]] = {}

    def track(self, frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=0.20,
            classes=self.target_class_ids,
            verbose=False,
        )

        current_detected_tracks = {}
        current_person_centers = {}

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                class_name = self.model.names[class_id].lower()

                # Separate person threshold
                if class_name == "person" and confidence < 0.35:
                    continue

                # EMA coordinate smoothing
                if track_id in self.track_cache:
                    prev_box = self.track_cache[track_id]["bbox"]
                    x1 = int(self.smooth_alpha * x1 + (1.0 - self.smooth_alpha) * prev_box[0])
                    y1 = int(self.smooth_alpha * y1 + (1.0 - self.smooth_alpha) * prev_box[1])
                    x2 = int(self.smooth_alpha * x2 + (1.0 - self.smooth_alpha) * prev_box[2])
                    y2 = int(self.smooth_alpha * y2 + (1.0 - self.smooth_alpha) * prev_box[3])

                current_detected_tracks[track_id] = {
                    "track_id": track_id,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "missing": 0,
                    "is_coasting": False,
                }

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)

                if class_name == "person":
                    current_person_centers[track_id] = center

        # Coasting and anchor movement
        for tid, cached in list(self.track_cache.items()):
            if tid not in current_detected_tracks:
                cached["missing"] += 1
                if cached["missing"] <= self.grace_frames:
                    assoc_pid = cached.get("associated_person")
                    if (
                        assoc_pid in current_person_centers
                        and assoc_pid in self.prev_person_centers
                    ):
                        curr_pc = current_person_centers[assoc_pid]
                        prev_pc = self.prev_person_centers[assoc_pid]
                        dx = curr_pc[0] - prev_pc[0]
                        dy = curr_pc[1] - prev_pc[1]
                        bx1, by1, bx2, by2 = cached["bbox"]
                        cached["bbox"] = [bx1 + dx, by1 + dy, bx2 + dx, by2 + dy]

                    cached["is_coasting"] = True
                    current_detected_tracks[tid] = cached
                else:
                    del self.track_cache[tid]
                    self.track_history.pop(tid, None)

        # Update cache
        for tid, trk in current_detected_tracks.items():
            if tid not in self.track_cache:
                self.track_cache[tid] = trk
            else:
                self.track_cache[tid]["bbox"] = trk["bbox"]
                self.track_cache[tid]["confidence"] = trk["confidence"]
                self.track_cache[tid]["missing"] = trk["missing"]
                self.track_cache[tid]["is_coasting"] = trk["is_coasting"]

        self.prev_person_centers = current_person_centers
        return list(current_detected_tracks.values())

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """
        Executes tracking + Phase 4 Intelligence.

        Returns:
            (tracks, associations, object_states)
        """
        tracks = self.track(frame, timestamp=timestamp)
        associations, object_states = self.intelligence.process(
            tracks=tracks,
            track_histories=self.track_history,
            timestamp=timestamp,
        )

        # Record associations on cached tracks for anchoring
        for assoc in associations:
            oid = assoc["object_track_id"]
            pid = assoc["person_track_id"]
            if oid in self.track_cache:
                self.track_cache[oid]["associated_person"] = pid

        return tracks, associations, object_states



# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _draw_label(
    frame: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.44,
    thickness: int = 1,
    padding: int = 4,
    bg_alpha: float = 0.65,
) -> None:
    """
    Draw text with a semi-transparent filled pill behind it for readability.

    Args:
        anchor: Top-left corner (x, y) of the label pill.
        bg_alpha: Opacity of the background rectangle (0=invisible, 1=opaque).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = anchor
    # Clamp so the pill never goes off-screen
    x = max(x, 0)
    y = max(y, th + padding * 2)

    rx1, ry1 = x, y - th - padding
    rx2, ry2 = x + tw + padding * 2, y + baseline + padding

    # Semi-transparent background
    sub = frame[ry1:ry2, rx1:rx2]
    if sub.size > 0:
        bg = np.zeros_like(sub)
        bg[:] = (15, 15, 15)
        cv2.addWeighted(bg, bg_alpha, sub, 1.0 - bg_alpha, 0, sub)
        frame[ry1:ry2, rx1:rx2] = sub

    cv2.putText(
        frame, text,
        (x + padding, y),
        font, font_scale, color, thickness, cv2.LINE_AA,
    )


def process_video(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    display: bool = True,
    max_frames: Optional[int] = None,
):
    input_path = Path(input_path)
    if not input_path.exists():
        input_path = _ROOT / input_path

    if not input_path.exists():
        print(f"ERROR: Video not found at {input_path}")
        return

    tracker = AssociationTracker()
    cap = cv2.VideoCapture(str(input_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preserve orientation
    if orig_h > orig_w:
        width, height = 360, 640
    else:
        width, height = 640, 360

    out = None
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    # Safe ASCII window name (em-dash causes encoding issues on some Windows builds)
    window_name = "EYEON - Phase 4 Association & Ownership Intelligence"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(width, 1024), min(height, 576))

    frame_count = 0
    start_time = time.time()

    print("=" * 65)
    print(" EYEON Phase 4 - Person-Object Association Pipeline")
    print(f" Source:           {input_path.name}")
    print(f" Resolution:       {width}x{height} @ {fps:.1f} FPS")
    print(" Intelligence:     Spatial + Motion + Temporal Voting + State Machine")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            cur_timestamp = frame_count / fps
            frame = cv2.resize(frame, (width, height))

            tracks, associations, object_states = tracker.process_frame(
                frame,
                timestamp=cur_timestamp,
            )

            # ----------------------------------------------------------------
            # 1. Person boxes  (green)
            # ----------------------------------------------------------------
            for track in tracks:
                if track["class"] == "person":
                    x1, y1, x2, y2 = track["bbox"]
                    pid = track["track_id"]
                    color_p = (46, 204, 113)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_p, 2)
                    _draw_label(
                        frame,
                        f"Person #{pid}",
                        (x1, max(y1 - 4, 24)),
                        color_p,
                        font_scale=0.46,
                    )

            # ----------------------------------------------------------------
            # 2. Object boxes with state-aware color + pill label
            # ----------------------------------------------------------------
            unattended_count = 0
            for track in tracks:
                if track["class"] == "person":
                    continue

                oid = track["track_id"]
                ocls = track["class"]
                x1, y1, x2, y2 = track["bbox"]
                state_rec = object_states.get(oid, {})
                state     = state_rec.get("state", ObjectState.UNKNOWN)
                owner     = state_rec.get("confirmed_owner", None)
                unattended_sec = state_rec.get("unattended_seconds", 0.0)

                if state == ObjectState.CARRIED:
                    color    = (0, 220, 255)          # Cyan
                    line1    = f"{ocls.capitalize()} #{oid}"
                    line2    = f"CARRIED  P#{owner}"
                elif state == ObjectState.DETACHED:
                    color    = (0, 165, 255)          # Amber
                    line1    = f"{ocls.capitalize()} #{oid}"
                    line2    = f"DETACHED  P#{owner}  {unattended_sec:.0f}s"
                elif state == ObjectState.UNATTENDED:
                    color    = (0, 60, 255)           # Red
                    line1    = f"{ocls.capitalize()} #{oid}"
                    line2    = f"!! UNATTENDED  {unattended_sec:.0f}s"
                    unattended_count += 1
                else:
                    color    = (180, 180, 180)        # Gray
                    line1    = f"{ocls.capitalize()} #{oid}"
                    line2    = "DETECTED"

                thickness = 3 if state == ObjectState.UNATTENDED else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Two-line pill label stacked above the box
                line_h = 18  # approximate pixel height per line
                label_y = max(y1 - 4, line_h * 2 + 8)
                _draw_label(frame, line1, (x1, label_y - line_h), color, font_scale=0.42)
                _draw_label(frame, line2, (x1, label_y), color, font_scale=0.40)

            # ----------------------------------------------------------------
            # 3. Association lines  (yellow, thin)
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # 4. Top HUD bar
            # ----------------------------------------------------------------
            avg_fps = frame_count / (time.time() - start_time + 1e-6)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 32), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            hud_text = (
                f"EYEON  |  Frame {frame_count:04d}"
                f"  |  {avg_fps:.1f} FPS"
                f"  |  Links: {len(associations)}"
            )
            cv2.putText(
                frame, hud_text,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (0, 240, 190), 1, cv2.LINE_AA,
            )

            # ----------------------------------------------------------------
            # 5. Unattended alert banner  (bottom)
            # ----------------------------------------------------------------
            if unattended_count > 0:
                b_overlay = frame.copy()
                cv2.rectangle(b_overlay, (0, height - 34), (width, height), (0, 0, 160), -1)
                cv2.addWeighted(b_overlay, 0.88, frame, 0.12, 0, frame)
                alert_text = f"  WARNING: {unattended_count} UNATTENDED OBJECT(S) DETECTED!"
                cv2.putText(
                    frame, alert_text,
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
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
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    test_vid = Path("data/videos/test_crowded_scene.mp4")
    out_vid  = Path("data/processed/test_crowded_scene_phase4.mp4")
    process_video(test_vid, out_vid)