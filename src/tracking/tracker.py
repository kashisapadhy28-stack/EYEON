"""
EYEON - Phase 3: Multi-Object Tracking Module (ByteTrack & BoT-SORT)

Transforms frame-by-frame detections into persistent tracked identities:
- Person #1, Person #2, Person #3 with unique IDs
- Supports both ByteTrack (bytetrack.yaml) and BoT-SORT (botsort.yaml)
- Formats structured track data:
  {
      "track_id": 12,
      "class": "person",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.94,
      "timestamp": 1.42,
      "frame_id": 43,
      "center": (cx, cy),
      "foot_point": (cx, y2)
  }
- Maintains per-ID movement trajectory history with customizable trail lengths
- Renders distinct identity colors and trajectory lines
"""

import sys
import time
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config_loader import load_config, get_project_root, get_path, get_config_value
from src.detection.video_input import VideoStream, FramePacket

# Curated vibrant palette for unique track IDs (BGR)
TRACK_PALETTE = [
    (46, 204, 113),   # Emerald Green
    (52, 152, 219),   # Peter River Blue
    (155, 89, 182),   # Amethyst Purple
    (241, 196, 15),   # Sun Yellow
    (230, 126, 34),   # Carrot Orange
    (231, 76, 60),    # Alizarin Red
    (26, 188, 156),   # Turquoise
    (243, 156, 18),   # Orange Amber
    (52, 73, 94),     # Wet Asphalt Slate
    (142, 68, 173),   # Wisteria Violet
    (0, 168, 255),    # Sky Cyan
    (255, 107, 129),  # Wild Watermelon
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Returns a deterministic, vibrant color for a given track ID."""
    return TRACK_PALETTE[abs(int(track_id)) % len(TRACK_PALETTE)]


class MultiObjectTracker:
    """
    Multi-Object Tracker supporting ByteTrack and BoT-SORT algorithms.
    Maintains persistent track IDs, motion history, and state across frames.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tracker_type: str = "bytetrack",
        track_buffer: Optional[int] = None,
        conf_person: Optional[float] = None,
        conf_object: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        max_history_len: int = 50,
    ):
        """
        Args:
            model_path: Path to YOLO model weights.
            tracker_type: 'bytetrack' or 'botsort'.
            track_buffer: Tracking buffer frames (default from config.yaml: 30).
            conf_person: Confidence threshold for person.
            conf_object: Confidence threshold for objects.
            device: 'cpu' or 0 (GPU).
            imgsz: Inference image size (640).
            max_history_len: Maximum historical positions stored per track.
        """
        self.config = load_config()
        self.project_root = get_project_root()

        # Model resolution
        if model_path is None:
            model_rel = self.config.get("detection", {}).get("model", "models/yolo11n.pt")
            resolved = self.project_root / model_rel
            if not resolved.exists():
                resolved = self.project_root / "yolo11n.pt"
            self.model_path = resolved
        else:
            self.model_path = Path(model_path)
            if not self.model_path.is_absolute():
                self.model_path = self.project_root / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        # Tracker configuration
        norm_type = tracker_type.lower().replace("-", "").replace("_", "")
        if "bot" in norm_type:
            self.tracker_name = "botsort.yaml"
            self.tracker_label = "BoT-SORT"
        else:
            self.tracker_name = "bytetrack.yaml"
            self.tracker_label = "ByteTrack"

        self.track_buffer = track_buffer or self.config.get("tracking", {}).get("track_buffer", 30)

        # Confidence thresholds
        conf_cfg = self.config.get("detection", {}).get("confidence", {})
        self.conf_person = conf_person if conf_person is not None else conf_cfg.get("person", 0.40)
        self.conf_object = conf_object if conf_object is not None else conf_cfg.get("object", 0.25)
        self.min_conf = min(self.conf_person, self.conf_object)

        # Device handling
        raw_device = device or self.config.get("detection", {}).get("device", "auto")
        if raw_device in ("auto", None, ""):
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device

        self.imgsz = imgsz or self.config.get("detection", {}).get("image_size", 640)
        self.max_history_len = max_history_len

        # Target classes
        classes_cfg = self.config.get("detection", {}).get("classes", {})
        self.person_classes = set(c.lower() for c in classes_cfg.get("person", ["person"]))
        raw_objects = classes_cfg.get("objects", ["backpack", "handbag", "suitcase", "cell phone", "laptop", "bottle"])
        self.object_classes = set(c.lower() for c in raw_objects)
        if "cell phone" in self.object_classes:
            self.object_classes.add("cellphone")

        self.monitored_classes = self.person_classes.union(self.object_classes)

        # Initialize model
        self.model = YOLO(str(self.model_path))

        # Filter target class IDs
        self.target_class_ids = []
        self.id_to_name = {}
        for cid, name in self.model.names.items():
            norm = name.lower()
            if norm in self.monitored_classes or norm.replace(" ", "") in self.monitored_classes:
                self.target_class_ids.append(cid)
                self.id_to_name[cid] = norm

        # Movement history: {track_id: deque([(x, y), ...], maxlen=max_history_len)}
        self.track_history: Dict[int, deque] = {}
        self.unique_track_ids_seen = set()

    def reset(self):
        """Clears tracking history and state."""
        self.track_history.clear()
        self.unique_track_ids_seen.clear()

    def track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Tracks objects across frames, assigning persistent IDs.

        Args:
            frame: Video frame (BGR numpy array).
            frame_id: Current frame counter.
            timestamp: Seconds elapsed from stream start.

        Returns:
            List of track dicts:
            [
                {
                    "track_id": 12,
                    "class": "person",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.94,
                    "timestamp": 1.42,
                    "frame_id": 43,
                    "center": (cx, cy),
                    "foot_point": (cx, y2)
                }, ...
            ]
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_name,
            conf=self.min_conf,
            classes=self.target_class_ids,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        tracks = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.float().cpu().tolist()
            coords = boxes.xyxy.int().cpu().tolist()

            for i in range(len(ids)):
                track_id = int(ids[i])
                class_id = int(clss[i])
                confidence = float(confs[i])
                class_name = self.id_to_name.get(class_id, self.model.names.get(class_id, "unknown")).lower()

                # Filter by per-category confidence thresholds
                is_person = class_name in self.person_classes
                threshold = self.conf_person if is_person else self.conf_object
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = coords[i]

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                foot_point = (center_x, y2)

                # Maintain history (using foot_point for person, center for objects)
                pos = foot_point if is_person else (center_x, center_y)
                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.max_history_len)
                self.track_history[track_id].append(pos)
                self.unique_track_ids_seen.add(track_id)

                tracks.append({
                    "track_id": track_id,
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(confidence, 4),
                    "timestamp": round(timestamp, 3),
                    "frame_id": frame_id,
                    "center": (center_x, center_y),
                    "foot_point": foot_point,
                })

        return tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        show_trail: bool = True,
        show_labels: bool = True,
    ) -> np.ndarray:
        """
        Renders sleek bounding boxes, persistent ID labels, and movement trails.
        """
        annotated = frame.copy()
        active_ids = []

        for trk in tracks:
            track_id = trk["track_id"]
            cls_name = trk["class"]
            conf = trk["confidence"]
            x1, y1, x2, y2 = trk["bbox"]
            color = get_track_color(track_id)
            active_ids.append(track_id)

            # 1. Draw movement trail (historical trajectory)
            if show_trail and track_id in self.track_history:
                pts = list(self.track_history[track_id])
                for j in range(1, len(pts)):
                    # Fade trail line thickness & alpha
                    thickness = max(1, int(3 * (j / len(pts))))
                    cv2.line(annotated, pts[j - 1], pts[j], color, thickness, cv2.LINE_AA)
                if pts:
                    cv2.circle(annotated, pts[-1], 4, color, -1)

            # 2. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 3. Label badge: "person #1 0.94"
            if show_labels:
                label = f"{cls_name.capitalize()} #{track_id} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.52
                thickness = 1
                (lw, lh), _ = cv2.getTextSize(label, font, scale, thickness)

                badge_y1 = max(0, y1 - lh - 8)
                badge_y2 = y1
                badge_x2 = min(annotated.shape[1], x1 + lw + 8)

                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
                # Black text for light badge, white text for dark
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - 4),
                    font,
                    scale,
                    (255, 255, 255) if sum(color) < 400 else (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        return annotated


class ByteTracker(MultiObjectTracker):
    """ByteTrack Tracker implementation."""
    def __init__(self, **kwargs):
        super().__init__(tracker_type="bytetrack", **kwargs)


class BoTSORTTracker(MultiObjectTracker):
    """BoT-SORT Tracker implementation."""
    def __init__(self, **kwargs):
        super().__init__(tracker_type="botsort", **kwargs)


def process_video_tracking(
    source: Optional[Union[str, int, Path]] = None,
    tracker_type: str = "bytetrack",
    output_path: Optional[Union[str, Path]] = None,
    resize: Optional[Tuple[int, int]] = None,
    display: bool = True,
    save_output: bool = False,
    max_frames: Optional[int] = None,
    conf_person: Optional[float] = None,
    conf_object: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Processes video stream through the Multi-Object Tracking pipeline.

    Returns:
        (total_frames_processed, total_unique_tracks)
    """
    stream = VideoStream(
        source=source,
        resize=resize,
        save_output=save_output,
        output_path=output_path,
    )

    tracker = MultiObjectTracker(
        tracker_type=tracker_type,
        conf_person=conf_person,
        conf_object=conf_object,
    )

    print("=" * 65)
    print(f" EYEON Phase 3 — {tracker.tracker_label} Tracking Pipeline")
    print(f" Model:            {tracker.model_path.name}")
    print(f" Tracker:          {tracker.tracker_name}")
    print(f" Source:           {stream.source}")
    print(f" Resolution:       {stream.output_width}x{stream.output_height} @ {stream.source_fps:.1f} FPS")
    print(f" Confidence:       Person >= {tracker.conf_person:.2f}, Object >= {tracker.conf_object:.2f}")
    if stream.save_output:
        print(f" Saving Output:    {stream.output_path}")
    print("=" * 65)
    print(" Controls: [Q/ESC] Quit | [SPACE] Pause | [S] Screenshot")

    window_name = f"EYEON — {tracker.tracker_label}"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(stream.output_width, 1024), min(stream.output_height, 576))

    paused = False
    processed_count = 0

    try:
        while True:
            if not paused:
                ret, packet = stream.read_frame()
                if not ret or packet is None:
                    print("\nStream finished / end of video reached.")
                    break

                processed_count += 1

                # Multi-Object Tracking
                tracks = tracker.track(
                    packet.frame,
                    frame_id=packet.frame_id,
                    timestamp=packet.timestamp,
                )

                # Draw tracks, boxes, IDs, trails
                annotated = tracker.draw_tracks(packet.frame, tracks)

                # Render Top HUD
                avg_fps = stream.get_average_fps()
                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

                active_count = len(tracks)
                cumulative_tracks = len(tracker.unique_track_ids_seen)
                hud_text = (
                    f"EYEON {tracker.tracker_label.upper()} | "
                    f"Frame: {packet.frame_id}/{stream.total_frames} | "
                    f"FPS: {packet.fps:.1f} (Avg: {avg_fps:.1f}) | "
                    f"Active: {active_count} | Total IDs: {cumulative_tracks}"
                )
                cv2.putText(
                    annotated,
                    hud_text,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )

                if stream.save_output:
                    stream.write_frame(annotated)

                if display:
                    cv2.imshow(window_name, annotated)

                if max_frames and processed_count >= max_frames:
                    print(f"\nReached max frame limit: {max_frames}")
                    break

            if display:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nTracking stopped by user.")
                    break
                elif key in (ord(" "), ord("p"), ord("P")):
                    paused = not paused
                    print(f"\n[STREAM {'PAUSED' if paused else 'RESUMED'}]")
                elif key in (ord("s"), ord("S")):
                    snap_dir = get_path("database.snapshots")
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap_file = snap_dir / f"tracking_snapshot_{packet.frame_id}.jpg"
                    cv2.imwrite(str(snap_file), annotated)
                    print(f"\nSaved tracking snapshot to: {snap_file}")

    finally:
        stream.release()
        if display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print(f" {tracker.tracker_label} Tracking Completed Successfully!")
    print(f" Total Frames Processed:    {processed_count}")
    print(f" Total Unique Tracks Seen:  {len(tracker.unique_track_ids_seen)}")
    print(f" Average Tracking FPS:      {stream.get_average_fps():.2f}")
    if stream.save_output and stream.output_path:
        print(f" Output Video Saved to:     {stream.output_path}")
    print("=" * 65)
    return processed_count, len(tracker.unique_track_ids_seen)


def main():
    parser = argparse.ArgumentParser(description="EYEON Phase 3 — Multi-Object Tracking Module")
    parser.add_argument("--source", type=str, default=None, help="Video file path or camera index")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracker algorithm")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None, help="Resize dimensions")
    parser.add_argument("--conf-person", type=float, default=None, help="Confidence threshold for person")
    parser.add_argument("--conf-object", type=float, default=None, help="Confidence threshold for objects")
    parser.add_argument("--save", action="store_true", help="Save annotated tracking video")
    parser.add_argument("--no-display", action="store_true", help="Headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    args = parser.parse_args()
    resize_tuple = tuple(args.resize) if args.resize else None

    process_video_tracking(
        source=args.source,
        tracker_type=args.tracker,
        output_path=args.output,
        resize=resize_tuple,
        display=not args.no_display,
        save_output=args.save,
        max_frames=args.max_frames,
        conf_person=args.conf_person,
        conf_object=args.conf_object,
    )


if __name__ == "__main__":
    main()