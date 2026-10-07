"""
EYEON - Phase 2: Person & Object Detection Module

Performs pure YOLO detection on video frames without tracking logic:
- Detects persons and designated everyday objects (backpack, handbag, suitcase, cell phone, laptop, bottle)
- Applies distinct confidence thresholds for persons vs objects (configured via config.yaml)
- Formats structured detection output dictionaries:
  {
      "class": "person",
      "confidence": 0.94,
      "bbox": [x1, y1, x2, y2],
      "center": (cx, cy),
      "foot_point": (cx, y2)
  }
- Renders bounding boxes and detection badges
- Seamlessly consumes frames from Phase 1 VideoStream
"""

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
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


# Color palette for detection bounding boxes (BGR)
COLOR_PERSON = (46, 204, 113)    # Emerald Green
COLOR_BAG = (241, 196, 15)       # Amber Gold (backpack, handbag, suitcase)
COLOR_ELECTRONIC = (230, 126, 34) # Orange (laptop, cell phone)
COLOR_OBJECT = (52, 152, 219)    # Sky Blue (bottle, default)

CLASS_COLORS = {
    "person": COLOR_PERSON,
    "backpack": COLOR_BAG,
    "handbag": COLOR_BAG,
    "suitcase": COLOR_BAG,
    "laptop": COLOR_ELECTRONIC,
    "cell phone": COLOR_ELECTRONIC,
    "cellphone": COLOR_ELECTRONIC,
    "bottle": COLOR_OBJECT,
}


@dataclass
class Detection:
    """Structured detection object."""
    cls_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[int, int]
    foot_point: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.cls_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "center": self.center,
            "foot_point": self.foot_point,
        }


class ObjectDetector:
    """
    COCO-pretrained YOLO Detector for persons and target everyday objects.
    Enforces per-class confidence thresholds and extracts structured detections.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_person: Optional[float] = None,
        conf_object: Optional[float] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ):
        self.config = load_config()
        self.project_root = get_project_root()

        # Model path resolution
        if model_path is None:
            model_rel = self.config.get("detection", {}).get("model", "models/yolo11n.pt")
            resolved_model = self.project_root / model_rel
            if not resolved_model.exists():
                # Fallback to root model file if present
                resolved_model = self.project_root / "yolo11n.pt"
            self.model_path = resolved_model
        else:
            self.model_path = Path(model_path)
            if not self.model_path.is_absolute():
                self.model_path = self.project_root / self.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Detection model not found at: {self.model_path}")

        # Detection settings
        conf_cfg = self.config.get("detection", {}).get("confidence", {})
        self.conf_person = conf_person if conf_person is not None else conf_cfg.get("person", 0.40)
        self.conf_object = conf_object if conf_object is not None else conf_cfg.get("object", 0.25)
        self.min_conf = min(self.conf_person, self.conf_object)

        raw_device = device or self.config.get("detection", {}).get("device", "auto")
        if raw_device in ("auto", None, ""):
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = raw_device

        self.imgsz = imgsz or self.config.get("detection", {}).get("image_size", 640)

        # Classes to monitor
        classes_cfg = self.config.get("detection", {}).get("classes", {})
        self.person_classes = set(c.lower() for c in classes_cfg.get("person", ["person"]))
        raw_obj_classes = classes_cfg.get("objects", ["backpack", "handbag", "suitcase", "cell phone", "laptop", "bottle"])
        self.object_classes = set()
        for c in raw_obj_classes:
            self.object_classes.add(c.lower())
            if c.lower() == "cell phone":
                self.object_classes.add("cellphone")

        self.monitored_classes = self.person_classes.union(self.object_classes)

        # Initialize YOLO model
        self.model = YOLO(str(self.model_path))

        # Map target class names to model class IDs for efficient YOLO filtering
        self.target_class_ids = []
        self.id_to_name = {}
        for cls_id, name in self.model.names.items():
            norm_name = name.lower()
            if norm_name in self.monitored_classes or norm_name.replace(" ", "") in self.monitored_classes:
                self.target_class_ids.append(cls_id)
                self.id_to_name[cls_id] = norm_name

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO object detection on a single frame.

        Args:
            frame: Video frame (BGR numpy array)

        Returns:
            List of detection dicts:
            [
                {
                    "class": "person",
                    "confidence": 0.94,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "foot_point": (cx, y2)
                }, ...
            ]
        """
        results = self.model.predict(
            frame,
            conf=self.min_conf,
            classes=self.target_class_ids,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = self.id_to_name.get(class_id, self.model.names.get(class_id, "unknown")).lower()

                # Apply per-category confidence thresholds
                is_person = class_name in self.person_classes
                threshold = self.conf_person if is_person else self.conf_object
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                # Coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                foot_x = center_x
                foot_y = y2

                det = Detection(
                    cls_name=class_name,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    center=(center_x, center_y),
                    foot_point=(foot_x, foot_y),
                )
                detections.append(det.to_dict())

        return detections

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        show_summary: bool = True,
    ) -> np.ndarray:
        """
        Renders sleek bounding boxes and labels onto the frame.

        Args:
            frame: Video frame (BGR numpy array)
            detections: List of detection dictionaries
            show_labels: Whether to show class name and confidence badges
            show_summary: Whether to show detection count in bottom corner

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        person_count = 0
        object_count = 0
        obj_breakdown = {}

        for det in detections:
            cls_name = det["class"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            if cls_name == "person":
                person_count += 1
            else:
                object_count += 1
                obj_breakdown[cls_name] = obj_breakdown.get(cls_name, 0) + 1

            # Bounding box with corner accents
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                label = f"{cls_name} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, scale, thickness)

                # Label background badge
                badge_y1 = max(0, y1 - label_h - 8)
                badge_y2 = y1
                badge_x2 = min(annotated.shape[1], x1 + label_w + 8)

                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, y1 - 4),
                    font,
                    scale,
                    (0, 0, 0),  # Black text on bright background
                    thickness,
                    cv2.LINE_AA,
                )

                # Mark foot-point or center
                fx, fy = det.get("foot_point", ((x1 + x2) // 2, y2))
                cv2.circle(annotated, (fx, fy), 4, color, -1)

        # Bottom-left summary badge
        if show_summary:
            summary_text = f"Detections: {person_count} Person(s) | {object_count} Object(s)"
            if obj_breakdown:
                items_str = ", ".join(f"{k}:{v}" for k, v in obj_breakdown.items())
                summary_text += f" ({items_str})"

            h = annotated.shape[0]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, h - 35), (min(annotated.shape[1] - 10, 15 + len(summary_text) * 9), h - 10), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.putText(
                annotated,
                summary_text,
                (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated


def process_video_detection(
    source: Optional[Union[str, int, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    resize: Optional[Tuple[int, int]] = None,
    display: bool = True,
    save_output: bool = False,
    max_frames: Optional[int] = None,
    conf_person: Optional[float] = None,
    conf_object: Optional[float] = None,
) -> int:
    """
    Processes video stream through the YOLO detector pipeline.
    """
    stream = VideoStream(
        source=source,
        resize=resize,
        save_output=save_output,
        output_path=output_path,
    )

    detector = ObjectDetector(
        conf_person=conf_person,
        conf_object=conf_object,
    )

    print("=" * 65)
    print(" EYEON Phase 2 - Person & Object Detection Pipeline")
    print(f" Model:           {detector.model_path.name}")
    print(f" Source:          {stream.source}")
    print(f" Resolution:      {stream.output_width}x{stream.output_height} @ {stream.source_fps:.1f} FPS")
    print(f" Person Conf:     >= {detector.conf_person:.2f}")
    print(f" Object Conf:     >= {detector.conf_object:.2f}")
    print(f" Target Classes:  {', '.join(sorted(detector.monitored_classes))}")
    if stream.save_output:
        print(f" Saving to:       {stream.output_path}")
    print("=" * 65)
    print(" Controls: [Q/ESC] Quit | [SPACE] Pause | [S] Screenshot")

    window_name = "EYEON - Phase 2 YOLO Detection"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(stream.output_width, 1024), min(stream.output_height, 576))

    paused = False
    processed_count = 0
    total_detections_logged = 0

    try:
        while True:
            if not paused:
                ret, packet = stream.read_frame()
                if not ret or packet is None:
                    print("\nStream finished / end of video reached.")
                    break

                processed_count += 1

                # Phase 2 Detection
                detections = detector.detect(packet.frame)
                total_detections_logged += len(detections)

                # Draw detections and HUD
                annotated = detector.draw_detections(packet.frame, detections)

                # Add top HUD
                avg_fps = stream.get_average_fps()
                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

                hud_text = f"EYEON DETECT | Frame: {packet.frame_id}/{stream.total_frames} | FPS: {packet.fps:.1f} (Avg: {avg_fps:.1f}) | Targets: {len(detections)}"
                cv2.putText(
                    annotated,
                    hud_text,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
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
                    print("\nDetection stopped by user.")
                    break
                elif key in (ord(" "), ord("p"), ord("P")):
                    paused = not paused
                    print(f"\n[STREAM {'PAUSED' if paused else 'RESUMED'}]")
                elif key in (ord("s"), ord("S")):
                    snap_dir = get_path("database.snapshots")
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap_file = snap_dir / f"detection_snapshot_{packet.frame_id}.jpg"
                    cv2.imwrite(str(snap_file), annotated)
                    print(f"\nSaved detection snapshot to: {snap_file}")

    finally:
        stream.release()
        if display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print(" Detection Pipeline Completed Successfully!")
    print(f" Total Frames Processed:    {processed_count}")
    print(f" Total Objects Detected:    {total_detections_logged}")
    print(f" Average Detection FPS:     {stream.get_average_fps():.2f}")
    if stream.save_output and stream.output_path:
        print(f" Output Video Saved to:     {stream.output_path}")
    print("=" * 65)
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="EYEON Phase 2 - Person & Object Detection Module")
    parser.add_argument("--source", type=str, default=None, help="Video file path or camera index")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None, help="Resize dimensions")
    parser.add_argument("--conf-person", type=float, default=None, help="Confidence threshold for person")
    parser.add_argument("--conf-object", type=float, default=None, help="Confidence threshold for objects")
    parser.add_argument("--save", action="store_true", help="Save annotated video")
    parser.add_argument("--no-display", action="store_true", help="Headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    args = parser.parse_args()
    resize_tuple = tuple(args.resize) if args.resize else None

    process_video_detection(
        source=args.source,
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