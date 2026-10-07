"""
EYEON - Phase 1: Video & Camera Input Module

Provides reliable, high-performance video streaming from files or cameras:
- Reads frames continuously and handles stream end/errors gracefully
- Calculates instantaneous and moving-average FPS
- Resizes frames while preserving aspect ratio or fitting specific resolutions
- Displays frames with a clean, informative HUD
- Optionally saves annotated/processed video to disk
- Seamlessly integrates with config.yaml
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path for direct execution
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass
from typing import Optional, Tuple, Generator, Union
import cv2
import numpy as np

from src.utils.config_loader import load_config, get_project_root, get_path


@dataclass
class FramePacket:
    """Structured data packet for each video frame."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    fps: float
    is_last: bool
    original_size: Tuple[int, int]  # (width, height)


class VideoStream:
    """
    Robust video reader supporting video files, video directories, and live cameras.
    Can be used as an iterator or frame-by-frame reader.
    """

    def __init__(
        self,
        source: Optional[Union[str, int, Path]] = None,
        resize: Optional[Tuple[int, int]] = None,
        save_output: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        target_fps: Optional[float] = None,
    ):
        """
        Initialize the video stream.

        Args:
            source: Path to video file, camera index (0, 1), or None (reads from config).
            resize: Optional (width, height) to resize frames.
            save_output: Whether to save output video.
            output_path: Path to save processed video.
            target_fps: Override output video FPS (if None, matches source FPS).
        """
        self.config = load_config()
        self.project_root = get_project_root()

        # Determine source
        if source is None:
            config_source = self.config.get("video", {}).get("source", "data/videos")
            source_path = self.project_root / config_source
            if source_path.is_dir():
                # Pick first available video in folder
                videos = list(source_path.glob("*.mp4")) + list(source_path.glob("*.avi"))
                if not videos:
                    raise FileNotFoundError(f"No video files found in directory: {source_path}")
                self.source = str(videos[0])
            else:
                self.source = str(source_path)
        else:
            self.source = source

        self.resize = resize
        self.save_output = save_output
        self.target_fps = target_fps

        # Determine output path if saving
        if self.save_output:
            if output_path is None:
                out_dir = get_path("video.output")
                out_dir.mkdir(parents=True, exist_ok=True)
                source_name = Path(str(self.source)).stem if isinstance(self.source, (str, Path)) else "cam"
                self.output_path = out_dir / f"{source_name}_processed.mp4"
            else:
                self.output_path = Path(output_path)
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path = None

        # State tracking
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.start_time = 0.0
        self.prev_frame_time = 0.0
        self.current_fps = 0.0

        # Video metadata
        self.original_width = 0
        self.original_height = 0
        self.output_width = 0
        self.output_height = 0
        self.source_fps = 30.0
        self.total_frames = -1

        self._open()

    def _open(self):
        """Opens video capture and initializes writer if requested."""
        # Convert integer strings to int for camera indexes
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        elif isinstance(src, str):
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = self.project_root / src_path
            src = str(src_path)

        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps_val if fps_val > 0 else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine target output size with orientation preservation
        if self.resize:
            rw, rh = self.resize
            if self.original_height > self.original_width and rw > rh:
                # Portrait video with landscape target -> swap to portrait
                self.output_width, self.output_height = rh, rw
            elif self.original_width > self.original_height and rh > rw:
                # Landscape video with portrait target -> swap to landscape
                self.output_width, self.output_height = rh, rw
            else:
                self.output_width, self.output_height = rw, rh
        else:
            self.output_width, self.output_height = self.original_width, self.original_height

        # Initialize VideoWriter if requested
        if self.save_output and self.output_path:
            fps_out = self.target_fps or self.source_fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                fps_out,
                (self.output_width, self.output_height),
            )

        self.start_time = time.time()
        self.prev_frame_time = self.start_time

    def read_frame(self) -> Tuple[bool, Optional[FramePacket]]:
        """
        Reads a single frame, resizes if requested, and calculates FPS.

        Returns:
            (ret, FramePacket or None)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        self.frame_count += 1
        now = time.time()
        delta = now - self.prev_frame_time
        self.current_fps = (1.0 / delta) if delta > 0 else self.source_fps
        self.prev_frame_time = now

        # Resize if required
        if self.resize and (self.original_width != self.output_width or self.original_height != self.output_height):
            frame = cv2.resize(frame, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)

        is_last = (self.frame_count >= self.total_frames) if self.total_frames > 0 else False

        packet = FramePacket(
            frame=frame,
            frame_id=self.frame_count,
            timestamp=now - self.start_time,
            fps=self.current_fps,
            is_last=is_last,
            original_size=(self.original_width, self.original_height),
        )

        return True, packet

    def write_frame(self, frame: np.ndarray):
        """Writes frame to output video file if saving is enabled."""
        if self.writer:
            if (frame.shape[1], frame.shape[0]) != (self.output_width, self.output_height):
                frame = cv2.resize(frame, (self.output_width, self.output_height))
            self.writer.write(frame)

    def frames(self) -> Generator[FramePacket, None, None]:
        """Generator that yields FramePacket until stream ends."""
        try:
            while True:
                ret, packet = self.read_frame()
                if not ret or packet is None:
                    break
                yield packet
        finally:
            self.release()

    def __iter__(self):
        return self.frames()

    def get_average_fps(self) -> float:
        """Returns average processing FPS across the entire session."""
        elapsed = time.time() - self.start_time
        return (self.frame_count / elapsed) if elapsed > 0 else 0.0

    def release(self):
        """Releases video capture and video writer resources cleanly."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        if self.writer:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.release()


def draw_hud(
    frame: np.ndarray,
    frame_id: int,
    total_frames: int,
    fps: float,
    avg_fps: float,
    res: Tuple[int, int],
) -> np.ndarray:
    """Renders a sleek, modern HUD onto the frame."""
    hud_frame = frame.copy()
    h, w = hud_frame.shape[:2]

    # Semi-transparent overlay top banner
    overlay = hud_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, hud_frame, 0.3, 0, hud_frame)

    progress_str = f"Frame: {frame_id}/{total_frames}" if total_frames > 0 else f"Frame: {frame_id}"
    fps_str = f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})"
    res_str = f"{res[0]}x{res[1]}"

    text = f"EYEON LIVE | {progress_str} | {fps_str} | {res_str}"
    cv2.putText(
        hud_frame,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return hud_frame


def process_video(
    source: Optional[Union[str, int, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    resize: Optional[Tuple[int, int]] = None,
    display: bool = True,
    save_output: bool = False,
    max_frames: Optional[int] = None,
) -> int:
    """
    Processes video stream, displaying frames and calculating FPS.

    Args:
        source: Video source (file path, webcam index 0, or None for config default)
        output_path: Target path for processed video if save_output is True
        resize: Target resolution (width, height)
        display: Whether to show OpenCV GUI window
        save_output: Whether to record video to output_path
        max_frames: Stop after max_frames (useful for headless testing)

    Returns:
        Total frames processed
    """
    stream = VideoStream(
        source=source,
        resize=resize,
        save_output=save_output,
        output_path=output_path,
    )

    print("=" * 60)
    print(f" EYEON Video Input - Stream Started")
    print(f" Source:            {stream.source}")
    print(f" Source Resolution: {stream.original_width}x{stream.original_height} @ {stream.source_fps:.1f} FPS")
    print(f" Stream Resolution: {stream.output_width}x{stream.output_height}")
    print(f" Total Frames:      {stream.total_frames if stream.total_frames > 0 else 'Live Stream'}")
    if stream.save_output:
        print(f" Saving to:         {stream.output_path}")
    print("=" * 60)
    print(" Controls: [Q/ESC] Quit | [SPACE] Pause | [S] Screenshot")

    window_name = "EYEON - Phase 1 Video Input"
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
                hud_frame = draw_hud(
                    packet.frame,
                    packet.frame_id,
                    stream.total_frames,
                    packet.fps,
                    stream.get_average_fps(),
                    (stream.output_width, stream.output_height),
                )

                if stream.save_output:
                    stream.write_frame(hud_frame)

                if display:
                    cv2.imshow(window_name, hud_frame)

                if max_frames and processed_count >= max_frames:
                    print(f"\nReached max frame limit: {max_frames}")
                    break

            if display:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("\nPlayback stopped by user.")
                    break
                elif key in (ord(" "), ord("p"), ord("P")):  # Space / P
                    paused = not paused
                    state = "PAUSED" if paused else "RESUMED"
                    print(f"\n[STREAM {state}]")
                elif key in (ord("s"), ord("S")):  # S
                    snap_dir = get_path("database.snapshots")
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snap_file = snap_dir / f"snapshot_frame_{packet.frame_id}.jpg"
                    cv2.imwrite(str(snap_file), packet.frame)
                    print(f"\nSaved snapshot to: {snap_file}")

    finally:
        stream.release()
        if display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f" Stream Completed Successfully!")
    print(f" Total Frames Processed: {processed_count}")
    print(f" Overall Processing FPS: {stream.get_average_fps():.2f}")
    if stream.save_output and stream.output_path:
        print(f" Video Saved to:         {stream.output_path}")
    print("=" * 60)
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="EYEON Video / Camera Input Module")
    parser.add_argument("--source", type=str, default=None, help="Video file path or camera index (default: from config.yaml)")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: data/processed/<name>_processed.mp4)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None, help="Resize dimensions (e.g. 640 360)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (don't open OpenCV window)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (useful for test scripts)")

    args = parser.parse_args()
    resize_tuple = tuple(args.resize) if args.resize else None

    process_video(
        source=args.source,
        output_path=args.output,
        resize=resize_tuple,
        display=not args.no_display,
        save_output=args.save,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()