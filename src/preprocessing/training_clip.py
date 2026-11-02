"""Lightweight preprocessing used to export training-ready review clips.

This module intentionally has no web-application imports or side effects.  The
spatial transform matches ``src/web/app.py``: resize both dimensions by 0.5
with ``cv2.INTER_LINEAR`` and then take a fixed 224x224 center crop.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def training_center_crop(frame, target_size: int = 224):
    """Half-scale a BGR frame and return its fixed-size center crop."""
    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * 0.5)))
    scaled_height = max(1, int(round(height * 0.5)))
    if scaled_width < target_size or scaled_height < target_size:
        return None

    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )
    start_x = (scaled_width - target_size) // 2
    start_y = (scaled_height - target_size) // 2
    return resized[start_y : start_y + target_size, start_x : start_x + target_size]


def _transcode_video(raw_path: Path, out_path: Path) -> tuple[bool, str | None]:
    """Use the same browser-compatible transcode as the review exporter."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            raw_path.unlink(missing_ok=True)
            return True, None
        error = result.stderr.strip() or "ffmpeg failed without an error message"
    else:
        error = "ffmpeg was not found; kept the OpenCV mp4v output"

    if out_path.exists():
        out_path.unlink()
    raw_path.replace(out_path)
    return False, error


def write_segment_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    *,
    target_size: int = 224,
    target_fps: float = 25.0,
) -> tuple[bool, str | None, int]:
    """Export one source-time segment with the training crop and FPS cadence."""
    if out_path.exists() or out_path.with_suffix(".raw.mp4").exists():
        return False, f"Refusing to overwrite an existing output: {out_path}", 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"Cannot open video: {video_path}", 0

    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))
    output_frame_count = max(
        1,
        int((max(0.0, float(end_sec) - float(start_sec))) * target_fps),
    )
    output_indices = np.rint(
        start_frame + (np.arange(output_frame_count, dtype=np.float64) * fps / target_fps)
    ).astype(np.int64)
    output_indices = np.clip(output_indices, start_frame, end_frame - 1)
    output_counts = {}
    for index in output_indices.tolist():
        output_counts[int(index)] = output_counts.get(int(index), 0) + 1

    scaled_width = max(1, int(round(width * 0.5)))
    scaled_height = max(1, int(round(height * 0.5)))
    if scaled_width < target_size or scaled_height < target_size:
        cap.release()
        return (
            False,
            f"Scaled frame size {scaled_width}x{scaled_height} is smaller than "
            f"{target_size}x{target_size}.",
            0,
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (target_size, target_size),
    )
    if not writer.isOpened():
        cap.release()
        return False, f"Cannot create writer for {out_path}", 0

    wrote_frames = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        repeat_count = output_counts.get(frame_idx, 0)
        if repeat_count <= 0:
            frame_idx += 1
            continue
        cropped = training_center_crop(frame, target_size=target_size)
        if cropped is None:
            writer.release()
            cap.release()
            raw_path.unlink(missing_ok=True)
            return False, "Frame is too small for the training center crop.", wrote_frames
        for _ in range(repeat_count):
            writer.write(cropped)
            wrote_frames += 1
        frame_idx += 1

    writer.release()
    cap.release()

    if wrote_frames == 0:
        raw_path.unlink(missing_ok=True)
        return False, "No frames were written to the clip.", 0

    _, transcode_warning = _transcode_video(raw_path, out_path)
    return True, transcode_warning, wrote_frames
