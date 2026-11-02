"""
Video preprocessing utilities for rugby play classification.

This module provides functions for:
- Loading and validating videos
- Extracting frames from videos
- Normalizing and resizing frames
- Creating video clips
"""

from pathlib import Path
from typing import Tuple, Optional, List
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_video(video_path: Path) -> Tuple[cv2.VideoCapture, dict]:
    """
    Load a video file and extract metadata.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Tuple of (VideoCapture object, metadata dict)
    
    Raises:
        ValueError: If video cannot be opened
    """
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_sec': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    
    logger.info(f"Loaded video: {video_path.name} - {metadata}")
    return cap, metadata


def extract_frames(
    video_path: Path,
    target_fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to the video file
        target_fps: Target FPS for frame extraction (None = use original)
        start_frame: First frame to extract (inclusive)
        end_frame: Last frame to extract (exclusive, None = end of video)
    
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap, metadata = load_video(video_path)
    original_fps = metadata['fps']
    total_frames = metadata['frame_count']
    
    if end_frame is None:
        end_frame = total_frames
    
    if target_fps is None or target_fps >= original_fps:
        if target_fps is None:
            sample_counts = None
        else:
            start_time = start_frame / original_fps
            end_time = end_frame / original_fps
            sample_count = max(1, int(np.floor((end_time - start_time) * target_fps)))
            sample_times = start_time + (np.arange(sample_count, dtype=np.float32) / float(target_fps))
            sample_indices = np.clip(
                np.rint(sample_times * original_fps).astype(np.int32),
                start_frame,
                end_frame - 1,
            ).tolist()
            sample_counts = {}
            for index in sample_indices:
                sample_counts[index] = sample_counts.get(index, 0) + 1
    else:
        start_time = start_frame / original_fps
        end_time = end_frame / original_fps
        sample_count = max(1, int(np.floor((end_time - start_time) * target_fps)))
        sample_times = start_time + (np.arange(sample_count, dtype=np.float32) / float(target_fps))
        sample_indices = np.clip(
            np.rint(sample_times * original_fps).astype(np.int32),
            start_frame,
            end_frame - 1,
        ).tolist()
        sample_counts = {}
        for index in sample_indices:
            sample_counts[index] = sample_counts.get(index, 0) + 1
    
    frames = []
    current_frame = 0
    
    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Keep frames in range and, when requested, at timestamp-based target FPS.
            if current_frame >= start_frame:
                repeat_count = sample_counts.get(current_frame, 0) if sample_counts is not None else 1
                for _ in range(repeat_count):
                    frames.append(frame.copy())
            
            current_frame += 1
        
        logger.info(
            f"Extracted {len(frames)} frames from {video_path.name} "
            f"(target_fps: {target_fps or original_fps:.2f}, range: {start_frame}-{end_frame})"
        )
        
    finally:
        cap.release()
    
    return frames


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """
    Resize a frame to target size.
    
    Args:
        frame: Input frame (BGR)
        target_size: Target size as (height, width)
        keep_aspect_ratio: Whether to maintain aspect ratio (crops/pads if needed)
    
    Returns:
        Resized frame
    """
    target_h, target_w = target_size
    
    if not keep_aspect_ratio:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate scaling factor
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    
    return padded


def normalize_frame(
    frame: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normalize frame with ImageNet statistics.
    
    Args:
        frame: Input frame (BGR, 0-255)
        mean: Mean values for normalization (RGB order)
        std: Std values for normalization (RGB order)
    
    Returns:
        Normalized frame (RGB, float32)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Scale to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    normalized = (frame_float - mean) / std
    
    return normalized


def save_video_clip(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 25.0,
    codec: str = 'mp4v',
) -> None:
    """
    Save a list of frames as a video file.
    
    Args:
        frames: List of frames (BGR format)
        output_path: Output video path
        fps: Output video FPS
        codec: Video codec (default: 'mp4v')
    """
    if not frames:
        logger.warning("No frames to save")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    try:
        for frame in frames:
            writer.write(frame)
        
        logger.info(f"Saved video clip: {output_path} ({len(frames)} frames)")
    finally:
        writer.release()


def extract_video_segment(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    target_fps: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Extract and save a segment from a video.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        start_sec: Start time in seconds
        end_sec: End time in seconds
        target_fps: Target FPS (None = use original)
        target_size: Target size as (height, width) (None = use original)
    """
    cap, metadata = load_video(video_path)
    original_fps = metadata['fps']
    
    # Convert times to frame numbers
    start_frame = int(start_sec * original_fps)
    end_frame = int(end_sec * original_fps)
    
    # Extract frames
    frames = extract_frames(video_path, target_fps, start_frame, end_frame)
    
    # Resize if needed
    if target_size is not None:
        frames = [resize_frame(f, target_size) for f in frames]
    
    # Save
    save_fps = target_fps if target_fps else original_fps
    save_video_clip(frames, output_path, save_fps)


def get_video_info(video_path: Path) -> dict:
    """
    Get video information without loading all frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    _, metadata = load_video(video_path)
    return metadata
