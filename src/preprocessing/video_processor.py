"""
Main video processor script for rugby play classification.

Usage:
    python src/preprocessing/video_processor.py --config configs/base_config.yaml
    python src/preprocessing/video_processor.py --input data/raw/match1.mp4 --output data/processed/
"""

import argparse
import yaml
from pathlib import Path
from typing import List, Dict
import logging

import cv2
from tqdm import tqdm

from video_utils import (
    load_video,
    extract_frames,
    resize_frame,
    save_video_clip,
    extract_video_segment,
    get_video_info,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 25.0,
    target_size: tuple = (224, 224),
) -> None:
    """
    Process a single video file: resize, normalize, and save.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save processed video
        target_fps: Target frames per second
        target_size: Target frame size (height, width)
    """
    logger.info(f"Processing video: {video_path.name}")
    
    # Get video info
    info = get_video_info(video_path)
    logger.info(f"  Original: {info['width']}x{info['height']}, "
                f"{info['fps']:.2f} FPS, {info['duration_sec']:.2f}s")
    
    # Extract frames
    frames = extract_frames(video_path, target_fps=target_fps)
    logger.info(f"  Extracted {len(frames)} frames")
    
    # Resize frames
    resized_frames = []
    for frame in tqdm(frames, desc="Resizing frames"):
        resized = resize_frame(frame, target_size, keep_aspect_ratio=True)
        resized_frames.append(resized)
    
    # Save processed video
    output_path = output_dir / f"processed_{video_path.name}"
    save_video_clip(resized_frames, output_path, fps=target_fps)
    
    logger.info(f"  Saved to: {output_path}")


def process_video_batch(
    input_dir: Path,
    output_dir: Path,
    target_fps: float = 25.0,
    target_size: tuple = (224, 224),
    video_extensions: List[str] = ['.mp4', '.avi', '.mov'],
) -> None:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save processed videos
        target_fps: Target frames per second
        target_size: Target frame size (height, width)
        video_extensions: List of valid video file extensions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_path in video_files:
        try:
            process_single_video(video_path, output_dir, target_fps, target_size)
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            continue
    
    logger.info("Batch processing complete!")


def segment_videos_from_annotations(
    annotations_file: Path,
    raw_dir: Path,
    output_dir: Path,
    target_fps: float = 25.0,
    target_size: tuple = (224, 224),
) -> None:
    """
    Segment videos based on annotations file.
    
    Expected CSV format: video_id,play_type,start_sec,end_sec
    
    Args:
        annotations_file: Path to annotations CSV
        raw_dir: Directory containing raw videos
        output_dir: Directory to save segmented clips
        target_fps: Target FPS
        target_size: Target size (height, width)
    """
    import pandas as pd
    
    if not annotations_file.exists():
        logger.error(f"Annotations file not found: {annotations_file}")
        return
    
    # Load annotations
    df = pd.read_csv(annotations_file)
    logger.info(f"Loaded {len(df)} annotations from {annotations_file.name}")
    
    # Create output subdirectories for each play type
    for play_type in df['play_type'].unique():
        (output_dir / play_type).mkdir(parents=True, exist_ok=True)
    
    # Process each annotation
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting videos"):
        video_id = row['video_id']
        play_type = row['play_type']
        start_sec = row['start_sec']
        end_sec = row['end_sec']
        
        # Find video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mov']:
            candidate = raw_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            logger.warning(f"Video not found for {video_id}")
            continue
        
        # Output path
        output_path = output_dir / play_type / f"{video_id}_{play_type}_{start_sec:.1f}-{end_sec:.1f}.mp4"
        
        try:
            extract_video_segment(
                video_path,
                output_path,
                start_sec,
                end_sec,
                target_fps,
                target_size,
            )
        except Exception as e:
            logger.error(f"Error segmenting {video_id} at {start_sec}-{end_sec}: {e}")
            continue
    
    logger.info("Video segmentation complete!")


def main():
    parser = argparse.ArgumentParser(description="Process rugby videos for classification")
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--input',
        type=Path,
        help='Input video file or directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory'
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        help='Annotations CSV file for segmentation'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=25.0,
        help='Target FPS (default: 25.0)'
    )
    parser.add_argument(
        '--size',
        nargs=2,
        type=int,
        default=[224, 224],
        help='Target frame size as height width (default: 224 224)'
    )
    parser.add_argument(
        '--mode',
        choices=['process', 'segment'],
        default='process',
        help='Processing mode: process (resize videos) or segment (extract clips)'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        input_dir = Path(config['data']['raw_dir'])
        output_dir = Path(config['data']['processed_dir'])
        target_fps = config['data']['target_fps']
        target_size = tuple(config['data']['frame_size'])
        annotations_file = Path(config['data']['annotations_file'])
    else:
        input_dir = args.input
        output_dir = args.output
        target_fps = args.fps
        target_size = tuple(args.size)
        annotations_file = args.annotations
    
    # Validate inputs
    if not input_dir or not output_dir:
        logger.error("Please provide input and output paths via config or arguments")
        return
    
    # Run processing based on mode
    if args.mode == 'segment' and annotations_file:
        segment_videos_from_annotations(
            annotations_file,
            input_dir,
            output_dir,
            target_fps,
            target_size,
        )
    else:
        if input_dir.is_file():
            # Process single video
            process_single_video(input_dir, output_dir, target_fps, target_size)
        else:
            # Process batch
            process_video_batch(input_dir, output_dir, target_fps, target_size)


if __name__ == '__main__':
    main()
