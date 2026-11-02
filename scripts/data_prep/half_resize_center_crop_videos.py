"""Half-scale videos and center-crop to 224x224.

Workflow per frame:
1) Resize to 50% of original width/height.
2) Crop the central 224x224 square.

Default behavior is dry-run. Use --run to write outputs.

Examples:
  # Dry-run all videos in data/7seg unique
  python scripts/half_resize_center_crop_videos.py

  # Process a single video for testing
  python scripts/half_resize_center_crop_videos.py --single-video "Todo_F10_ataque_000_cut.mp4" --run

  # Process all videos and overwrite existing outputs
  python scripts/half_resize_center_crop_videos.py --run --overwrite
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
from pathlib import Path
from typing import Iterable, List
import cv2
from tqdm import tqdm
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

def find_videos(src: Path, pattern: str='*.mp4') -> List[Path]:
    return sorted([p for p in _model_artifact_glob(src, pattern, recursive=False) if p.suffix.lower() in VIDEO_EXTS])

def center_crop(frame, crop_size: int=224):
    h, w = frame.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0:y0 + crop_size, x0:x0 + crop_size]

def process_video(in_path: Path, out_path: Path, scale: float=0.5, crop_size: int=224, overwrite: bool=False) -> bool:
    if _resolve_model_artifact(out_path).exists() and (not overwrite):
        return False
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f'Cannot open video: {in_path}')
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_w = max(1, int(round(width * scale)))
    scaled_h = max(1, int(round(height * scale)))
    if scaled_w < crop_size or scaled_h < crop_size:
        print(f'Skipping {in_path.name}: scaled size {scaled_w}x{scaled_h} is smaller than {crop_size}x{crop_size}.')
        cap.release()
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (crop_size, crop_size))
    ok = True
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            cropped = center_crop(resized, crop_size=crop_size)
            if cropped is None:
                ok = False
                break
            writer.write(cropped)
    except Exception as exc:
        print(f'Error processing {in_path}: {exc}')
        ok = False
    finally:
        cap.release()
        writer.release()
    return ok

def resolve_candidates(src: Path, pattern: str, single_video: str | None, sample: int) -> Iterable[Path]:
    if single_video:
        p = Path(single_video)
        if not p.is_absolute():
            p = src / p
        return [p]
    videos = find_videos(src, pattern)
    if sample > 0:
        videos = videos[:sample]
    return videos

def main():
    parser = argparse.ArgumentParser(description='Resize each frame to half size, then center-crop to 224x224.')
    parser.add_argument('--src', default='data/7seg unique', help='Source directory')
    parser.add_argument('--dst', default='data/videos/clips_224', help='Destination directory')
    parser.add_argument('--pattern', default='*.mp4', help='Glob pattern for input videos')
    parser.add_argument('--single-video', default=None, help='Single video filename or path to process (for testing)')
    parser.add_argument('--sample', type=int, default=0, help='Process first N videos (ignored when --single-video is used)')
    parser.add_argument('--scale', type=float, default=0.5, help='Resize ratio before crop')
    parser.add_argument('--crop-size', type=int, default=224, help='Center crop size')
    parser.add_argument('--run', action='store_true', help='Actually write output videos')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    if not _resolve_model_artifact(src).exists():
        raise SystemExit(f'Source directory not found: {src}')
    candidates = list(resolve_candidates(src, args.pattern, args.single_video, args.sample))
    if not candidates:
        print(f'No videos found in {src} with pattern {args.pattern}')
        return
    total = 0
    processed = 0
    skipped = 0
    for in_path in tqdm(candidates, desc='Processing videos'):
        total += 1
        if not _resolve_model_artifact(in_path).exists():
            print(f'Missing file: {in_path}')
            skipped += 1
            continue
        out_path = dst / in_path.name
        if not args.run:
            print(f'DRY RUN -> scale {args.scale}, center-crop {args.crop_size}x{args.crop_size}: {in_path} -> {out_path}')
            skipped += 1
            continue
        ok = process_video(in_path=in_path, out_path=out_path, scale=args.scale, crop_size=args.crop_size, overwrite=args.overwrite)
        if ok:
            processed += 1
        else:
            skipped += 1
    print('\nSummary:')
    print(f'  Total scanned: {total}')
    if not args.run:
        print('  (dry-run - no files modified)')
    print(f'  Processed:     {processed}')
    print(f'  Skipped:       {skipped}')
if __name__ == '__main__':
    main()
