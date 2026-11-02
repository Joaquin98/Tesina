"""Split a long video into N-minute chunks and apply training-identical preprocessing.

Preprocessing (mirrors half_resize_center_crop_videos.py / data_utils.py):
  1. Scale each frame to 50% of original resolution.
  2. Center-crop the central 224×224 square.
  3. Preserve original FPS.

Usage:
  python scripts/data_prep/split_and_preprocess_video.py --input "path/to/video.mp4"
  python scripts/data_prep/split_and_preprocess_video.py --input "path/to/video.mp4" --chunk-minutes 20 --out-dir "data/raw"
  python scripts/data_prep/split_and_preprocess_video.py --input "path/to/video.mp4" --chunk-minutes 10 --no-preprocess

Arguments:
  --input            Path to the source video file (required).
  --chunk-minutes    Chunk duration in minutes (default: 20).
  --out-dir          Output directory (default: same directory as input file).
  --scale            Downscale factor applied before cropping (default: 0.5 = 50%).
  --crop-size        Final square crop size in pixels (default: 224).
  --no-preprocess    Skip resize/crop; write raw chunks only (useful for debugging).
  --overwrite        Overwrite existing output files.
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
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
from tqdm import tqdm

def center_crop(frame, crop_size: int=224):
    """Crop the central square of size crop_size×crop_size from frame."""
    h, w = frame.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0:y0 + crop_size, x0:x0 + crop_size]

def video_info(cap: cv2.VideoCapture):
    """Return (fps, width, height, total_frames) from an open VideoCapture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return (fps, w, h, n)

def safe_filename_part(value: str) -> str:
    """Return a Windows-safe filename component while keeping readable names."""
    safe = ''.join((ch if ch.isalnum() or ch in ' ._-' else '_' for ch in value)).strip()
    safe = ' '.join(safe.split())
    return safe or 'unknown'

def infer_youtube_creator(input_path: Path) -> str | None:
    """Infer creator from data/raw/youtube/<creator>/<video> paths."""
    try:
        resolved = input_path.resolve()
    except OSError:
        resolved = input_path
    parts = list(resolved.parts)
    lowered = [part.lower() for part in parts]
    if 'youtube' not in lowered:
        return None
    youtube_idx = lowered.index('youtube')
    if youtube_idx + 1 >= len(parts):
        return None
    return parts[youtube_idx + 1]

def output_stem_for_source(input_path: Path) -> str:
    """Build chunk output stem, preserving YouTube creator when available."""
    stem = input_path.stem
    creator = infer_youtube_creator(input_path)
    if not creator:
        return stem
    safe_creator = safe_filename_part(creator)
    prefix = f'{safe_creator}__'
    if stem.lower().startswith(prefix.lower()):
        return stem
    return f'{prefix}{stem}'

def split_and_preprocess(input_path: Path, out_dir: Path, chunk_minutes: float=20.0, scale: float=0.5, crop_size: int=224, preprocess: bool=True, overwrite: bool=False) -> list[Path]:
    """Split *input_path* into fixed-length chunks with optional preprocessing.

    Returns the list of written output paths.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {input_path}')
        sys.exit(1)
    fps, orig_w, orig_h, total_frames = video_info(cap)
    total_seconds = total_frames / fps
    chunk_frames = int(round(chunk_minutes * 60 * fps))
    print(f"\n{'─' * 60}")
    print(f'  Input  : {input_path.name}')
    print(f'  Size   : {orig_w}×{orig_h}  |  FPS: {fps:.2f}  |  Duration: {total_seconds / 60:.1f} min')
    print(f'  Frames : {total_frames:,}  |  Chunk length: {chunk_frames:,} frames ({chunk_minutes} min)')
    if preprocess:
        scaled_w = max(1, int(round(orig_w * scale)))
        scaled_h = max(1, int(round(orig_h * scale)))
        if scaled_w < crop_size or scaled_h < crop_size:
            print(f'\n[ERROR] Scaled size {scaled_w}×{scaled_h} is smaller than the requested crop ({crop_size}×{crop_size}).\n  Lower --scale or --crop-size and retry.')
            cap.release()
            sys.exit(1)
        out_w, out_h = (crop_size, crop_size)
        print(f'  Scale  : {scale}  →  {scaled_w}×{scaled_h}  →  crop {crop_size}×{crop_size}')
    else:
        out_w, out_h = (orig_w, orig_h)
        print(f'  Mode   : raw split (no preprocessing)')
    print(f'  Output : {out_dir}')
    print(f"{'─' * 60}\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem_for_source(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    n_chunks = max(1, int(total_frames / chunk_frames) + (1 if total_frames % chunk_frames else 0))
    n_chunks = (total_frames + chunk_frames - 1) // chunk_frames

    def _finalize_chunk(tmp_path: Path, final_path: Path) -> None:
        """Re-encode tmp_path (mp4v) → final_path (H.264) via ffmpeg."""
        cmd = ['ffmpeg', '-y', '-i', str(tmp_path), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', str(final_path)]
        result = subprocess.run(cmd, capture_output=True)
        tmp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f'\n[ERROR] ffmpeg failed for {final_path.name}:\n{result.stderr.decode()}')
    written: list[Path] = []
    frame_idx = 0
    chunk_idx = 0
    writer = None
    out_path = None
    tmp_path = None
    pbar = tqdm(total=total_frames, unit='frames', desc='Processing')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % chunk_frames == 0:
                if writer is not None:
                    writer.release()
                    writer = None
                    _finalize_chunk(tmp_path, out_path)
                    print(f'\n  ✔  Saved chunk {chunk_idx:02d}: {out_path.name}')
                    written.append(out_path)
                chunk_idx += 1
                chunk_label = f'part{chunk_idx:02d}'
                out_path = out_dir / f'{stem}_{chunk_label}.mp4'
                tmp_path = out_dir / f'.tmp_{stem}_{chunk_label}.mp4'
                if _resolve_model_artifact(out_path).exists() and (not overwrite):
                    print(f'\n  [SKIP] {out_path.name} already exists (use --overwrite to replace)')
                    frames_to_skip = min(chunk_frames, total_frames - frame_idx)
                    for _ in range(frames_to_skip - 1):
                        cap.read()
                        pbar.update(1)
                    frame_idx += frames_to_skip
                    pbar.update(1)
                    continue
                writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (out_w, out_h))
                if not writer.isOpened():
                    print(f'\n[ERROR] Cannot open VideoWriter for {tmp_path}')
                    break
            if preprocess:
                frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                frame = center_crop(frame, crop_size=crop_size)
            if writer is not None:
                writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        if writer is not None:
            writer.release()
            if out_path is not None and tmp_path is not None and _resolve_model_artifact(tmp_path).exists():
                _finalize_chunk(tmp_path, out_path)
                if _resolve_model_artifact(out_path).exists():
                    print(f'\n  ✔  Saved chunk {chunk_idx:02d}: {out_path.name}')
                    written.append(out_path)
        cap.release()
    print(f"\n{'─' * 60}")
    print(f'  Done! {len(written)} chunk(s) written to: {out_dir}')
    for p in written:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f'    • {p.name}  ({size_mb:.1f} MB)')
    print(f"{'─' * 60}\n")
    return written

def parse_args():
    p = argparse.ArgumentParser(description='Split a long video into chunks with training-identical preprocessing.', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--input', required=True, metavar='VIDEO', help='Path to the source video file.')
    p.add_argument('--chunk-minutes', type=float, default=20.0, metavar='MIN', help='Chunk duration in minutes (default: 20).')
    p.add_argument('--out-dir', metavar='DIR', default=None, help='Output directory. Default: same directory as input.')
    p.add_argument('--scale', type=float, default=0.5, metavar='F', help='Downscale factor before center-crop (default: 0.5).')
    p.add_argument('--crop-size', type=int, default=224, metavar='PX', help='Final square crop size in pixels (default: 224).')
    p.add_argument('--no-preprocess', action='store_true', help='Skip resize/crop; write raw chunks only.')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output files.')
    return p.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not _resolve_model_artifact(input_path).exists():
        print(f'[ERROR] Input file not found: {input_path}')
        sys.exit(1)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else input_path.parent
    split_and_preprocess(input_path=input_path, out_dir=out_dir, chunk_minutes=args.chunk_minutes, scale=args.scale, crop_size=args.crop_size, preprocess=not args.no_preprocess, overwrite=args.overwrite)
if __name__ == '__main__':
    main()
