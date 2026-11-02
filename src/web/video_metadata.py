"""Small, cached metadata reads for videos shown in the web library."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
from pathlib import Path
import cv2

def _finite_non_negative(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0

@lru_cache(maxsize=1024)
def _read_video_metadata_cached(path: str, modified_ns: int, size: int) -> tuple[int, int, float]:
    """Read dimensions and duration, invalidating the cache when a file changes."""
    del modified_ns, size
    cap = cv2.VideoCapture(path)
    try:
        width = int(_finite_non_negative(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(_finite_non_negative(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = _finite_non_negative(cap.get(cv2.CAP_PROP_FPS))
        frame_count = _finite_non_negative(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    duration = 0.0
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    return (width, height, duration)

def read_video_metadata(video_file: Path) -> dict[str, int | float]:
    """Return stable JSON-ready video metadata without decoding the full file."""
    stat = video_file.stat()
    width, height, duration = _read_video_metadata_cached(str(video_file), stat.st_mtime_ns, stat.st_size)
    return {'width': width, 'height': height, 'duration': round(duration, 3)}

def read_video_metadata_many(video_files: list[Path]) -> dict[Path, dict[str, int | float]]:
    """Read a library concurrently so the initial web request stays responsive."""
    unique_files = list(dict.fromkeys(video_files))
    if not unique_files:
        return {}
    worker_count = min(8, len(unique_files))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        metadata = pool.map(read_video_metadata, unique_files)
        return dict(zip(unique_files, metadata))
