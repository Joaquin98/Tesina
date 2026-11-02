"""Export fixed-length video clips using source-time selection or timing normalization."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / 'data' / '7seg_exact_7s_half_center224_20260716'
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi'}

@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    frames: int
    width: int
    height: int

@dataclass
class AuditRow:
    output_name: str
    source_file: str
    mode: str
    source_duration_s: float | None
    source_fps: float | None
    source_frames: int | None
    output_duration_s: float | None
    output_fps: float | None
    output_frames: int | None
    status: str
    detail: str

def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)

def _rate(value: str | None) -> float:
    if not value or value in {'0/0', 'N/A'}:
        return 0.0
    return float(Fraction(value))

def probe_video(path: Path, *, count_frames: bool=False) -> VideoInfo:
    command = ['ffprobe', '-v', 'error']
    if count_frames:
        command.append('-count_frames')
    command.extend(['-select_streams', 'v:0', '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration', '-of', 'json', str(path)])
    result = _run(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or 'ffprobe failed')
    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise RuntimeError('no video stream')
    stream = streams[0]
    fps = _rate(stream.get('avg_frame_rate'))
    raw_frames = stream.get('nb_read_frames') or stream.get('nb_frames') or 0
    frames = int(raw_frames) if str(raw_frames).isdigit() else 0
    duration = float(stream.get('duration') or 0.0)
    if frames > 0 and fps > 0:
        duration = frames / fps
    return VideoInfo(duration_s=duration, fps=fps, frames=frames, width=int(stream.get('width') or 0), height=int(stream.get('height') or 0))

def selection_pairs(source_dir: Path, selection_dir: Path | None, selection_prefix: str) -> list[tuple[Path, Path]]:
    sources = {path.name.lower(): path for path in source_dir.iterdir() if _resolve_model_artifact(path).is_file() and path.suffix.lower() in VIDEO_EXTENSIONS}
    if selection_dir is None:
        return [(source, Path(source.name)) for source in sorted(sources.values())]
    pairs = []
    for selected in sorted(selection_dir.iterdir()):
        if not _resolve_model_artifact(selected).is_file() or selected.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        source_name = selected.name
        if selection_prefix and source_name.startswith(selection_prefix):
            source_name = source_name[len(selection_prefix):]
        source = sources.get(source_name.lower())
        if source is None:
            pairs.append((source_dir / source_name, Path(selected.name)))
        else:
            pairs.append((source, Path(selected.name)))
    return pairs

def spatial_filter() -> str:
    return 'scale=trunc(iw*0.5/2)*2:trunc(ih*0.5/2)*2,crop=224:224:(iw-224)/2:(ih-224)/2'

def build_filter(mode: str, source_duration: float) -> str:
    if mode == 'exact-from-source':
        temporal = 'setpts=PTS-STARTPTS,trim=start=5:duration=7,setpts=PTS-STARTPTS,fps=25'
    else:
        factor = 7.0 / source_duration
        temporal = f'setpts=(PTS-STARTPTS)*{factor:.12f},fps=25,tpad=stop_mode=clone:stop_duration=7,trim=duration=7,setpts=PTS-STARTPTS'
    return f'{temporal},{spatial_filter()}'

def process_one(source: Path, output: Path, mode: str, minimum_existing_duration: float) -> AuditRow:
    base = {'output_name': output.name, 'source_file': str(source.resolve()), 'mode': mode}
    if not _resolve_model_artifact(source).exists():
        return AuditRow(**base, source_duration_s=None, source_fps=None, source_frames=None, output_duration_s=None, output_fps=None, output_frames=None, status='missing_source', detail='No matching source file was found.')
    if _resolve_model_artifact(output).exists():
        return AuditRow(**base, source_duration_s=None, source_fps=None, source_frames=None, output_duration_s=None, output_fps=None, output_frames=None, status='existing_output', detail='Refused to overwrite the output.')
    try:
        source_info = probe_video(source, count_frames=True)
    except Exception as exc:
        return AuditRow(**base, source_duration_s=None, source_fps=None, source_frames=None, output_duration_s=None, output_fps=None, output_frames=None, status='probe_failed', detail=str(exc))
    if source_info.width * 0.5 < 224 or source_info.height * 0.5 < 224:
        return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=None, output_fps=None, output_frames=None, status='source_too_small', detail='Half-sized frame is smaller than the 224 x 224 crop.')
    if mode == 'exact-from-source' and source_info.duration_s < 12.0:
        return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=None, output_fps=None, output_frames=None, status='source_too_short', detail='The decoded source does not contain the complete 5-12 s interval.')
    if mode == 'normalize-existing' and source_info.duration_s < minimum_existing_duration:
        return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=None, output_fps=None, output_frames=None, status='source_too_short', detail=f'Decoded content is shorter than the configured minimum of {minimum_existing_duration:.2f} s.')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-n', '-i', str(source), '-map', '0:v:0', '-an', '-vf', build_filter(mode, source_info.duration_s), '-frames:v', '175', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    result = _run(command)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=None, output_fps=None, output_frames=None, status='ffmpeg_failed', detail=result.stderr.strip()[-1000:])
    try:
        output_info = probe_video(output, count_frames=True)
    except Exception as exc:
        return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=None, output_fps=None, output_frames=None, status='output_probe_failed', detail=str(exc))
    valid = output_info.frames == 175 and abs(output_info.fps - 25.0) < 0.001 and (output_info.width == 224) and (output_info.height == 224)
    return AuditRow(**base, source_duration_s=round(source_info.duration_s, 6), source_fps=round(source_info.fps, 6), source_frames=source_info.frames, output_duration_s=round(output_info.duration_s, 6), output_fps=round(output_info.fps, 6), output_frames=output_info.frames, status='ok' if valid else 'validation_failed', detail='' if valid else 'Expected 224 x 224, 25 FPS and exactly 175 frames.')

def write_reports(output_dir: Path, rows: list[AuditRow], args: argparse.Namespace) -> None:
    report_path = output_dir / 'regeneration_manifest.csv'
    with _resolve_model_artifact(report_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows((asdict(row) for row in rows))
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'source_dir': str(args.source_dir.resolve()), 'selection_dir': str(args.selection_dir.resolve()) if args.selection_dir else None, 'output_dir': str(output_dir.resolve()), 'source_files_modified': False, 'requested_output': {'duration_s': 7.0, 'fps': 25.0, 'frames': 175, 'width': 224, 'height': 224}, 'status_counts': status_counts, 'warning': 'normalize-existing adjusts timing only; exact-from-source selects source frames.' if args.mode == 'normalize-existing' else None}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('exact-from-source', 'normalize-existing'), default='exact-from-source')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--selection-dir', type=Path)
    parser.add_argument('--selection-prefix', default='Todo_')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--minimum-existing-duration', type=float, default=5.0)
    parser.add_argument('--workers', type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument('--limit', type=int, default=0, help='Process only the first N selected videos; zero means all videos.')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        raise SystemExit('ffmpeg and ffprobe must be available on PATH.')
    if not _resolve_model_artifact(args.source_dir).is_dir():
        raise SystemExit(f'Source directory not found: {args.source_dir}')
    if args.selection_dir and (not _resolve_model_artifact(args.selection_dir).is_dir()):
        raise SystemExit(f'Selection directory not found: {args.selection_dir}')
    source_resolved = args.source_dir.resolve()
    output_resolved = args.output_dir.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise SystemExit('Output must not be the source directory or one of its children.')
    pairs = selection_pairs(args.source_dir, args.selection_dir, args.selection_prefix)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit('No selected videos were found.')
    print(f'Selected videos: {len(pairs)}')
    print(f'Mode: {args.mode}')
    print(f'Output: {output_resolved}')
    if not args.run:
        print('Dry run only. Add --run to create the isolated dataset and reports.')
        return 0
    if _resolve_model_artifact(output_resolved).exists():
        raise SystemExit(f'Refusing to reuse existing output directory: {output_resolved}')
    clips_dir = output_resolved / 'clips'
    clips_dir.mkdir(parents=True)
    rows: list[AuditRow] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_one, source, clips_dir / output_name, args.mode, args.minimum_existing_duration): output_name.name for source, output_name in pairs}
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            completed += 1
            print(f'[{completed}/{len(futures)}] {row.output_name}: {row.status}', flush=True)
    rows.sort(key=lambda row: row.output_name.lower())
    write_reports(output_resolved, rows, args)
    ok_count = sum((row.status == 'ok' for row in rows))
    print(f'Completed: {ok_count}/{len(rows)} valid clips')
    print(f"Report: {output_resolved / 'regeneration_manifest.csv'}")
    unexpected = {row.status for row in rows if row.status not in {'ok', 'source_too_short'}}
    return 0 if not unexpected else 2
if __name__ == '__main__':
    sys.exit(main())
