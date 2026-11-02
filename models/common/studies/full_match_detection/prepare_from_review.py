"""Normalize a complete-match human-review export for the isolated experiment.

This script intentionally uses only the Python standard library so it can run in
the same environment as the web reviewer without installing extra packages.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
EXPERIMENT_ROOT = experiment_directory(__file__)
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
LABELS = ('line', 'scrum', 'other', 'no_event')
MAX_ALLOWED_GAP_SECONDS = 0.25
MAX_ALLOWED_OVERLAP_SECONDS = 0.25

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepara una exportación de revisión full_match para evaluación.')
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument('--session-id', help='ID de sesión de revisión exportada.')
    source_group.add_argument('--source', type=Path, help='Carpeta exportada de la revisión.')
    parser.add_argument('--match-id', required=True, help='Identificador estable del partido.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT, help='Raíz del repositorio; por defecto se autodetecta.')
    parser.add_argument('--output-root', type=Path, help='Carpeta de salida; por defecto data/full_match_detection/runs.')
    parser.add_argument('--no-copy-clips', action='store_true', help='Conservar las rutas exportadas en vez de copiar los clips al run.')
    parser.add_argument('--allow-gaps', action='store_true', help='Permitir huecos temporales mayores a 0.25 s, dejando constancia en summary.json.')
    return parser.parse_args()

def safe_name(value: str) -> str:
    cleaned = re.sub('[^A-Za-z0-9_.-]+', '_', value.strip())
    return cleaned.strip('._') or 'match'

def read_csv(path: Path) -> list[dict[str, str]]:
    with _resolve_model_artifact(path).open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _resolve_model_artifact(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _resolve_model_artifact(path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

def normalize_label(value: object, field_name: str) -> str:
    raw = str(value or '').strip().lower()
    aliases = {'lineout': 'line', 'line-out': 'line', 'linea': 'line', 'línea': 'line', 'scrum': 'scrum', 'melé': 'scrum', 'mele': 'scrum', 'otro': 'other', 'others': 'other', 'no-event': 'no_event', 'noevent': 'no_event', 'background': 'no_event', 'none': 'no_event', 'pausa': 'no_event'}
    normalized = aliases.get(raw, raw)
    if normalized not in LABELS:
        raise ValueError(f"Etiqueta inválida en {field_name}: {value!r}. Se esperaba una de {', '.join(LABELS)}.")
    return normalized

def float_value(row: dict[str, str], key: str, row_number: int) -> float:
    raw = row.get(key)
    if raw is None or str(raw).strip() == '':
        raise ValueError(f'Falta {key} en la fila {row_number}.')
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f'{key} inválido en la fila {row_number}: {raw!r}') from exc

def locate_source(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    if args.source:
        source = args.source if args.source.is_absolute() else project_root / args.source
        source = source.resolve()
        if not _resolve_model_artifact(source).is_dir():
            raise FileNotFoundError(f'La fuente no es una carpeta: {source}')
        return source
    if not args.session_id:
        raise ValueError('Indicá --session-id o --source.')
    candidates = [project_root / 'data' / 'full_match_detection' / 'review_sessions' / args.session_id, project_root / 'data' / 'active_learning_dataset' / 'sessions' / args.session_id, project_root / 'data' / 'active_learning_review' / 'exports' / args.session_id]
    for candidate in candidates:
        if _resolve_model_artifact(candidate).is_dir():
            return candidate.resolve()
    searched = '\n'.join((f'  - {candidate}' for candidate in candidates))
    raise FileNotFoundError(f'No se encontró la sesión {args.session_id}. Se buscó en:\n{searched}')

def find_clip(source_dir: Path, project_root: Path, row: dict[str, str]) -> Path | None:
    raw_path = str(row.get('clip_path') or '').strip()
    raw_name = str(row.get('clip_name') or row.get('video_name') or '').strip()
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path if path.is_absolute() else source_dir / path)
        candidates.append(project_root / path)
    if raw_name:
        candidates.extend([source_dir / raw_name, source_dir / 'clips' / raw_name, source_dir / 'cropped_clips' / raw_name])
    for candidate in candidates:
        if _resolve_model_artifact(candidate).is_file():
            return candidate.resolve()
    return None

def source_session_id(source_dir: Path, rows: list[dict[str, str]]) -> str:
    for row in rows:
        if row.get('session_id'):
            return str(row['session_id'])
    return source_dir.name

def validate_review_mode(source_dir: Path) -> None:
    snapshot_path = source_dir / 'session_snapshot.json'
    if not _resolve_model_artifact(snapshot_path).is_file():
        return
    try:
        snapshot = json.loads(_resolve_model_artifact(snapshot_path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'session_snapshot.json inválido: {snapshot_path}') from exc
    review_mode = snapshot.get('review_mode')
    if review_mode and review_mode != 'full_match':
        raise ValueError(f"La sesión indica review_mode={review_mode!r}; este experimento requiere review_mode='full_match'.")

def normalize_rows(source_dir: Path, project_root: Path, rows: list[dict[str, str]], match_id: str, output_dir: Path, copy_clips: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    seen_exact: set[tuple[str, float, float, str]] = set()
    interval_labels: dict[tuple[str, float, float], str] = {}
    session_id = source_session_id(source_dir, rows)
    clips_dir = output_dir / 'clips'
    if copy_clips:
        clips_dir.mkdir(parents=True, exist_ok=True)
    for row_number, row in enumerate(rows, start=2):
        start = round(float_value(row, 'start_time', row_number), 3)
        end = round(float_value(row, 'end_time', row_number), 3)
        if end <= start:
            raise ValueError(f'El intervalo de la fila {row_number} no es válido: {start}–{end}.')
        final_raw = row.get('final_label') or row.get('correct_label') or row.get('tag')
        predicted_raw = row.get('predicted_label') or row.get('prediction') or 'other'
        final_label = normalize_label(final_raw, 'final_label')
        predicted_label = normalize_label(predicted_raw, 'predicted_label')
        source_video = str(row.get('source_video') or row.get('video') or '').strip()
        exact_key = (source_video, start, end, final_label)
        interval_key = (source_video, start, end)
        if exact_key in seen_exact:
            conflicts.append({'type': 'duplicate', 'row_number': row_number, 'source_video': source_video, 'start_time': start, 'end_time': end, 'final_label': final_label})
            continue
        previous_label = interval_labels.get(interval_key)
        if previous_label and previous_label != final_label:
            conflicts.append({'type': 'conflicting_label', 'row_number': row_number, 'source_video': source_video, 'start_time': start, 'end_time': end, 'previous_label': previous_label, 'final_label': final_label})
            continue
        seen_exact.add(exact_key)
        interval_labels[interval_key] = final_label
        clip = find_clip(source_dir, project_root, row)
        target_clip: Path | None = clip
        if copy_clips and clip:
            block_index = len(normalized)
            target_name = f"block_{block_index:05d}_{int(start * 1000):09d}_{int(end * 1000):09d}_{final_label}{clip.suffix.lower() or '.mp4'}"
            target_clip = clips_dir / target_name
            shutil.copy2(clip, target_clip)
        normalized.append({'match_id': match_id, 'source_session_id': session_id, 'source_video': source_video, 'block_index': len(normalized), 'start_time': start, 'end_time': end, 'duration': round(end - start, 3), 'predicted_label': predicted_label, 'final_label': final_label, 'clip_path': str(target_clip) if target_clip else '', 'manifest_video_name': row.get('clip_name') or row.get('video_name') or '', 'confidence': row.get('confidence') or '', 'reviewed_at': row.get('reviewed_at') or '', 'review_outcome': row.get('review_outcome') or '', 'predicted_probabilities': row.get('predicted_probabilities') or ''})
    return (normalized, conflicts)

def audit_timeline(rows: list[dict[str, object]], allow_gaps: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    timeline: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row['source_video']), []).append(row)
    for source_video, source_rows in by_source.items():
        ordered = sorted(source_rows, key=lambda item: (float(item['start_time']), float(item['end_time'])))
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = float(previous['end_time'])
            current_start = float(current['start_time'])
            delta = round(current_start - previous_end, 3)
            event = {'source_video': source_video, 'previous_end_time': previous_end, 'current_start_time': current_start, 'delta_seconds': delta}
            if delta > MAX_ALLOWED_GAP_SECONDS:
                event['type'] = 'gap'
                errors.append(event)
            elif delta < -MAX_ALLOWED_OVERLAP_SECONDS:
                event['type'] = 'overlap'
                errors.append(event)
            timeline.append(event)
    gaps = [event for event in errors if event['type'] == 'gap']
    overlaps = [event for event in errors if event['type'] == 'overlap']
    if overlaps:
        raise ValueError(f'Se detectaron {len(overlaps)} solapamientos temporales mayores a {MAX_ALLOWED_OVERLAP_SECONDS} s. Revisá la exportación.')
    if gaps and (not allow_gaps):
        raise ValueError(f'Se detectaron {len(gaps)} huecos temporales mayores a {MAX_ALLOWED_GAP_SECONDS} s. Revisá la cobertura o usá --allow-gaps.')
    return (timeline, errors)

def sha256_file(path: Path | None) -> str | None:
    if not path or not _resolve_model_artifact(path).is_file():
        return None
    digest = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_dir = locate_source(args)
    validate_review_mode(source_dir)
    review_path = source_dir / 'review_log.csv'
    manifest_path = source_dir / 'train_manifest.csv'
    input_path = review_path if _resolve_model_artifact(review_path).is_file() else manifest_path
    if not _resolve_model_artifact(input_path).is_file():
        raise FileNotFoundError(f'La fuente no contiene review_log.csv ni train_manifest.csv: {source_dir}')
    rows = read_csv(input_path)
    if not rows:
        raise ValueError(f'La exportación está vacía: {input_path}')
    if not any((str(row.get('start_time') or '').strip() for row in rows)):
        raise ValueError("La exportación no contiene start_time/end_time. Volvé a exportar usando la modalidad 'Revisar partido completo'.")
    match_id = safe_name(args.match_id)
    output_root = args.output_root or project_root / 'data' / 'full_match_detection' / 'runs'
    output_dir = (output_root / match_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized, conflicts = normalize_rows(source_dir=source_dir, project_root=project_root, rows=rows, match_id=match_id, output_dir=output_dir, copy_clips=not args.no_copy_clips)
    timeline, timeline_errors = audit_timeline(normalized, allow_gaps=args.allow_gaps)
    missing_clips = sum((1 for row in normalized if not row['clip_path']))
    class_counts = Counter((str(row['final_label']) for row in normalized))
    predicted_counts = Counter((str(row['predicted_label']) for row in normalized))
    total_duration = sum((float(row['duration']) for row in normalized))
    manifest_fields = list(normalized[0].keys()) if normalized else []
    write_csv(output_dir / 'normalized_manifest.csv', manifest_fields, normalized)
    write_csv(output_dir / 'block_annotations.csv', manifest_fields, normalized)
    write_csv(output_dir / 'external_test.csv', ['video_name', 'tag'], [{'video_name': row['clip_path'] or row['manifest_video_name'], 'tag': row['final_label']} for row in normalized])
    write_csv(output_dir / 'conflicts.csv', list(conflicts[0].keys()) if conflicts else ['type', 'row_number', 'source_video', 'start_time', 'end_time'], conflicts)
    summary = {'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'match_id': match_id, 'source_dir': str(source_dir), 'input_path': str(input_path), 'source_session_id': source_session_id(source_dir, rows), 'block_duration_seconds': 7.0, 'block_count': len(normalized), 'total_reviewed_duration_seconds': round(total_duration, 3), 'class_counts': {label: class_counts.get(label, 0) for label in LABELS}, 'predicted_counts': {label: predicted_counts.get(label, 0) for label in LABELS}, 'duplicate_or_conflict_count': len(conflicts), 'missing_clip_count': missing_clips, 'timeline_transition_count': len(timeline), 'timeline_issue_count': len(timeline_errors), 'timeline_issues': timeline_errors, 'allow_gaps': args.allow_gaps, 'clips_copied': not args.no_copy_clips, 'first_clip_sha256': sha256_file(Path(str(normalized[0]['clip_path']))) if normalized and normalized[0]['clip_path'] else None}
    write_json(output_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'ERROR: {exc}')
