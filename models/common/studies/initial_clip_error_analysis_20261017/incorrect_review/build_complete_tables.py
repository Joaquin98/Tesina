"""Reproducible counts from preserved predictions and explicit manual annotations."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from collections import Counter
import csv, json, statistics, hashlib
OUT = experiment_directory(__file__)

def read(p):
    return list(csv.DictReader(_resolve_model_artifact(Path(p)).open(encoding='utf-8-sig')))

def write(p, rows):
    with _resolve_model_artifact(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
DISPLAY = {'R': 'Ruck sin formación fija identificada', 'S': 'Con scrum visible', 'L': 'Con lineout visible', 'M': 'Maul sin formación fija identificada', 'C': 'Evidencia insuficiente o fase indeterminada', 'O': 'Juego abierto sin las fases anteriores', 'K': 'Patada sin las fases anteriores', 'PS': 'Preparación o reinicio de scrum', 'P': 'Pausa o reorganización'}
ORDER = ['R', 'S', 'L', 'M', 'C', 'O', 'K', 'PS', 'P']

def group(tags):
    return next((k for k in ['S', 'L', 'PS', 'PL', 'M', 'R', 'C', 'K', 'P', 'O'] if k in tags.split()), 'C')
ann = json.loads(_resolve_model_artifact(OUT / 'complete_annotations.json').read_text(encoding='utf-8'))
audit = read(OUT.parent / 'per_clip_audit.csv')
original = {r['video_name']: r for r in audit}
records = []
for r in read(OUT / 'manifest.csv'):
    a = ann[r['review_id']]
    assert a['final']
    o = original[r['video_name']]
    g = group(a['tags'])
    records.append(dict(**r, confidence=o['confidence'], visual_tags=a['tags'], visual_group=g, visual_group_display=DISPLAY[g], target_visibility=a['visibility'], note=a['note'], detail_note=a.get('detail_note', ''), processed_note=a.get('processed_note', ''), review_basis=a['review_basis'], source_detail_reviewed=int(a.get('source_detail_reviewed', False)), processed_detail_reviewed=int(a.get('processed_detail_reviewed', False)), historical_labels_modified=0))
assert len(records) == 154 and len(set((r['video_name'] for r in records))) == 154
assert len(audit) == 817 and sum((r['is_error'] == '1' for r in audit)) == 154
write(OUT / 'complete_error_review.csv', records)
fp = [r for r in records if r['true_label'] == 'other']
visual = []
for g in ORDER:
    sub = [r for r in fp if r['visual_group'] == g]
    visual.append(dict(group=g, description=DISPLAY[g], n=len(sub), percent=100 * len(sub) / len(fp), pred_line=sum((r['predicted_label'] == 'line' for r in sub)), pred_scrum=sum((r['predicted_label'] == 'scrum' for r in sub))))
assert sum((r['n'] for r in visual)) == 113
write(OUT / 'complete_false_detection_situations.csv', visual)
omissions = [r for r in records if r['predicted_label'] == 'other']
vis = []
for k, label in [('recognizable', 'Formación reconocible'), ('insufficient_crop', 'Evidencia insuficiente por recorte'), ('insufficient_continuation', 'Continuación en maul sin lineout identificado')]:
    sub = [r for r in omissions if r['target_visibility'] == k]
    vis.append(dict(status=k, description=label, line=sum((r['true_label'] == 'line' for r in sub)), scrum=sum((r['true_label'] == 'scrum' for r in sub)), n=len(sub), percent=100 * len(sub) / len(omissions)))
assert sum((r['n'] for r in vis)) == 36 and vis[0]['n'] == 32
write(OUT / 'complete_omission_visibility.csv', vis)
dirs = []
for true, pred in [('other', 'line'), ('other', 'scrum'), ('line', 'other'), ('scrum', 'other'), ('scrum', 'line'), ('line', 'scrum')]:
    sub = [r for r in records if r['true_label'] == true and r['predicted_label'] == pred]
    conf = [float(r['confidence']) for r in sub]
    dirs.append(dict(true_label=true, predicted_label=pred, n=len(sub), percent_errors=100 * len(sub) / len(records), percent_true_class=100 * len(sub) / sum((r['true_label'] == true for r in audit)), mean_confidence=statistics.mean(conf), median_confidence=statistics.median(conf), confidence_ge_90=sum((c >= 0.9 for c in conf)), confidence_ge_95=sum((c >= 0.95 for c in conf))))
write(OUT / 'complete_error_directions.csv', dirs)
byname = {r['video_name']: r for r in records}
joined = []
for r in audit:
    rev = byname.get(r['video_name'])
    joined.append(dict(**r, complete_error_review_id=rev['review_id'] if rev else '', complete_error_visual_tags=rev['visual_tags'] if rev else '', complete_error_visual_note=' '.join((rev[k] for k in ['note', 'processed_note', 'detail_note'])).strip() if rev else '', complete_error_visual_group=rev['visual_group'] if rev else '', complete_error_target_visibility=rev['target_visibility'] if rev else '', complete_error_reviewed=int(rev is not None)))
write(OUT / 'audit_817_with_complete_error_review.csv', joined)
dur = [float(r['duration_s']) for r in records]
summary = dict(date='2026-10-19', test_samples=817, errors=154, visually_reviewed_errors=154, error_review_coverage=1.0, false_detections=113, omissions=36, cross_confusions=5, false_detection_groups=visual, omission_visibility=vis, error_directions=dirs, same_as_prediction_formation_visible_in_other=sum((('L' if r['predicted_label'] == 'line' else 'S') in r['visual_tags'].split() for r in fp)), processed_error_duration_s=dict(min=min(dur), max=max(dur), median=statistics.median(dur), less_than_7=sum((d < 6.99 for d in dur))), reviewed_current_media_only=True, historical_media_hash_verified=False, inference_rerun=False, labels_modified=False)
(OUT / 'complete_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
paths = [OUT.parent / 'per_clip_audit.csv', OUT.parent / 'summary.json', OUT / 'manifest.csv', OUT / 'complete_annotations.json', OUT / 'complete_error_review.csv', OUT / 'complete_summary.json'] + sorted(_model_artifact_glob(OUT, 'manual_batch_*.json', recursive=False)) + [OUT / 'complete_resolutions.json']
write(OUT / 'complete_input_sha256.csv', [dict(path=str(p), sha256=hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()) for p in paths])
print(json.dumps(summary, ensure_ascii=False, indent=2))
