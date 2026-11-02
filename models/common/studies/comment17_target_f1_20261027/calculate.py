"""Read historical clip results and compute mean F1 for line and scrum.

No training, inference, or thesis modification. Reconstruction uses unique
integer class totals consistent with stored reports and four-decimal metrics.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import csv
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
ROOT = experiment_directory(__file__).parents[1]
OUT = experiment_directory(__file__)
BASE = ROOT / 'models/M0/results'
LABELS = ['line', 'other', 'scrum']
SUPPORTS = [183, 544, 90]
results, provenance = ([], {})

def read(path):
    path = Path(path)
    data = _resolve_model_artifact(path).read_bytes()
    provenance[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')

def record(group, name, fs, method, source):
    results.append(dict(group=group, name=name, f1_line=fs[0], f1_other=fs[1], f1_scrum=fs[2], f1_macro=sum(fs) / 3, f1_target=(fs[0] + fs[2]) / 2, method=method, source=str(source)))

def reconstruct(group, name, meta, source, report=None):
    diag = [round(float(meta['recall_' + c]) * s) for c, s in zip(LABELS, SUPPORTS)]
    assert all((abs(t / s - float(meta['recall_' + c])) < 5.001e-05 for c, s, t in zip(LABELS, SUPPORTS, diag)))
    parsed = {}
    if report is not None:
        parsed = {m[0]: list(map(float, m[1:])) for m in re.findall('^\\s*(line|other|scrum)\\s+([\\d.]+)\\s+([\\d.]+)\\s+([\\d.]+)\\s+(\\d+)\\s*$', read(report), re.M)}
        assert set(parsed) == set(LABELS)
    choices = []
    for c, t, s in zip(LABELS, diag, SUPPORTS):
        choices.append([k for k in range(max(1, t), 818) if not parsed or (abs(t / k - parsed[c][0]) < 0.00500001 and abs(2 * t / (s + k) - parsed[c][2]) < 0.00500001)])
    mid_choices = set(choices[1])
    macro = float(meta.get('f1_macro', meta.get('val_f1_macro')))
    weighted = float(meta.get('f1_weighted', meta.get('val_f1_weighted')))
    hits = []
    for a, b in itertools.product(choices[0], choices[2]):
        mid = 817 - a - b
        if mid not in mid_choices:
            continue
        cols = [a, mid, b]
        off_total = 817 - sum(diag)
        if any((s - t + k - t > off_total for s, k, t in zip(SUPPORTS, cols, diag))):
            continue
        fs = [2 * t / (s + k) for t, s, k in zip(diag, SUPPORTS, cols)]
        if abs(sum(fs) / 3 - macro) < 5.001e-05 and abs(sum((f * s for f, s in zip(fs, SUPPORTS))) / 817 - weighted) < 5.001e-05:
            hits.append((cols, fs))
    assert len(hits) == 1, (name, len(hits))
    record(group, name, hits[0][1], 'unique_integer_margins', source)

def from_predictions(group, name, rows, source):
    cm = [[0] * 3 for _ in LABELS]
    for r in rows:
        cm[LABELS.index(r['true_label'])][LABELS.index(r['predicted_label'])] += 1
    assert [sum(row) for row in cm] == SUPPORTS
    fs = [2 * cm[i][i] / (sum(cm[i]) + sum((row[i] for row in cm))) for i in range(3)]
    record(group, name, fs, 'saved_predictions', source)
for name, m, r in [('Individual de referencia', 'best_model/best_model_results.csv', 'best_model/classification_report.txt'), ('Reentrenamiento enfocado', 'best_model_focused/best_focused_metrics.json', 'best_model_focused/classification_report_focused.txt'), ('Revision humana', 'human_review_retrain/metrics.json', 'human_review_retrain/classification_report.txt'), ('Ligero 20 fotogramas', 'light_live/resnet50__lstm__frames20/metrics.json', 'light_live/resnet50__lstm__frames20/classification_report.txt'), ('Ligero 25 fotogramas', 'light_live/resnet50__lstm__frames25/metrics.json', 'light_live/resnet50__lstm__frames25/classification_report.txt')]:
    p = BASE / m
    meta = json.loads(read(p)) if p.suffix == '.json' else next(csv.DictReader(read(p).splitlines()))
    reconstruct('main', name, meta, str(p.relative_to(ROOT)), BASE / r)
p = BASE / 'ensemble_top3_results.json'
e = json.loads(read(p))
cm = e['confusion_matrix']
fs = [2 * cm[i][i] / (sum(cm[i]) + sum((row[i] for row in cm))) for i in range(3)]
assert abs(sum(fs) / 3 - e['f1_macro']) < 1e-12
record('main', 'Ensemble 3', fs, 'saved_confusion_matrix', str(p.relative_to(ROOT)))
p = BASE / 'frame_only_baseline/per_video_predictions.csv'
from_predictions('main', 'Sin dimension temporal', list(csv.DictReader(read(p).splitlines())), str(p.relative_to(ROOT)))
p = BASE / 'best_model_sparse_frames/predictions.csv'
preds = list(csv.DictReader(read(p).splitlines()))
for run in ['full_density', 'uniform_frames10', 'uniform_frames25', 'uniform_frames50', 'uniform_frames100']:
    selected = [r for r in preds if r['run'] == run]
    if not selected:
        print('Available sparse run identifiers:', sorted(set((r['run'] for r in preds))))
        raise ValueError(run)
    from_predictions('sparse', run, selected, str(p.relative_to(ROOT)))
p = BASE / 'grid_search_results.csv'
rows = list(csv.DictReader(read(p).splitlines()))
original = [r for r in rows if float(r['p_learning_rate']) in [0.0001, 0.001] and float(r['p_dropout']) in [0.3, 0.5] and (int(r['p_batch_size']) in [16, 32])]
selected = sorted(original, key=lambda r: float(r['val_f1_macro']), reverse=True)[:5]
for extr in ['efficientnetb0', 'inceptionv3']:
    selected.append(max((r for r in original if r['extractor'] == extr), key=lambda r: float(r['val_f1_macro'])))
for row in selected:
    reconstruct('grid', row['trial_name'], row, str(p.relative_to(ROOT)), BASE / 'grid_search' / row['trial_name'] / 'classification_report.txt')
focus_ids = ['resnet50__gru__u128__bi__drop2__d0__lr3e-4__bs16', 'resnet50__lstm__u128__bi__drop5__d64__lr3e-4__bs16', 'resnet50__gru__u64__bi__drop2__d0__lr3e-4__bs32', 'resnet50__gru__u64__bi__drop5__d64__lr3e-4__bs24', 'resnet50__gru__u128__bi__drop4__d64__lr1e-3__bs32']
for ident in focus_ids:
    row = next((r for r in rows if r['trial_name'] == ident))
    reconstruct('focused_grid', ident, row, str(p.relative_to(ROOT)), BASE / 'grid_search' / ident / 'classification_report.txt')
p = ROOT / 'models/two_stream/results/two_stream_results.csv'
meta = next((r for r in csv.DictReader(read(p).splitlines()) if r['f1_macro'] == '0.6284'))
reconstruct('main', 'Two-Stream hibrida', meta, str(p.relative_to(ROOT)))
p = ROOT / 'models/two_stream/results/mobilenetv2_corrected_full_seed42_20260823/two_stream_results.csv'
meta = next(csv.DictReader(read(p).splitlines()))
reconstruct('main', 'Two-Stream MobileNetV2', meta, str(p.relative_to(ROOT)), p.parent / 'classification_report.txt')
reference = next((r for r in results if r['name'] == 'Individual de referencia'))
dense = next((r for r in results if r['name'] == 'full_density'))
assert abs(reference['f1_target'] - dense['f1_target']) < 1e-12
with _resolve_model_artifact(OUT / 'results.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)
(OUT / 'verification.json').write_text(json.dumps(dict(formula='(F1_line + F1_scrum)/2; other remains a possible prediction and contributes false positives/negatives', rows=len(results), sources_sha256=provenance, reconstruction='Each reconstructed row has exactly one set of integer prediction totals consistent with stored recalls, rounded precision/F1, macro F1 and weighted F1; no full confusion matrix is claimed.', reference_crosscheck='Matches exact saved full-density predictions', not_calculated=['Ensemble 5 and 10: no matching full-precision per-class report or confusion matrix located', 'Random search winner: validation, not comparable to test scores'], thesis_modified=False), ensure_ascii=False, indent=2), encoding='utf-8')
for r in results:
    print(f"{r['group']:14} {r['name']:65} macro={r['f1_macro']:.6f} target={r['f1_target']:.6f}")
