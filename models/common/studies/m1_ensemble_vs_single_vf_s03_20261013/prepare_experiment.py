"""Prepare an isolated replication on the newly labeled VF_S03."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, csv, hashlib, ast
from datetime import datetime, timezone
B = experiment_directory(__file__)
ROOT = B.parents[1]
PREVIOUS = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
HUMAN = ROOT / 'data/active_learning_review/sessions/review_session_20261013_110635_bf83de.json'
VIDEO = ROOT / 'data/videos/partidos_evaluacion/VF_S03_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
session = json.loads(_resolve_model_artifact(HUMAN).read_text(encoding='utf-8'))
cs = session['candidates']
assert len(cs) == 172 and all((c['final_label'] in ['line', 'other', 'scrum'] and c['manual_labeling'] and c['reviewed_at'] for c in cs))
assert Path(session['video_path']).resolve() == VIDEO.resolve()
assert all((c['candidate_index'] == i and c['start_time'] == i * 7 and (abs(c['end_time'] - min((i + 1) * 7, 1201.2)) < 1e-08) for i, c in enumerate(cs)))
for name in ['inputs', 'results', 'logs', 'figures', 'runtime']:
    (B / name).mkdir(exist_ok=True)

def write(name, data):
    with _resolve_model_artifact(B / name).open('x', encoding='utf-8') as f:
        f.write(data)

def copy(source, target):
    write(target, _resolve_model_artifact(PREVIOUS / source).read_text(encoding='utf-8'))
write('inputs/human_session.json', json.dumps(session, ensure_ascii=False, indent=2))
fields = ['dataset_id', 'candidate_index', 'start_time', 'end_time', 'correct_label']
with _resolve_model_artifact(B / 'inputs/human_blocks.csv').open('x', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    for c in cs:
        w.writerow(dict(dataset_id=f"VF_S03_{session['session_id']}_{c['candidate_index']:04d}", candidate_index=c['candidate_index'], start_time=c['start_time'], end_time=c['end_time'], correct_label=c['final_label']))
for name in ['M1-ensemble_manifest.json', 'M1-single_manifest.json', 'selection.json', 'independence_audit.json']:
    copy('inputs/' + name, 'inputs/' + name)
copy('results/training.json', 'results/training.json')
copy('runtime.py', 'runtime.py')
infer = _resolve_model_artifact(_resolve_model_artifact(PREVIOUS / 'infer.py')).read_text(encoding='utf-8').replace('VF_S01', 'VF_S03').replace('m1_ensemble_vs_all_development_20261012', 'm1_ensemble_vs_single_vf_s03_20261013').replace('_20261012', '_20261013')
write('infer.py', infer)
quality = _resolve_model_artifact(_resolve_model_artifact(PREVIOUS / 'quality_from_fresh_features.py')).read_text(encoding='utf-8')
quality = quality.replace("p=ROOT/'data/active_learning_review/sessions/review_session_20260830_112524_0b5e58.json'", "p=HERE/'inputs/human_session.json'")
quality = quality.replace("csv_path=ROOT/'data/active_learning_dataset/sessions/20260830_112524_0b5e58/train_manifest.csv'", "csv_path=HERE/'inputs/human_blocks.csv'")
quality = quality.replace("session['status']=='exported'", "all(c['final_label'] in LABELS and c['reviewed_at'] for c in session['candidates'])")
quality = quality.replace("'section':'S01'", "'section':'S03'")
write('quality_from_fresh_features.py', quality)
images = _resolve_model_artifact(_resolve_model_artifact(PREVIOUS / 'make_images.py')).read_text(encoding='utf-8').replace('VF_S01', 'VF_S03').replace(' and len(truths[0])==13', '')
write('make_images.py', images)
copy('verify.py', 'verify.py')
semantic = [{k: c[k] for k in ['candidate_index', 'start_time', 'end_time', 'final_label']} for c in cs]
request = {'utc': datetime.now(timezone.utc).isoformat(), 'authorization': 'Ground truth / six-fold M1 ensemble / all-development M1 single model comparison after VF_S03 labeling is complete.', 'video': str(VIDEO), 'human_session': str(HUMAN), 'human_session_id': session['session_id'], 'human_snapshot': str(B / 'inputs/human_session.json'), 'human_label_sha256': hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest(), 'reuse_models_from': str(PREVIOUS), 'retrain': False, 'tune_with_VF': False, 'fresh_video_each_model': True, 'sequential': True, 'prediction_grid': '7s windows / 1s stride', 'quality_grid': '172 exact human blocks, final block 4.2s', 'thresholds': {'line': 0.5, 'scrum': 0.45}, 'no_slide_or_thesis_edits': True, 'human_status_on_capture': session['status'], 'reviewed_blocks': len(cs)}
write('inputs/request.json', json.dumps(request, ensure_ascii=False, indent=2))
for p in _model_artifact_glob(B, '*.py', recursive=False):
    ast.parse(_resolve_model_artifact(p).read_text(encoding='utf-8'), filename=str(p))
write('state.json', json.dumps({'stage': 'ready', 'utc': datetime.now(timezone.utc).isoformat(), 'human_blocks': 172}, indent=2))
print('VF_S03 ready: 172 real blocks, same frozen models and thresholds, fresh sequential inference.')
