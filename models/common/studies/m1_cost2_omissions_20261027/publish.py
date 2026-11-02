"""Focused review of existing M1 cost2 omissions; no inference or training."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, copy, urllib.request, hashlib
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(source_directory(ROOT / 'data/m1_recall_cost_20261027')))
sys.dont_write_bytecode = True
from core import evaluate_events
CID = '20261027_argentina_allblacks_20250816_human_m1x2_omissions'

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def api(path, data=None):
    body = json.dumps(data, ensure_ascii=False).encode('utf-8') if data is not None else None
    req = urllib.request.Request('http://127.0.0.1:5000' + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)
ref = read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
for s in ref['provenance']:
    assert hashlib.sha256(_resolve_model_artifact(Path(s['path'])).read_bytes()).hexdigest() == s['sha256']
records = {}
evaluations = []
matches = {}
for stride in (1, 5):
    title = 'M1 ×2 · ' + ('completo' if stride == 1 else 'cada 5')
    r = read(ROOT / f'data/results/result_20261027_argentina_allblacks_20250816_m1_cost2_stride{stride}.json')
    r.pop('omission_review', None)
    r['model_name'] = title
    records[title] = r
    v = evaluate_events(ref['reference_segments'], r['detections'])
    assert v['metrics']['total']['tp'] == (46 if stride == 1 else 44)
    v.update(title=title, key='full' if stride == 1 else 'every5', stride=stride)
    evaluations.append(v)
    matches[title] = {x['reference_id'] for x in v['assignments']}
models = [dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[p['play_type'], p['start_time'], p['end_time']] for p in records[v['title']]['detections']]) for v in evaluations]
segments = [dict(**s, detected={v['key']: s['id'] in matches[v['title']] for v in evaluations}) for s in ref['reference_segments'] if not all((s['id'] in m for m in matches.values()))]
assert len(segments) == 3 and all((s['play_type'] == 'line' for s in segments))
first = next(iter(records.values()))
first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=segments, context_s=3, method='sameclass overlap>=1s, one-to-one')
payload = dict(id=CID, video_path=first['video_path'], video_name='Argentina–NZ 16/08/2025 · Omisiones M1 ×2 completo/cada 5', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={name: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for name, r in records.items()})
assert api('/api/comparison-history', payload)['success']
assert api('/api/comparison-history/' + CID)['comparison']['results'] == records
(HERE / 'review.json').write_text(json.dumps(dict(status='complete', comparison_url='/compare?comparison=' + CID + '#omisiones', reference_id=ref['revision_id'], segments=segments, evaluations=evaluations), ensure_ascii=False, indent=2), encoding='utf-8')
print('Published focused review: 1 full omission, 3 every5 omissions; 3 unique line segments')
