"""Cross-cost comparison from frozen results."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.publish import *

def main():
    job = JOBS[0]
    truth = read(job['reference'])['reference_segments']
    links = {}
    for stride in [1, 5]:
        ds = [payload(job, 'line8', stride, 'm0'), payload(job, 'x2', stride, 'm1')]
        records = {d['model_name']: d for d in ds}
        names = list(records)
        matched = {n: {a['reference_id'] for a in evaluate_events(truth, d['detections'])['assignments']} for n, d in records.items()}
        manual = ds[0]['manual_comparison']
        ds[0]['omission_review'] = dict(reference_id=manual['reference_id'], reference_sha256=manual['reference_sha256'], context_s=3, method='same-class one-to-one overlap>=1s', models=[dict(key=f'model_{i}', title=n, missed=records[n]['human_evaluation']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[n]['detections']]) for i, n in enumerate(names)], segments=[dict(**t, detected={f'model_{i}': t['id'] in matched[n] for i, n in enumerate(names)}) for t in truth if not all((t['id'] in m for m in matched.values()))])
        cid = f'20261030_argentina_allblacks_20250816_m0_line8_scrum1_vs_m1x2_stride{stride}'
        history = dict(id=cid, video_path=ds[0]['video_path'], video_name=f'16/08/2025 Â· M0 lineÃ—8/scrumÃ—1 vs M1Ã—2 Â· stride{stride}', model_names=names, parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={n: dict(total=len(d['detections']), by_type=d['summary']['by_type'], avg_confidence=d['statistics']['avg_confidence']) for n, d in records.items()})
        assert post('/api/comparison-history', history)['success']
        assert get('/api/comparison-history/' + cid)['comparison']['results'] == records
        links[str(stride)] = BASE + '/compare?comparison=' + cid + '#omisiones'
    save(HERE / 'requested_comparison.json', dict(links=links, reference_id=manual['reference_id'], extractors_executed=0))
    print(json.dumps(links), flush=True)
if __name__ == '__main__':
    main()
