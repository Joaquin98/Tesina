import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import importlib.util
from core import evaluate_events
spec = importlib.util.spec_from_file_location('verify_prior', _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/publish.py'))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

def main():
    if not _resolve_model_artifact(HERE / 'professional_m1_replay_control.json').exists():
        import subprocess
        subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'replay_control.py'))], cwd=ROOT, check=True)
    control = read(HERE / 'professional_m1_replay_control.json')
    assert control['status'] == 'PASS'
    for proof in control['controls'].values():
        assert sha(proof['source_result']) == proof['source_result_sha256']
        assert all((sha(p) == h for p, h in proof['model_hashes'].items()))
    protocol = read(HERE / 'protocol.json')
    train = read(HERE / 'training_complete.json')
    assert train['status'] == 'complete' and len(train['models']) == 7
    assert protocol['source_hashes'] == {p: sha(p) for p in protocol['source_hashes']}
    assert protocol['cache_stats'] == {p: stat(p) for p in protocol['cache_stats']}
    native_stats = read(HERE / 'source_cache_stats.json')
    assert native_stats == {p: stat(p) for p in native_stats}
    assert len({r['initial_weights_sha256'] for r in train['models'] if not r['reused']}) == 1
    for r in train['models']:
        assert sha(r['model_path']) == r['model_sha256']
    refs = {j['date']: read(j['reference']) for j in JOBS}
    truths = {j['date']: refs[j['date']]['reference_segments' if j['kind'] == 'human' else 'events'] for j in JOBS}
    truths['VF'] = read(ROOT / 'data/m1_recall_cost_20261027/vf25_results.json')['reference_segments']
    proofs = {}
    for path in _model_artifact_glob(HERE / 'evaluation', '*/*/metrics.json', recursive=False):
        v = read(path)
        truth = truths[v['date']]
        events = read(path.parent / 'events.json')
        again = evaluate_events(truth, events)
        assert again['metrics'] == v['metrics']
        check = prior.independent_check(truth, events, v['metrics'], 4804.8 if v['date'] == 'VF' else next((j['duration'] for j in JOBS if j['date'] == v['date'])))
        for k in ['unique_flagged_s', 'flagged_outside_true_segments_s']:
            assert abs(check[k] - v['review'][k]) < 1e-06
        if 'reference_file_sha256' in v:
            assert sha(next((j['reference'] for j in JOBS if j['date'] == v['date']))) == v['reference_file_sha256']
        for p, h in v.get('model_hashes', {}).items():
            assert sha(p) == h
        if 'source_result_sha256' in v:
            assert sha(v['source_result']) == v['source_result_sha256']
        proofs[f"{v['date']}_{v['family']}_{v['key']}_{v['stride']}"] = check
    assert len(proofs) == 70
    save(HERE / 'verification.json', dict(status='PASS', configurations=70, independent_matching_and_review=proofs, professional_m1_replay_control=control, training_sources_unchanged=True, native_cache_stats_unchanged=True, native_cache_files=len(native_stats), model_hashes_unchanged=True, extractors_executed=0))
if __name__ == '__main__':
    main()
