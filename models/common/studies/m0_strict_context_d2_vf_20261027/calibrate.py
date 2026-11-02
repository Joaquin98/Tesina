"""Select class confidence thresholds on D2 out-of-fold predictions only."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from rule import select, decide, GRID, MINIMUM, TARGET
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
DATA = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
labels = ['line', 'other', 'scrum']

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def main():
    assert not _resolve_model_artifact(HERE / 'thresholds.json').exists(), 'Never overwrite frozen thresholds'
    rows = json.loads(_resolve_model_artifact(DATA / 'manifest.json').read_text())
    idx = _load_model_result_array(P / 'train_indices.npy')
    fold = _load_model_result_array(P / 'folds.npy')
    train = [rows[i] for i in idx]
    assert len(train) == 8796 and all((x['domain'] != 'VF' for x in train))
    y = np.array([labels.index(x['tag']) for x in train])
    group = [x['domain'] + ':' + (x['group_id'] if x['domain'] == 'CV' else x['source_sha256']) for x in train]
    m = {}
    for g, f in zip(group, fold):
        if g in m:
            assert m[g] == f
        m[g] = f
    hashes = {str(x): sha(x) for x in [P / 'folds.npy', P / 'train_indices.npy', DATA / 'manifest.json', _resolve_model_artifact(HERE / 'rule.py'), _resolve_model_artifact(HERE / 'calibrate.py')]}
    output = {}
    for seed in [17, 42, 2026]:
        file = P / f'seed_{seed}/oof.npy'
        hashes[str(file)] = sha(file)
        p = _load_model_result_array(file)
        assert p.shape == (8796, 9, 3)
        for k in range(3):
            part = P / f'seed_{seed}/fold_{k}'
            m = json.loads(_resolve_model_artifact(part / 'complete.json').read_text())
            assert sha(part / 'pred.npy') == m['prediction_sha256']
            assert sha(part / 'indices.npy') == m['indices_sha256']
            np.testing.assert_array_equal(_load_model_result_array(part / 'indices.npy'), idx[fold == k])
            np.testing.assert_array_equal(_load_model_result_array(part / 'pred.npy'), p[fold == k])
        final, audit = select(p, y)
        cv = []
        for k in range(3):
            tr = fold != k
            val = fold == k
            t, _ = select(p[tr], y[tr])
            base = p[val, 0].argmax(1)
            pred = decide(p[val], t)
            change = pred != base
            pr, re, f, _ = precision_recall_fscore_support(y[val], pred, labels=[0, 1, 2], zero_division=0)
            cv.append(dict(fold=k, thresholds=t, changes=int(change.sum()), corrected=int(((base != y[val]) & (pred == y[val])).sum()), broken=int(((base == y[val]) & (pred != y[val])).sum()), change_correct=int((pred[change] == y[val][change]).sum()), precision=pr.tolist(), recall=re.tolist(), f1=f.tolist()))
        output[str(seed)] = dict(thresholds=final, selection=audit, calibration_diagnostic_by_fold=cv)
    info = dict(frozen=True, classes=['line', 'scrum'], grid=GRID.tolist(), minimum_overrides=MINIMUM, target_override_precision=TARGET, scope='only D2 out-of-fold predictions, CH by source hash CV by match', conflict='higher second confidence; tie scrum', no_eligible_threshold='disable that override class', cv_caveat='threshold folds are group separated but not a fully nested retraining of all base models', hashes=hashes, seeds=output)
    (HERE / 'thresholds.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({'thresholds': {s: v['thresholds'] for s, v in output.items()}, 'cv': {s: v['calibration_diagnostic_by_fold'] for s, v in output.items()}}))
if __name__ == '__main__':
    main()
