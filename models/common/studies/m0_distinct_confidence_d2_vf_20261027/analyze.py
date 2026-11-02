"""Exploratory separate confidence grid, D2 OOF followed by VF; no VF selection."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = R / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
DATA = R / 'data/multicrop_d2_stride25_vf_20261026'
BASE = R / 'data/central_recall_cost_d2_vf_20261027'
sys.path.insert(0, str(source_directory(R / 'data/m0_strict_context_d2_vf_20261027')))
from rule import decide, GRID
LAB = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def evaluate(frame, inputs):
    y = np.array([LAB.index(x) for x in frame.tag])
    records = []
    details = []
    for seed in SEEDS:
        p = inputs[seed]
        assert p.shape == (len(frame), 9, 3)
        base = p[:, 0].argmax(1)
        for l in GRID:
            for s in GRID:
                pred = decide(p, [float(l), float(s)])
                pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
                cm = confusion_matrix(y, pred, labels=[0, 1, 2])
                row = dict(seed=seed, line_threshold=float(l), scrum_threshold=float(s), accuracy=float(accuracy_score(y, pred)), f1_macro=float(f.mean()), changed=int((pred != base).sum()), corrected=int(((base != y) & (pred == y)).sum()), broken=int(((base == y) & (pred != y)).sum()))
                for j, k in enumerate(LAB):
                    row.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
                records.append(row)
                details.append(dict(seed=seed, line_threshold=float(l), scrum_threshold=float(s), predictions=pred))
    return (pd.DataFrame(records), details)

def main():
    assert not _resolve_model_artifact(HERE / 'd2_results.csv').exists(), 'Do not overwrite an existing run'
    rows = json.loads(_resolve_model_artifact(DATA / 'manifest.json').read_text())
    idx = _load_model_result_array(P / 'train_indices.npy')
    fold = _load_model_result_array(P / 'folds.npy')
    assert len(idx) == 8796 and all((rows[i]['domain'] != 'VF' for i in idx))
    train = pd.DataFrame([rows[i] for i in idx])
    groups = [r['domain'] + ':' + (r['group_id'] if r['domain'] == 'CV' else r['source_sha256']) for r in train.to_dict('records')]
    mapping = {}
    for g, f in zip(groups, fold):
        if g in mapping:
            assert mapping[g] == f
        mapping[g] = f
    paths = [P / 'train_indices.npy', P / 'folds.npy', DATA / 'manifest.json', _resolve_model_artifact(R / 'models/common/studies/m0_strict_context_d2_vf_20261027/rule.py')] + [P / f'seed_{s}/oof.npy' for s in SEEDS]
    hashes = {str(p): sha(p) for p in paths}
    oof = {s: _load_model_result_array(P / f'seed_{s}/oof.npy') for s in SEEDS}
    d2, _ = evaluate(train, oof)
    d2.to_csv(HERE / 'd2_results.csv', index=False)
    protocol = dict(grid=GRID.tolist(), n_combinations=len(GRID) ** 2, decision='central default; two context argmax votes same alternative event with class-specific probability threshold; highest second confidence wins, tie scrum', no_vf_threshold_selection=True, d2='out of fold predictions; CV grouped by match, CH grouped by source hash', limitations=['VF previously observed; exploratory', 'D2 OOF base models trained on 2/3 versus VF models all D2', 'no calibration guarantee; confidence levels are not per-view true precision', 'multiple comparisons'], input_hashes=hashes, d2_results_sha256=sha(HERE / 'd2_results.csv'))
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    frozen = sha(HERE / 'protocol.json')
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    assert len(frame) == 688
    vfs = {}
    for seed in SEEDS:
        p = P / f'seed_{seed}/vf_views.npy'
        hashes[str(p)] = sha(p)
        vfs[seed] = _load_model_result_array(p)
        np.testing.assert_allclose(vfs[seed][:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
    vf, preds = evaluate(frame, vfs)
    vf.to_csv(HERE / 'vf_results.csv', index=False)
    np.savez_compressed(HERE / 'vf_predictions.npz', **{f"seed_{i['seed']}_line_{round(i['line_threshold'] * 100)}_scrum_{round(i['scrum_threshold'] * 100)}": i['predictions'] for i in preds})
    assert sha(HERE / 'protocol.json') == frozen
    for path, h in hashes.items():
        assert sha(Path(path)) == h, path
    info = dict(complete=True, protocol_sha256=frozen, threshold_grid_fixed_before_vf=True, n_combinations=49, seeds=SEEDS, blocks=688, mean_vf=vf.groupby(['line_threshold', 'scrum_threshold']).mean(numeric_only=True).reset_index().to_dict(orient='records'), vf_input_hashes={str(P / f'seed_{s}/vf_views.npy'): hashes[str(P / f'seed_{s}/vf_views.npy')] for s in SEEDS})
    (HERE / 'completion.json').write_text(json.dumps(info, indent=2))
    print('Completed D2 OOF and VF: 49 combinations x 3 seeds')
if __name__ == '__main__':
    main()
