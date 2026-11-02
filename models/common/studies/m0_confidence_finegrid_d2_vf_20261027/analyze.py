"""Predeclared 51x51 confidence grid, D2 OOF before reused VF."""
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
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = R / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
DATA = R / 'data/multicrop_d2_stride25_vf_20261026'
BASE = R / 'data/central_recall_cost_d2_vf_20261027'
OLD = R / 'data/m0_distinct_confidence_d2_vf_20261027'
sys.path.insert(0, str(source_directory(R / 'data/m0_strict_context_d2_vf_20261027')))
from rule import decide
LAB = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]
GRID = np.arange(50, 101, dtype=np.int16) / 100

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def metric(y, p, base):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    d = cm.diagonal()
    pr = np.divide(d, cm.sum(0), out=np.zeros(3), where=cm.sum(0) != 0)
    re = np.divide(d, cm.sum(1), out=np.zeros(3), where=cm.sum(1) != 0)
    f = np.divide(2 * d, cm.sum(0) + cm.sum(1), out=np.zeros(3), where=cm.sum(0) + cm.sum(1) != 0)
    result = dict(accuracy=float(d.sum() / len(y)), f1_macro=float(f.mean()), changes=int((p != base).sum()), corrected=int(((base != y) & (p == y)).sum()), broken=int(((base == y) & (p != y)).sum()))
    for j, k in enumerate(LAB):
        result.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
    return result

def sweep(frame, inputs, save_predictions=False):
    y = np.array([LAB.index(x) for x in frame.tag])
    allrows = []
    arrays = {}
    for seed in SEEDS:
        p = inputs[seed]
        assert p.shape == (len(y), 9, 3) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(2), 1, atol=1e-05)
        v = p.argmax(2)
        base = v[:, 0]
        ls = np.sort(np.where(v[:, 1:] == 0, p[:, 1:, 0], -1.0), axis=1)[:, -2]
        ss = np.sort(np.where(v[:, 1:] == 2, p[:, 1:, 2], -1.0), axis=1)[:, -2]
        for li, si in [(0, 0), (20, 49), (40, 30), (49, 49), (50, 50)]:
            le = (ls >= GRID[li]) & (base != 0)
            se = (ss >= GRID[si]) & (base != 2)
            pred = np.where(le | se, np.where(np.where(se, ss, -1) >= np.where(le, ls, -1), 2, 0), base)
            np.testing.assert_array_equal(pred, decide(p, [GRID[li], GRID[si]]))
        matrix = np.empty((len(GRID), len(GRID), len(y)), dtype=np.uint8) if save_predictions else None
        for i, l in enumerate(GRID):
            le = (ls >= l) & (base != 0)
            for j, s in enumerate(GRID):
                se = (ss >= s) & (base != 2)
                pred = np.where(le | se, np.where(np.where(se, ss, -1) >= np.where(le, ls, -1), 2, 0), base)
                allrows.append(dict(seed=seed, line_threshold=float(l), scrum_threshold=float(s), **metric(y, pred, base)))
                if save_predictions:
                    matrix[i, j] = pred
        if save_predictions:
            arrays[f'seed_{seed}'] = matrix
        allrows.append(dict(seed=seed, line_threshold=np.nan, scrum_threshold=np.nan, **metric(y, base, base)))
    return (pd.DataFrame(allrows), arrays)

def main():
    if _resolve_model_artifact(HERE / 'protocol.json').exists():
        raise RuntimeError('Existing run: do not overwrite frozen protocol')
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
    inputs = [_resolve_model_artifact(HERE / 'analyze.py'), _resolve_model_artifact(R / 'models/common/studies/m0_strict_context_d2_vf_20261027/rule.py'), P / 'train_indices.npy', P / 'folds.npy', DATA / 'manifest.json'] + [P / f'seed_{s}/oof.npy' for s in SEEDS]
    hashes = {str(p): sha(p) for p in inputs}
    oof = {s: _load_model_result_array(P / f'seed_{s}/oof.npy') for s in SEEDS}
    d2, _ = sweep(train, oof)
    d2.to_csv(HERE / 'd2_results.csv', index=False)
    proto = dict(grid_min=0.5, grid_max=1.0, grid_step=0.01, combinations=2601, seeds=SEEDS, rule='central fallback; two contexts of same alternative event with class-specific probability threshold; larger second confidence, tie scrum', stage='D2 OOF completed before opening VF', hashes=hashes, d2_results_sha256=sha(HERE / 'd2_results.csv'), selection='no VF selection', limitations=['VF previously explored', 'thresholds uncalibrated', 'OOF base fit 2/3 of D2; VF base fit all D2', 'CV grouped by match; CH by source hash with incomplete provenance', '51x51 multiple comparisons; 3 seeds not ensemble'])
    (HERE / 'protocol.json').write_text(json.dumps(proto, indent=2))
    frozen = sha(HERE / 'protocol.json')
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    assert len(frame) == 688
    vf = {}
    for s in SEEDS:
        path = P / f'seed_{s}/vf_views.npy'
        hashes[str(path)] = sha(path)
        vf[s] = _load_model_result_array(path)
        np.testing.assert_allclose(vf[s][:, 0], _load_model_result_array(BASE / f'seed_{s}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
    r, arrays = sweep(frame, vf, True)
    r.to_csv(HERE / 'vf_results.csv', index=False)
    np.savez_compressed(HERE / 'vf_predictions.npz', grid=GRID, **arrays)
    assert sha(HERE / 'protocol.json') == frozen
    for p, h in hashes.items():
        assert sha(Path(p)) == h, p
    for name, df in [('d2', d2), ('vf', r)]:
        means = df.dropna(subset=['line_threshold', 'scrum_threshold']).groupby(['line_threshold', 'scrum_threshold']).mean(numeric_only=True).reset_index()
        means.to_csv(HERE / f'{name}_mean.csv', index=False)
    info = dict(complete=True, protocol_sha256=frozen, grid_size=2601, seeds=SEEDS, d2_rows=8796, vf_rows=688, seed_mean_not_ensemble=True, threshold_grid_fixed_before_vf=True, vf_hashes={str(P / f'seed_{s}/vf_views.npy'): hashes[str(P / f'seed_{s}/vf_views.npy')] for s in SEEDS})
    (HERE / 'completion.json').write_text(json.dumps(info, indent=2))
    print('DONE 2601 combinations x 3 seeds x D2/VF')
if __name__ == '__main__':
    main()
