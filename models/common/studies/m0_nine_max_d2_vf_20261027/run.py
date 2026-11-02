"""Evaluate frozen central M0 separately per view; choose maximum softmax."""
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
import hashlib, json, sys, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
PARENT = ROOT / 'data/central_preserving_d2_vf_20261026'
sys.path.insert(0, str(source_directory(ROOT / 'data/central_recall_cost_d2_vf_20261027')))
from analysis_math import metrics
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def select(logits):
    assert logits.ndim == 3 and logits.shape[1:] == (9, 3) and np.isfinite(logits).all()
    e = np.exp(logits - logits.max(axis=2, keepdims=True))
    p = e / e.sum(axis=2, keepdims=True)
    index = p.reshape(len(p), 27).argmax(axis=1)
    return (p, index // 3, index % 3)

def run():
    rows = json.loads(_resolve_model_artifact(PARENT / 'manifest.json').read_text())
    idx = np.array([i for i, r in enumerate(rows) if r['domain'] == 'VF'])
    assert len(idx) == 688
    frame = pd.DataFrame([rows[i] for i in idx])
    y = np.array([LABELS.index(v) for v in frame.tag])
    assert np.bincount(y).tolist() == [68, 581, 39]
    allpred = {v: [] for v in ['central', 'nueve_max']}
    records = []
    hashes = {}
    output = []
    for seed in SEEDS:
        cache = PARENT / f'embeddings/seed_{seed}'
        meta = json.loads(_resolve_model_artifact(cache / 'complete.json').read_text())
        identity = json.loads(_resolve_model_artifact(cache / 'identity.json').read_text())
        assert identity['row_indices'] == list(range(len(rows)))
        path = cache / 'logits.npy'
        assert sha(path) == meta['files']['logits.npy']
        hashes[str(path)] = sha(path)
        logits = _load_model_result_array(path, mmap_mode='r')[idx]
        p, view, pred = select(logits)
        base = p[:, 0].argmax(axis=1)
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(PARENT / f'seed_{seed}/central/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        assert np.all(p[np.arange(len(y)), view, pred] >= p[:, 0].max(axis=1) - 1e-07)
        for variant, result in [('central', base), ('nueve_max', pred)]:
            m = metrics(y, result)
            pr, re, f, _ = precision_recall_fscore_support(y, result, labels=[0, 1, 2], zero_division=0)
            cm = confusion_matrix(y, result, labels=[0, 1, 2])
            np.testing.assert_allclose(m['accuracy'], accuracy_score(y, result))
            np.testing.assert_allclose([m[k + '_recall'] for k in LABELS], re)
            counts = {k + '_' + name: int(val) for j, k in enumerate(LABELS) for name, val in [('tp', cm[j, j]), ('fn', cm[j].sum() - cm[j, j]), ('fp', cm[:, j].sum() - cm[j, j])]}
            records.append(dict(seed=seed, variant=variant, **m, **counts))
            allpred[variant].append(result)
        for i in range(len(y)):
            output.append(dict(seed=seed, clip_id=frame.clip_id.iloc[i], section=frame.section.iloc[i], start_s=frame.start_s.iloc[i], y_true=LABELS[y[i]], central=LABELS[base[i]], chosen_view=int(view[i]), prediction=LABELS[pred[i]], confidence=float(p[i, view[i], pred[i]]), **{f'view{v}_p_{k}': float(p[i, v, j]) for v in range(9) for j, k in enumerate(LABELS)}))
    result = pd.DataFrame(records)
    result.to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(output).to_csv(HERE / 'predictions.csv', index=False)
    sections = frame.section.to_numpy()
    bins = (frame.start_s.to_numpy() // 60).astype(int)
    groups = [[np.flatnonzero((sections == s) & (bins == b)) for b in np.unique(bins[sections == s])] for s in np.unique(sections)]
    keys = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'f1_macro', 'accuracy']
    rng = np.random.default_rng(20261027)
    deltas = []
    for _ in range(2000):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
        if any((np.sum(y[ix] == c) == 0 for c in range(3))):
            continue
        deltas.append([np.mean([metrics(y[ix], n[ix])[k] - metrics(y[ix], b[ix])[k] for b, n in zip(allpred['central'], allpred['nueve_max'])]) for k in keys])
    intervals = {k: dict(difference=float(np.mean([metrics(y, n)[k] - metrics(y, b)[k] for b, n in zip(allpred['central'], allpred['nueve_max'])])), ci95=np.quantile(np.asarray(deltas)[:, i], [0.025, 0.975]).tolist()) for i, k in enumerate(keys)}
    info = dict(passed=True, means=result.groupby('variant').mean(numeric_only=True).to_dict(orient='index'), intervals=intervals, valid_bootstraps=len(deltas), hashes=hashes, blocks=688, seeds=SEEDS, rule='argmax over view and class after per-view softmax', tie='first view then class', cost=1, model='same previously trained central M0 independently on nine views', training='reused; no training', sampling='1 Hz without interpolation', warning='VF previously observed; adaptive exploratory analysis, not independent test; mean seeds is not ensemble; CIs resample 60s clusters conditioned on fitted models')
    (HERE / 'results.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=True))
if __name__ == '__main__':
    run()
