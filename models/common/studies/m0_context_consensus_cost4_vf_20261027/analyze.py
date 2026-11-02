"""Exploratory retrospective: central fallback, context-only consensus."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
sys.path.insert(0, str(source_directory(BASE)))
from evaluate import bootstrap
LABELS = ['line', 'other', 'scrum']

def decide(v, k):
    central = v[:, 0]
    line = (v[:, 1:] == 0).sum(1)
    scrum = (v[:, 1:] == 2).sum(1)
    return np.where((line < k) & (scrum < k), central, np.where((scrum >= k) & ((line < k) | (scrum >= line)), 2, 0))

def main():
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    y = np.array([LABELS.index(t) for t in frame.tag])
    assert len(y) == 688 and np.bincount(y).tolist() == [68, 581, 39]
    results = []
    details = []
    preds = {x: [] for x in [0, 2, 3, 4]}
    hashes = {}
    for seed in [17, 42, 2026]:
        path = P / f'seed_{seed}/vf_views.npy'
        hashes[str(path)] = hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()
        a = _load_model_result_array(path)
        assert a.shape == (688, 9, 3)
        np.testing.assert_allclose(a[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        v = a.argmax(2)
        central = v[:, 0]
        for k in preds:
            pred = central if k == 0 else decide(v, k)
            preds[k].append(pred)
            pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
            cm = confusion_matrix(y, pred, labels=[0, 1, 2])
            row = dict(seed=seed, variant='central' if k == 0 else f'consenso_{k}', accuracy=accuracy_score(y, pred), f1_macro=f.mean(), corrected=int(((central != y) & (pred == y)).sum()), broken=int(((central == y) & (pred != y)).sum()))
            for j, c in enumerate(LABELS):
                row.update({c + '_precision': pr[j], c + '_recall': re[j], c + '_f1': f[j], c + '_tp': int(cm[j, j]), c + '_fn': int(cm[j].sum() - cm[j, j]), c + '_fp': int(cm[:, j].sum() - cm[j, j])})
            results.append(row)
        for i in range(len(y)):
            details.append(dict(seed=seed, clip_id=frame.clip_id.iloc[i], section=frame.section.iloc[i], start_s=frame.start_s.iloc[i], human=LABELS[y[i]], central=LABELS[central[i]], line_context_votes=int((v[i, 1:] == 0).sum()), scrum_context_votes=int((v[i, 1:] == 2).sum()), other_context_votes=int((v[i, 1:] == 1).sum()), **{f'consenso_{k}': LABELS[preds[k][-1][i]] for k in [2, 3, 4]}))
    df = pd.DataFrame(results)
    df.to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(details).to_csv(HERE / 'predictions.csv', index=False)
    boot = {f'consenso_{k}_vs_central': bootstrap(y, preds[0], preds[k], frame.section, frame.start_s) for k in [2, 3, 4]}
    result = dict(complete=True, means=df.groupby('variant').mean(numeric_only=True).to_dict(orient='index'), bootstrap=boot, hashes=hashes, method='eight contexts vote independently; >=k same event overrides central; highest eligible vote wins, scrum wins ties', threshold='vote count k, no probability threshold', limitations='VF repeatedly observed; k=2,3,4 exploratory; correlated views; seeds not ensemble, CH provenance incomplete')
    (HERE / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
if __name__ == '__main__':
    main()
