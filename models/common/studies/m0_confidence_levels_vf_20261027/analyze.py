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
import json, sys, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
P = R / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
BASE = R / 'data/central_recall_cost_d2_vf_20261027'
STRICT = R / 'data/m0_strict_context_d2_vf_20261027'
sys.path.insert(0, str(source_directory(STRICT)))
from rule import decide, GRID
labels = ['line', 'other', 'scrum']
seeds = [17, 42, 2026]

def main():
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    y = np.array([labels.index(x) for x in frame.tag])
    assert len(y) == 688 and np.bincount(y).tolist() == [68, 581, 39]
    records = []
    detail = []
    hashes = {}
    for seed in seeds:
        path = P / f'seed_{seed}/vf_views.npy'
        hashes[str(path)] = hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()
        p = _load_model_result_array(path)
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        central = p[:, 0].argmax(1)
        for t in [None] + GRID.tolist():
            pred = central if t is None else decide(p, [t, t])
            pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
            cm = confusion_matrix(y, pred, labels=[0, 1, 2])
            d = dict(seed=seed, variant='central' if t is None else f'{int(round(t * 100))}%', threshold=t, accuracy=accuracy_score(y, pred), f1_macro=f.mean(), changes=int((central != pred).sum()), corrected=int(((central != y) & (pred == y)).sum()), broken=int(((central == y) & (pred != y)).sum()))
            for j, k in enumerate(labels):
                d.update({k + '_precision': pr[j], k + '_recall': re[j], k + '_f1': f[j], k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
            records.append(d)
            for i in range(len(y)):
                detail.append(dict(seed=seed, threshold='central' if t is None else f'{int(round(t * 100))}%', clip_id=frame.clip_id.iloc[i], human=labels[y[i]], central=labels[central[i]], predicted=labels[pred[i]], changed=bool(central[i] != pred[i])))
    df = pd.DataFrame(records)
    df.to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(detail).to_csv(HERE / 'predictions.csv', index=False)
    means = df.groupby('variant').mean(numeric_only=True)
    data = dict(complete=True, thresholds=GRID.tolist(), rule='central default; at least two of eight contexts vote same event with probability >= level; if both qualify highest second confidence, scrum ties', same_level_both_classes=True, means=means.to_dict(orient='index'), input_hashes=hashes, limitations=['VF already examined repeatedly; results exploratory; not independent', 'grid compared with VF after previous analysis; no threshold selection on VF', 'context probabilities are not verified calibrated', 'seed averages not ensemble'])
    (HERE / 'results.json').write_text(json.dumps(data, indent=2))
    print(df.groupby('variant').mean(numeric_only=True)[['line_precision', 'line_recall', 'line_fp', 'scrum_precision', 'scrum_recall', 'scrum_fp', 'changes', 'corrected', 'broken', 'f1_macro']].to_string())
if __name__ == '__main__':
    main()
