import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
H = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
A = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
B = ROOT / 'data/m0_h2_d2_stride5_cached_vf_20261028'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
L = ['line', 'other', 'scrum']
assert json.loads(_resolve_model_artifact(H / 'verification.json').read_text())['passed']
frame = pd.DataFrame([x for x in json.loads(_resolve_model_artifact(M).read_text()) if x['domain'] == 'VF' and x['split'] == 'test'])
y = frame.tag.map({k: i for i, k in enumerate(L)}).to_numpy()
p = np.stack([_load_model_result_array(A / 'blocks' / f'{i:04d}.npy') for i in range(688)])
q = np.stack([_load_model_result_array(B / 'blocks' / f'{i:04d}.npy') for i in range(688)])
records = []
events = []
predout = []
for j in range(6):
    c = p[:, j, 0].argmax(1)
    v = q[:, j, 1:]
    votes = v.argmax(2) == 0
    out = frame[['clip_id', 'tag', 'section', 'start_s', 'end_s']].copy()
    out['fold'] = j + 1
    for th in [0, 0.35, 0.4, 0.45, 0.5]:
        high = votes & (v[:, :, 0] >= th)
        pred = np.where((c == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1), 0, c)
        assert np.array_equal(pred == 2, c == 2)
        out['t_' + str(th)] = pred
        pr, rc, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        records.append({'fold': j + 1, 'threshold': th, 'line_precision': pr[0], 'line_recall': rc[0], 'line_f1': f[0], 'line_tp': int(sum((y == 0) & (pred == 0))), 'line_fp': int(sum((y != 0) & (pred == 0))), 'scrum_precision': pr[2], 'scrum_recall': rc[2]})
        data = frame.copy()
        data['y_true'] = data.tag
        data['y_final'] = [L[x] for x in pred]
        data['p_line'] = p[:, j, 0, 0]
        data['p_scrum'] = p[:, j, 0, 2]
        strength = np.sort(np.where(votes, v[:, :, 0], 0), axis=1)[:, -2]
        data.loc[pred != c, 'p_line'] = strength[pred != c]
        for label in ['line', 'scrum']:
            true = events_from_blocks(data, label, prediction_column='y_true', probability_column='p_' + label)
            found = events_from_blocks(data, label, prediction_column='y_final', probability_column='p_' + label)
            n = len(match_events(true, found))
            events.append({'fold': j + 1, 'threshold': th, 'label': label, 'true_events': len(true), 'matched': n, 'unmatched': len(found) - n, 'precision': n / len(found) if found else 0, 'recall': n / len(true) if true else 0, 'f1': 2 * n / (len(true) + len(found)) if len(true) + len(found) else 0})
    predout.append(out)
pd.DataFrame(records).to_csv(H / 'lower_threshold_blocks.csv', index=False)
pd.DataFrame(events).to_csv(H / 'lower_threshold_events.csv', index=False)
pd.concat(predout).to_csv(H / 'lower_threshold_predictions.csv', index=False)
print('BLOCKS')
print(pd.DataFrame(records).groupby('threshold').mean(numeric_only=True).to_string())
print('EVENTS')
print(pd.DataFrame(events).groupby(['threshold', 'label']).mean(numeric_only=True).to_string())
