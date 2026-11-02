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
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = experiment_directory(__file__)
A = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
B = ROOT / 'data/m0_h2_d2_stride5_cached_vf_20261028'
manifest = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
assert json.loads(_resolve_model_artifact(A / 'verification.json').read_text())['passed'] and json.loads(_resolve_model_artifact(B / 'verification.json').read_text())['passed']
frame = pd.DataFrame([r for r in json.loads(_resolve_model_artifact(manifest).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test'])
y = frame.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
labels = ['line', 'other', 'scrum']
assert len(y) == 688
results = []
events = []
outputs = []
for folder, rate in [(A, 'todos'), (B, 'uno_de_cinco')]:
    probs = np.stack([_load_model_result_array(folder / 'blocks' / f'{i:04d}.npy') for i in range(688)])
    assert probs.shape == (688, 6, 9, 3)
    for fold in range(6):
        p = probs[:, fold]
        base = p[:, 0].argmax(1)
        out = frame[['clip_id', 'section', 'start_s', 'end_s', 'tag']].copy()
        out['fold'] = fold + 1
        out['sampling'] = rate
        for name, idx in [('central', []), ('cinco', [1, 4, 5, 8]), ('nueve', list(range(1, 9)))]:
            if len(idx):
                v = p[:, idx]
                vote = v.argmax(2) == 0
                high = vote & (v[:, :, 0] >= 0.5)
                pred = np.where((base == 1) & (vote.sum(1) >= 2) & (high.sum(1) >= 1), 0, base)
            else:
                pred = base
            assert np.array_equal(pred == 2, base == 2)
            out[name] = pred
            pr, rc, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
            results.append(dict(sampling=rate, fold=fold + 1, model=name, line_precision=pr[0], line_recall=rc[0], line_f1=f[0], line_fp=int(sum((y != 0) & (pred == 0))), scrum_precision=pr[2], scrum_recall=rc[2]))
            z = frame.copy()
            z['y_true'] = z.tag
            z['y_final'] = [labels[i] for i in pred]
            z['p_line'] = p[:, 0, 0]
            z['p_scrum'] = p[:, 0, 2]
            if len(idx):
                score = np.sort(np.where(vote, v[:, :, 0], 0), axis=1)[:, -2]
                z.loc[pred != base, 'p_line'] = score[pred != base]
            for lab in ['line', 'scrum']:
                true = events_from_blocks(z, lab, prediction_column='y_true', probability_column='p_' + lab)
                found = events_from_blocks(z, lab, prediction_column='y_final', probability_column='p_' + lab)
                n = len(match_events(true, found))
                events.append(dict(sampling=rate, fold=fold + 1, model=name, label=lab, true=len(true), matched=n, unmatched=len(found) - n, precision=n / len(found) if found else 0, recall=n / len(true) if true else 0, f1=2 * n / (len(true) + len(found)) if len(true) + len(found) else 0))
        outputs.append(out)
pd.DataFrame(results).to_csv(H / 'blocks.csv', index=False)
pd.DataFrame(events).to_csv(H / 'events.csv', index=False)
pd.concat(outputs).to_csv(H / 'predictions.csv', index=False)
(H / 'protocol.json').write_text(json.dumps({'selection': 'predetermined corners 1,4,5,8 plus central 0; spatial grid 2x4', 'rule': 'central other -> line when two contexts vote line, one with p>=0.5; scrum central unchanged', 'folds': 6, 'dataset': 'VF 688 previously observed; exploratory', 'sampling': 'all available 25fps observations vs one in five', 'source': 'existing per-view prediction arrays; no inference run or runtime measured'}, indent=2), encoding='utf8')
print('BLOCK')
print(pd.DataFrame(results).groupby(['sampling', 'model']).mean(numeric_only=True).to_string())
print('EVENTS')
print(pd.DataFrame(events).groupby(['sampling', 'model', 'label']).mean(numeric_only=True).to_string())
