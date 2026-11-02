import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = ROOT / 'data/m0_one90_plus_vote_d2_vf_20261027'
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
manifest = json.loads(_resolve_model_artifact(ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json').read_text())
idx = _load_model_result_array(P / 'train_indices.npy')
fold = _load_model_result_array(P / 'folds.npy')
labels = ['line', 'other', 'scrum']
result = []
details = []
for seed in [17, 42, 2026]:
    v = _load_model_result_array(P / f'seed_{seed}/oof.npy')
    prev = pd.read_csv(_resolve_model_artifact(H / 'fold_predictions.csv'))
    q = prev[prev.seed == seed].set_index('index')
    for k in range(3):
        mask = np.array([int(fold[j]) == k and manifest[i]['domain'] == 'CV' for j, i in enumerate(idx)])
        ix = idx[mask]
        vv = v[mask]
        frame = pd.DataFrame([manifest[i] for i in ix])
        assert not frame.duplicated(['group_id', 'section', 'block_index']).any()
        frame['y_true'] = frame.tag
        fixed = vv[:, 0].argmax(1)
        assert np.array_equal(q.loc[ix, 'central'].to_numpy(), fixed)
        for name in ['central', 'dos90', 'uno90_mas_uno']:
            a = q.loc[ix, name].to_numpy()
            f = frame.copy()
            f['y_final'] = [labels[i] for i in a]
            for j, l in enumerate(labels):
                f['p_' + l] = vv[:, 0, j]
            if name != 'central':
                changed = a != fixed
                votes = vv[:, 1:].argmax(2) == 0
                score = np.sort(np.where(votes, vv[:, 1:, 0], 0), axis=1)[:, -2]
                f.loc[changed, 'p_line'] = score[changed]
            for lab in ['line', 'scrum']:
                truth = events_from_blocks(f, lab, prediction_column='y_true', probability_column='p_' + lab)
                pred = events_from_blocks(f, lab, prediction_column='y_final', probability_column='p_' + lab)
                mt = match_events(truth, pred)
                n = len(mt)
                result.append(dict(seed=seed, fold=k, model=name, label=lab, clips=len(ix), real=len(truth), detected=len(pred), matched=n, missed=len(truth) - n, unmatched=len(pred) - n, precision=n / len(pred) if pred else 0, recall=n / len(truth) if truth else 0))
                details.append(dict(seed=seed, fold=k, model=name, label=lab, matches=[{'truth': i, 'prediction': j, 'distance': d} for i, j, d, w in mt]))
pd.DataFrame(result).to_csv(H / 'fold_cv_events.csv', index=False)
(H / 'fold_cv_event_matches.json').write_text(json.dumps(details, indent=2), encoding='utf8')
print(pd.DataFrame(result).groupby(['fold', 'model', 'label'])[['clips', 'real', 'matched', 'unmatched', 'precision', 'recall']].mean().round(4).to_string())
