import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = experiment_directory(__file__)
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
R = ROOT / 'data/m0_protect_scrum_rescue_line_d2_vf_20261027'
labels = ['line', 'other', 'scrum']
seeds = [17, 42, 2026]

def choose(v, mode):
    base = v[:, 0].argmax(1)
    votes = v[:, 1:].argmax(2) == 0
    high = votes & (v[:, 1:, 0] >= 0.9)
    yes = high.sum(1) >= 2 if mode == 'dos90' else (high.sum(1) >= 1) & (votes.sum(1) >= 2)
    return np.where((base == 1) & yes, 0, base)

def stats(y, a):
    pr, re, f, _ = precision_recall_fscore_support(y, a, labels=[0, 1, 2], zero_division=0)
    return dict(line_precision=pr[0], line_recall=re[0], line_f1=f[0], line_fp=int(sum((a == 0) & (y != 0))), scrum_precision=pr[2], scrum_recall=re[2], scrum_f1=f[2], scrum_fp=int(sum((a == 2) & (y != 2))))

def run_d2():
    data = json.loads(_resolve_model_artifact(ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json').read_text())
    idx = _load_model_result_array(P / 'train_indices.npy')
    y = np.array([labels.index(data[i]['tag']) for i in idx])
    rows = []
    for s in seeds:
        v = _load_model_result_array(P / f'seed_{s}/oof.npy')
        b = v[:, 0].argmax(1)
        for name in ['central', 'dos90', 'uno90_mas_uno']:
            a = b if name == 'central' else choose(v, name)
            rows.append(dict(seed=s, model=name, **stats(y, a), corrected=int(sum((b != y) & (a == y))), broken=int(sum((b == y) & (a != y)))))
    pd.DataFrame(rows).to_csv(H / 'd2_results.csv', index=False)
    (H / 'protocol.json').write_text(json.dumps({'rule': 'central other -> line only when >=2 context views vote line by argmax, at least one with p(line)>=.90; central line and scrum preserved; no minimum probability for the second vote', 'prior': 'two contexts each with p(line)>=.90', 'data': 'existing M0 cost4 predictions; OOF D2 and 688 VF blocks; seeds 17,42,2026', 'caveat': 'VF repeatedly used; exploratory, not independent; scores not calibrated'}, indent=2), encoding='utf8')

def run_vf():
    assert _resolve_model_artifact(H / 'protocol.json').exists()
    src = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    prev = pd.read_csv(_resolve_model_artifact(R / 'vf_predictions.csv'))
    rows = []
    events = []
    predictions = []
    for s in seeds:
        v = _load_model_result_array(P / f'seed_{s}/vf_views.npy')
        base = v[:, 0].argmax(1)
        y = np.array([labels.index(x) for x in src.tag])
        q = prev[prev.seed == s].reset_index(drop=True)
        assert q.clip_id.equals(src.clip_id)
        assert np.array_equal(choose(v, 'dos90'), q['t_0.9'])
        out = src[['clip_id', 'section', 'start_s', 'end_s', 'tag']].copy()
        out['seed'] = s
        for name in ['central', 'dos90', 'uno90_mas_uno']:
            a = base if name == 'central' else choose(v, name)
            assert np.array_equal(a == 2, base == 2)
            out[name] = a
            rows.append(dict(seed=s, model=name, **stats(y, a), corrected=int(sum((base != y) & (a == y))), broken=int(sum((base == y) & (a != y)))))
            f = src[['clip_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'tag']].copy()
            f['y_true'] = f.tag
            f['y_final'] = [labels[i] for i in a]
            for j, lab in enumerate(labels):
                f['p_' + lab] = v[:, 0, j]
            if name != 'central':
                changed = a != base
                votes = v[:, 1:].argmax(2) == 0
                score = np.sort(np.where(votes, v[:, 1:, 0], 0), axis=1)[:, -2]
                f.loc[changed, 'p_line'] = score[changed]
            for lab in ['line', 'scrum']:
                real = events_from_blocks(f, lab, prediction_column='y_true', probability_column='p_' + lab)
                det = events_from_blocks(f, lab, prediction_column='y_final', probability_column='p_' + lab)
                n = len(match_events(real, det))
                events.append(dict(seed=s, model=name, label=lab, real=len(real), detected=len(det), matched=n, unmatched=len(det) - n, precision=n / len(det) if det else 0, recall=n / len(real) if real else 0))
        predictions.append(out)
    pd.DataFrame(rows).to_csv(H / 'vf_results.csv', index=False)
    pd.DataFrame(events).to_csv(H / 'events.csv', index=False)
    pd.concat(predictions).to_csv(H / 'predictions.csv', index=False)
    print('BLOCKS')
    print(pd.DataFrame(rows).groupby('model').mean(numeric_only=True).to_string())
    print('EVENTS')
    print(pd.DataFrame(events).groupby(['model', 'label']).mean(numeric_only=True).to_string())
run_d2()
run_vf()
