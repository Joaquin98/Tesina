import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = experiment_directory(__file__)
A = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
B = ROOT / 'data/m0_h2_d2_stride5_cached_vf_20261028'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
L = ['line', 'other', 'scrum']
frame = pd.DataFrame([x for x in json.loads(_resolve_model_artifact(M).read_text()) if x['domain'] == 'VF' and x['split'] == 'test'])
truth = frame.tag.eq('line').to_numpy()
assert sum(truth) == 68
allp = np.stack([_load_model_result_array(A / 'blocks' / f'{i:04d}.npy') for i in range(688)])
sp = np.stack([_load_model_result_array(B / 'blocks' / f'{i:04d}.npy') for i in range(688)])
prev = pd.read_csv(_resolve_model_artifact(H / 'predictions.csv'))
blocks = []
evrows = []
evdetails = []

def eval(v, central):
    votes = v.argmax(2) == 0
    high = votes & (v[:, :, 0] >= 0.5)
    rescue = (central == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1)
    return (np.where(rescue, 0, central), votes.sum(1), high.sum(1), votes)
for j in range(6):
    p = allp[:, j]
    s = sp[:, j]
    c = p[:, 0].argmax(1)
    a, va, ha, aa = eval(p[:, 1:], c)
    b, vb, hb, bb = eval(s[:, 1:], c)
    q = prev[prev.fold == j + 1]
    assert np.array_equal(a, q.all_frames) and np.array_equal(b, q.hybrid)
    lost = truth & (a == 0) & (b != 0)
    gained = truth & (a != 0) & (b == 0)
    blocks.append({'fold': j + 1, 'all_correct': int(sum(truth & (a == 0))), 'hybrid_correct': int(sum(truth & (b == 0))), 'lost': int(sum(lost)), 'gained': int(sum(gained)), 'loss_zero_votes': int(sum(lost & (vb == 0))), 'loss_one_vote': int(sum(lost & (vb == 1))), 'loss_no_50': int(sum(lost & (vb >= 2) & (hb == 0)))})
    for i in np.flatnonzero(lost | gained):
        evdetails.append(dict(kind='block_lost' if lost[i] else 'block_gained', fold=j + 1, section=frame.iloc[i].section, start_s=frame.iloc[i].start_s, end_s=frame.iloc[i].end_s, central=L[c[i]], votes_all=int(va[i]), votes_hybrid=int(vb[i]), votes50_all=int(ha[i]), votes50_hybrid=int(hb[i]), clip_id=frame.iloc[i].clip_id))
    matched_sets = {}
    events_ref = None
    for name, pred, vote, values in [('all', a, aa, p), ('hybrid', b, bb, s)]:
        f = frame.copy()
        f['y_true'] = f.tag
        f['y_final'] = [L[i] for i in pred]
        f['p_line'] = p[:, 0, 0]
        strength = np.sort(np.where(vote, values[:, 1:, 0], 0), axis=1)[:, -2]
        f.loc[pred != c, 'p_line'] = strength[pred != c]
        t = events_from_blocks(f, 'line', prediction_column='y_true', probability_column='p_line')
        d = events_from_blocks(f, 'line', prediction_column='y_final', probability_column='p_line')
        m = match_events(t, d)
        matched_sets[name] = {i for i, j, dist, w in m}
        events_ref = t
    old = matched_sets['all']
    new = matched_sets['hybrid']
    evrows.append({'fold': j + 1, 'true_events': len(events_ref), 'all_matched': len(old), 'hybrid_matched': len(new), 'only_all': len(old - new), 'only_hybrid': len(new - old), 'both': len(old & new), 'neither': len(events_ref) - len(old | new)})
    for ix in sorted(old - new):
        e = events_ref[ix]
        evdetails.append({'kind': 'event_all_only', 'fold': j + 1, 'section': e.section, 'start_s': e.start, 'end_s': e.end, 'clip_id': '', 'central': '', 'votes_all': '', 'votes_hybrid': ''})
pd.DataFrame(blocks).to_csv(H / 'line_block_changes.csv', index=False)
pd.DataFrame(evrows).to_csv(H / 'line_event_changes.csv', index=False)
pd.DataFrame(evdetails).to_csv(H / 'line_change_examples.csv', index=False)
print('BLOCKS')
print(pd.DataFrame(blocks).to_string(index=False))
print('mean', pd.DataFrame(blocks).mean().to_dict())
print('EVENTS')
print(pd.DataFrame(evrows).to_string(index=False))
print('mean', pd.DataFrame(evrows).mean().to_dict())
print('block examples')
print(pd.DataFrame(evdetails).query("kind=='block_lost'").groupby(['section', 'start_s', 'end_s']).size().sort_values(ascending=False).head(8).to_string())
print('event examples')
print(pd.DataFrame(evdetails).query("kind=='event_all_only'").groupby(['section', 'start_s', 'end_s']).size().sort_values(ascending=False).head(8).to_string())
