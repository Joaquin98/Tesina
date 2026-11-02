import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, numpy as np, pandas as pd
H = experiment_directory(__file__)
A = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_h2_d2_allframes_nine_vf_20261027')
D = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026/manifest.json')
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_allframes_nine_rule_sweep_vf_20261028')
assert json.loads(_resolve_model_artifact(H / 'verification.json').read_text())['passed'] and json.loads(_resolve_model_artifact(H / 'cache_verification.json').read_text())['passed']
rows = [r for r in json.loads(_resolve_model_artifact(D).read_text()) if r['domain'] == 'VF' and r['split'] == 'test']
df = pd.DataFrame(rows)
truth = df.tag.eq('line').to_numpy()
assert sum(truth) == 68
old = np.stack([_load_model_result_array(A / 'blocks' / f'{i:04d}.npy') for i in range(688)])
new = np.stack([_load_model_result_array(H / 'blocks' / f'{i:04d}.npy') for i in range(688)])
assert old.shape == new.shape == (688, 6, 9, 3)

def eval_probs(p):
    central = p[:, 0].argmax(1)
    vote = p[:, 1:].argmax(2) == 0
    high = vote & (p[:, 1:, 0] >= 0.5)
    rescue = (central == 1) & (vote.sum(1) >= 2) & (high.sum(1) >= 1)
    combined = np.where(rescue, 0, central)
    return (central, rescue, combined, vote.sum(1), high.sum(1))
records = []
details = []
for fold in range(6):
    ca, ra, pa, va, ha = eval_probs(old[:, fold])
    cb, rb, pb, vb, hb = eval_probs(new[:, fold])
    la = truth & (pa == 0)
    lb = truth & (pb == 0)
    d = {'fold': fold + 1, 'line_blocks': int(sum(truth)), 'all_central_line': int(sum(truth & (ca == 0))), 'sparse_central_line': int(sum(truth & (cb == 0))), 'all_central_other': int(sum(truth & (ca == 1))), 'sparse_central_other': int(sum(truth & (cb == 1))), 'all_central_scrum': int(sum(truth & (ca == 2))), 'sparse_central_scrum': int(sum(truth & (cb == 2))), 'all_rescued': int(sum(truth & ra)), 'sparse_rescued': int(sum(truth & rb)), 'all_final_correct': int(sum(la)), 'sparse_final_correct': int(sum(lb)), 'both_final_correct': int(sum(la & lb)), 'all_only_final_correct': int(sum(la & ~lb)), 'sparse_only_final_correct': int(sum(lb & ~la)), 'both_final_wrong': int(sum(truth & ~la & ~lb)), 'lost_central_was_line': int(sum(la & ~lb & (ca == 0))), 'lost_central_was_rescued': int(sum(la & ~lb & ra)), 'lost_both_central_other_only_all_rescues': int(sum(la & ~lb & (ca == 1) & (cb == 1) & ra & ~rb)), 'central_all_yes_sparse_no': int(sum(truth & (ca == 0) & (cb != 0))), 'central_sparse_yes_all_no': int(sum(truth & (cb == 0) & (ca != 0)))}
    assert d['all_only_final_correct'] == d['lost_central_was_line'] + d['lost_central_was_rescued']
    records.append(d)
    for i in np.flatnonzero(truth):
        details.append({'clip_id': df.iloc[i].clip_id, 'section': df.iloc[i].section, 'start_s': df.iloc[i].start_s, 'end_s': df.iloc[i].end_s, 'fold': fold + 1, 'central_all': int(ca[i]), 'central_1of5': int(cb[i]), 'rescue_all': bool(ra[i]), 'rescue_1of5': bool(rb[i]), 'final_all': int(pa[i]), 'final_1of5': int(pb[i]), 'votes_all': int(va[i]), 'votes_1of5': int(vb[i]), 'votes50_all': int(ha[i]), 'votes50_1of5': int(hb[i])})
rd = pd.DataFrame(records)
detail = pd.DataFrame(details)
rd.to_csv(H / 'line_cause_by_fold.csv', index=False)
detail.to_csv(H / 'line_cause_by_block.csv', index=False)
means = rd.mean(numeric_only=True)
print('FOLD')
print(rd.to_string(index=False))
print('MEANS')
print(means.to_string())
lost = detail[(detail.final_all == 0) & (detail.final_1of5 != 0)]
print('COMMON LOST')
print(lost.groupby(['clip_id', 'section', 'start_s', 'end_s']).size().sort_values(ascending=False).head(10).to_string())
print('NEW GAINS')
print(detail[(detail.final_all != 0) & (detail.final_1of5 == 0)].groupby(['clip_id', 'section', 'start_s', 'end_s']).size().sort_values(ascending=False).head(5).to_string())
(H / 'line_cause_verification.json').write_text(json.dumps({'passed': True, 'true_line_blocks': 68, 'folds': 6, 'expected_aggregate_observations': 408, 'reconciled_loss_categories': True, 'unit': 'block, not independent event', 'mean': means.to_dict()}, indent=2), encoding='utf8')
