import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, itertools, time
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
f = pd.DataFrame([x for x in json.loads(_resolve_model_artifact(M).read_text()) if x['domain'] == 'VF' and x['split'] == 'test'])
y = f.tag.map({x: i for i, x in enumerate(L)}).to_numpy()
assert len(y) == 688
assert json.loads(_resolve_model_artifact(A / 'verification.json').read_text())['passed'] and json.loads(_resolve_model_artifact(B / 'verification.json').read_text())['passed']
start = time.time()
out = []
configs = list(itertools.combinations(range(1, 9), 4))
assert len(configs) == 70
for directory, rate in [(A, 'todos'), (B, 'uno_de_cinco')]:
    p = np.stack([_load_model_result_array(directory / 'blocks' / f'{i:04d}.npy') for i in range(688)])
    assert p.shape == (688, 6, 9, 3)
    for combo in configs:
        for j in range(6):
            q = p[:, j]
            base = q[:, 0].argmax(1)
            v = q[:, list(combo)]
            votes = v.argmax(2) == 0
            high = votes & (v[:, :, 0] >= 0.5)
            a = np.where((base == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1), 0, base)
            assert np.array_equal(a == 2, base == 2)
            pr, re, f1, _ = precision_recall_fscore_support(y, a, labels=[0, 1, 2], zero_division=0)
            z = f.copy()
            z['y_true'] = z.tag
            z['y_final'] = [L[x] for x in a]
            z['p_line'] = q[:, 0, 0]
            score = np.sort(np.where(votes, v[:, :, 0], 0), axis=1)[:, -2]
            z.loc[a != base, 'p_line'] = score[a != base]
            truth = events_from_blocks(z, 'line', prediction_column='y_true', probability_column='p_line')
            found = events_from_blocks(z, 'line', prediction_column='y_final', probability_column='p_line')
            n = len(match_events(truth, found))
            out.append(dict(sampling=rate, views='-'.join(map(str, combo)), fold=j + 1, block_precision=pr[0], block_recall=re[0], block_f1=f1[0], block_fp=int(sum((a == 0) & (y != 0))), scrum_precision=pr[2], scrum_recall=re[2], event_true=len(truth), event_matched=n, event_unmatched=len(found) - n, event_precision=n / len(found) if found else 0, event_recall=n / len(truth) if truth else 0, event_f1=2 * n / (len(truth) + len(found)) if len(truth) + len(found) else 0))
        print(rate, combo, flush=True)
df = pd.DataFrame(out)
df.to_csv(H / 'fold_results.csv', index=False)
summary = df.groupby(['sampling', 'views'], as_index=False).mean(numeric_only=True).drop(columns=['fold'])
summary.to_csv(H / 'summary.csv', index=False)
for rate in ['todos', 'uno_de_cinco']:
    print('TOP by F1', rate)
    print(summary[summary.sampling == rate].sort_values(['event_f1', 'event_recall'], ascending=False).head(8).to_string(index=False))
    print('TOP recall', rate)
    print(summary[summary.sampling == rate].sort_values(['event_recall', 'event_precision'], ascending=False).head(5).to_string(index=False))
(H / 'protocol.json').write_text(json.dumps({'combinations': 70, 'total_views': 5, 'central': 0, 'context_options': list(range(1, 9)), 'decision': 'central other becomes line if >=2 of the 4 contexts vote line and >=1 has p>=0.5; scrum central preserved', 'data': 'existing 6 M0 checkpoints and 688 previously used VF blocks, both sampling rates', 'selection': 'all results and rankings computed on VF; post-hoc exploratory; not independent selection', 'events': 'consecutive blocks; one-to-one matching by center/anchor <=35s; unmatched not visually verified'}, indent=2))
(H / 'completion.json').write_text(json.dumps({'complete': True, 'configurations': 140, 'fold_evaluations': len(df), 'elapsed_seconds': time.time() - start}, indent=2))
