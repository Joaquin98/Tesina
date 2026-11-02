import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
OUT = HERE / 'vf'
OUT.mkdir(exist_ok=True)
inv = json.loads(_resolve_model_artifact(ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022/inventory.json').read_text())
f = pd.concat([pd.read_csv(_resolve_model_artifact(x['probability_path'])) for x in inv['sources']], ignore_index=True)
p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
f['y_final'] = np.array(['line', 'other', 'scrum'])[p.argmax(1)]
selected = []
rows = []
for c in ['line', 'scrum']:
    tr = m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c)
    pr = m.events_from_blocks(f, c, prediction_column='y_final', probability_column='p_' + c)
    pairs = m.match_events(tr, pr)
    matched_p = {j: i for i, j, _, _ in pairs}
    matched_t = {i for i, _, _, _ in pairs}
    for status, events in [('prediction', pr), ('truth', tr)]:
        for j, e in enumerate(events):
            if status == 'truth' and j in matched_t:
                continue
            active = f[f.section.eq(e.section) & f.block_index.isin(e.blocks)]
            idx = active['p_' + c].idxmax()
            selected.append(idx)
            rows.append(dict(label=c, status=('TP' if j in matched_p else 'FP') if status == 'prediction' else 'FN', section=e.section, start_s=e.start, end_s=e.end, anchor_s=(f.loc[idx, 'start_s'] + f.loc[idx, 'end_s']) / 2, block_id=f.loc[idx, 'block_id'], anchor_kind='maximum fused class probability within predicted event' if status == 'prediction' else 'maximum fused class probability within missed true event', truth_at_anchor=f.loc[idx, 'y_true'], event_index=j))
pd.DataFrame(rows).to_csv(OUT / 'anchor_events.csv', index=False)
f.loc[sorted(set(selected))].to_csv(OUT / 'anchors.csv', index=False)
print(pd.DataFrame(rows).groupby(['label', 'status']).size().to_string())
print('Unique anchor blocks:', len(set(selected)))
