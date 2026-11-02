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
import pandas as pd, numpy as np
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
inventory = json.loads(_resolve_model_artifact(ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022/inventory.json').read_text())
frames = [pd.read_csv(_resolve_model_artifact(s['probability_path'])) for s in inventory['sources']]
f = pd.concat(frames, ignore_index=True)
p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
labels = np.array(['line', 'other', 'scrum'])
rules = [('original', 0.5000000000000001, 0.4500000000000001), ('argmax', None, None)] + [(f'line_{t:.2f}', t, 'argmax') for t in [0.58, 0.45, 0.35, 0.25, 0.15]]
rows = []
for policy, line, scrum in rules:
    if policy == 'original':
        y = m.apply_class_thresholds(p, line=line, scrum=scrum)
    elif policy == 'argmax':
        y = p.argmax(1)
    else:
        y = np.where(p.argmax(1) == 2, 2, np.where(p[:, 0] >= line, 0, 1))
    d = f.assign(y_final=labels[y])
    for c, k in [('line', 0), ('scrum', 2)]:
        truth = m.events_from_blocks(d, c, prediction_column='y_true', probability_column='p_' + c)
        pred = m.events_from_blocks(d, c, prediction_column='y_final', probability_column='p_' + c)
        matched = {j for _, j, _, _ in m.match_events(truth, pred)}
        for j, e in enumerate(pred):
            if j in matched:
                continue
            ids = d.index[d.section.eq(e.section) & d.block_index.isin(e.blocks)].to_numpy()
            idx = int(ids[np.argmax(d.loc[ids, 'p_' + c].to_numpy())])
            near = [t for t in truth if t.section == e.section]
            distance = min((abs(t.anchor - e.anchor) for t in near), default=float('inf'))
            rows.append(dict(fp_id=f'{policy}_{c}_{e.section}_{j}', policy=policy, line_rule=line, scrum_rule=scrum, label=c, section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor, anchor_block_id=d.loc[idx, 'block_id'], truth_at_anchor=d.loc[idx, 'y_true'], nearest_truth_distance_s=distance, fp_type='Duplicado' if distance <= 35 else 'Sin evento real a <=35 s', blocks=','.join(map(str, e.blocks))))
r = pd.DataFrame(rows)
r.to_csv(HERE / 'fp_inventory.csv', index=False)
anchors = f[f.block_id.isin(r.anchor_block_id)].copy()
anchors['original_fp'] = anchors.block_id.isin(r[r.policy.eq('original')].anchor_block_id)
anchors.sort_values(['original_fp', 'section', 'block_index'], ascending=[False, True, True]).to_csv(HERE / 'anchors.csv', index=False)
print(r.groupby(['policy', 'label']).size().to_string())
print('Unique anchors:', len(anchors), 'original:', int(anchors.original_fp.sum()))
