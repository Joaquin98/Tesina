"""Compare grouped nested historical baseline and RGB+YOLO candidate."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def main():
    b = []
    for i in range(1, 6):
        p = REPO / f'data/independent_evaluation/outputs/nested_evaluation/outer_{i:02d}/outer_metrics.json'
        j = json.loads(_resolve_model_artifact(p).read_text())
        b.append({'fold': i, 'accuracy': j['accuracy'], 'f1_macro': j['f1_macro']})
    c = json.loads(_resolve_model_artifact(HERE / 'nested_group_candidate_summary.json').read_text())['folds']
    rows = []
    for base, cand in zip(b, c):
        rows.append({'fold': base['fold'], 'historical': base, 'rgb_yolo': cand['evaluation'], 'delta_accuracy': cand['evaluation']['accuracy'] - base['accuracy'], 'delta_f1_macro': cand['evaluation']['f1_macro'] - base['f1_macro']})
    out = {'folds': rows, 'historical_mean_accuracy': float(np.mean([x['historical']['accuracy'] for x in rows])), 'candidate_mean_accuracy': float(np.mean([x['rgb_yolo']['accuracy'] for x in rows])), 'historical_mean_f1_macro': float(np.mean([x['historical']['f1_macro'] for x in rows])), 'candidate_mean_f1_macro': float(np.mean([x['rgb_yolo']['f1_macro'] for x in rows])), 'mean_delta_accuracy': float(np.mean([x['delta_accuracy'] for x in rows])), 'mean_delta_f1_macro': float(np.mean([x['delta_f1_macro'] for x in rows])), 'generalization_gate': '>= +0.05 absolute accuracy on all/grouped folds'}
    (HERE / 'group_comparison_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
