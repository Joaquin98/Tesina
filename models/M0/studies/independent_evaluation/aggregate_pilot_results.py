"""Aggregate completed pilot folds without re-running either outer evaluation."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import OUTPUT_DIR, SEED, stable_seed, write_json
from models.M0.studies.independent_evaluation.run_nested_evaluation import group_bootstrap
ROOT = OUTPUT_DIR / 'nested_evaluation_pilot_2fold_e12_manual'

def main() -> None:
    rows = []
    predictions = []
    for fold in (1, 2):
        fold_root = ROOT / f'outer_{fold:02d}'
        summary = json.loads(_resolve_model_artifact(fold_root / 'fold_summary.json').read_text(encoding='utf-8'))
        rows.append({'outer_fold': fold, 'selected_candidate': summary['selected_candidate'], 'train_clips': summary['train_clips'], 'validation_clips': summary['validation_clips'], 'evaluation_clips': summary['evaluation_clips'], 'accuracy': summary['accuracy'], 'f1_macro': summary['f1_macro'], 'f1_weighted': summary['f1_weighted'], 'recall_line': summary['per_class']['line']['recall'], 'recall_other': summary['per_class']['other']['recall'], 'recall_scrum': summary['per_class']['scrum']['recall']})
        prediction = pd.read_csv(_resolve_model_artifact(fold_root / 'outer_predictions.csv'))
        prediction['outer_fold'] = fold
        predictions.append(prediction)
    flat = pd.DataFrame(rows)
    combined = pd.concat(predictions, ignore_index=True)
    if combined.video_name.duplicated().any():
        raise AssertionError('A clip was evaluated in more than one pilot outer fold')
    fold_groups = [set(frame.source_group) for frame in predictions]
    if fold_groups[0] & fold_groups[1]:
        raise AssertionError('A source group was evaluated in both pilot outer folds')
    flat.to_csv(ROOT / 'fold_metrics.csv', index=False)
    combined.to_csv(ROOT / 'all_outer_predictions.csv', index=False)
    metrics = ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')
    summary = {'mode': 'pilot', 'completed_outer_folds': [1, 2], 'candidate_count': 2, 'epochs_cap': 12, 'evaluated_unique_clips': int(combined.video_name.nunique()), 'evaluated_source_groups': int(combined.source_group.nunique()), 'metrics_mean': {name: float(flat[name].mean()) for name in metrics}, 'metrics_standard_deviation': {name: float(flat[name].std(ddof=1)) for name in metrics}, 'pooled_group_bootstrap_95': group_bootstrap(combined, 2000, stable_seed(SEED, 'pilot_pooled_bootstrap')), 'warning': 'Reduced two-fold nested grouped estimate. It is valid for these folds, but uncertainty is larger than in the planned five-fold evaluation.'}
    write_json(ROOT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
