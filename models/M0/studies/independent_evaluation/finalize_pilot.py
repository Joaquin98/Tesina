"""Select completed inner candidates and evaluate one untouched pilot outer fold."""
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
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import OUTPUT_DIR, SEED, SPLIT_DIR, assert_disjoint, stable_seed, write_json
from models.M0.studies.independent_evaluation.run_nested_evaluation import create_memmap_dataset, evaluate_selected
RESULT_ROOT = OUTPUT_DIR / 'nested_evaluation_pilot_2fold_e12_manual'
FEATURE_ROOT = OUTPUT_DIR / 'nested_evaluation_pilot_2fold' / '_feature_arrays'
CANDIDATE_IDS = ('principal_resnet50_bigru_u64', 'historical_val_leader_resnet50_bigru_u128')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer-fold', type=int, required=True, choices=(1, 2))
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    args = parser.parse_args()
    fold_number = args.outer_fold
    fold_dir = RESULT_ROOT / f'outer_{fold_number:02d}'
    candidate_results = []
    for candidate_id in CANDIDATE_IDS:
        path = fold_dir / 'candidates' / candidate_id / 'inner_result.json'
        if not _resolve_model_artifact(path).exists():
            raise FileNotFoundError(f'Pending inner candidate: {path}')
        candidate_results.append(json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8')))
    selected = sorted(candidate_results, key=lambda item: (-float(item['inner_validation_f1_macro']), -float(item['inner_validation_accuracy']), str(item['candidate_id'])))[0]
    pd.DataFrame(candidate_results).to_csv(fold_dir / 'inner_candidate_selection.csv', index=False)
    write_json(fold_dir / 'selected_candidate.json', selected)
    split_root = SPLIT_DIR / f'outer_{fold_number:02d}'
    train = pd.read_csv(_resolve_model_artifact(split_root / 'train.csv'))
    validation = pd.read_csv(_resolve_model_artifact(split_root / 'validation.csv'))
    evaluation = pd.read_csv(_resolve_model_artifact(split_root / 'evaluation.csv'))
    assert_disjoint({'train': train, 'validation': validation, 'evaluation': evaluation})
    evaluation_data = create_memmap_dataset(evaluation, str(selected['extractor']), FEATURE_ROOT / f'outer_{fold_number:02d}' / 'evaluation')
    metrics, predictions = evaluate_selected(selected, evaluation, fold_dir, args.bootstrap_draws, stable_seed(SEED, fold_number, 'bootstrap'), evaluation_data)
    predictions['outer_fold'] = fold_number
    predictions.to_csv(fold_dir / 'outer_predictions.csv', index=False)
    summary = {'outer_fold': fold_number, 'mode': 'pilot', 'selected_candidate': selected['candidate_id'], 'train_clips': len(train), 'train_groups': int(train.source_group.nunique()), 'validation_clips': len(validation), 'validation_groups': int(validation.source_group.nunique()), 'evaluation_clips': len(evaluation), 'evaluation_groups': int(evaluation.source_group.nunique()), 'epochs_executed': selected['epochs_executed'], 'training_seconds_selected': selected['duration_seconds'], 'all_candidate_training_seconds': sum((float(x['duration_seconds']) for x in candidate_results)), **{key: metrics[key] for key in ('accuracy', 'f1_macro', 'f1_weighted', 'per_class', 'bootstrap_95_group')}}
    write_json(fold_dir / 'fold_summary.json', summary)
    completed = []
    all_predictions = []
    for number in (1, 2):
        path = RESULT_ROOT / f'outer_{number:02d}' / 'fold_summary.json'
        prediction_path = RESULT_ROOT / f'outer_{number:02d}' / 'outer_predictions.csv'
        if _resolve_model_artifact(path).exists() and _resolve_model_artifact(prediction_path).exists():
            completed.append(json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8')))
            all_predictions.append(pd.read_csv(_resolve_model_artifact(prediction_path)))
    if len(completed) == 2:
        flat = pd.DataFrame([{'outer_fold': row['outer_fold'], 'selected_candidate': row['selected_candidate'], 'train_clips': row['train_clips'], 'validation_clips': row['validation_clips'], 'evaluation_clips': row['evaluation_clips'], 'accuracy': row['accuracy'], 'f1_macro': row['f1_macro'], 'f1_weighted': row['f1_weighted'], 'recall_line': row['per_class']['line']['recall'], 'recall_other': row['per_class']['other']['recall'], 'recall_scrum': row['per_class']['scrum']['recall']} for row in completed])
        flat.to_csv(RESULT_ROOT / 'fold_metrics.csv', index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(RESULT_ROOT / 'all_outer_predictions.csv', index=False)
        metric_columns = ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')
        pilot_summary = {'mode': 'pilot', 'completed_outer_folds': [1, 2], 'candidate_count': 2, 'epochs_cap': 12, 'metrics_mean': {col: float(flat[col].mean()) for col in metric_columns}, 'metrics_standard_deviation': {col: float(flat[col].std(ddof=1)) for col in metric_columns}, 'warning': 'Reduced two-fold nested grouped estimate. It is valid for these folds, but uncertainty is larger than in the planned five-fold evaluation.'}
        write_json(RESULT_ROOT / 'summary.json', pilot_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
