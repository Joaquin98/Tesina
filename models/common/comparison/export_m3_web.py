"""Build web manifests for the two selected M3 cross-fold ensembles.

The manifests only point at immutable artifacts produced by the completed
Panel-V runs.  They do not retrain or copy checkpoints.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
from pathlib import Path
import numpy as np
EXPERIMENT_DIR = experiment_directory(__file__)
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
RUNS_DIR = EXPERIMENT_DIR / 'outputs' / 'runs'
SHARED_DYNAMIC_PCA = PROJECT_ROOT / 'data' / 'joint_multimodal_20260813' / 'rgb_dynamics_seed42' / 'dynamic_pca.joblib'

def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

def _latest_completed_attempt(run_id: str) -> Path:
    required = ('run.json', 'metrics.json', 'temporal.pt', 'line_verifier.joblib')
    for attempt in sorted(_model_artifact_glob(RUNS_DIR / run_id, 'attempt_*', recursive=False), reverse=True):
        if all((_resolve_model_artifact(attempt / name).is_file() for name in required)):
            return attempt
    raise FileNotFoundError(f'No completed web-compatible attempt for {run_id}')

def _build(level: str, rank: int) -> Path:
    model_name = f'm3_{level.lower()}_d2_cross_fold_ensemble'
    folds = []
    metric_rows = []
    for fold_number in range(1, 7):
        fold_name = f'PV{fold_number:02d}'
        run_id = f'PV-M3-{level}-D2-{fold_name}-S42'
        attempt = _latest_completed_attempt(run_id)
        metadata = json.loads(_resolve_model_artifact(attempt / 'run.json').read_text(encoding='utf-8'))
        metrics = json.loads(_resolve_model_artifact(attempt / 'metrics.json').read_text(encoding='utf-8'))
        static_dir = attempt / 'static'
        artifact_paths = {Path(path).name: Path(path) for path in metadata['artifacts']}
        if level == 'H2':
            dynamic_pca = static_dir / 'dynamic_pca.joblib'
            scaler = static_dir / 'quantile_scaler.joblib'
            logistic = static_dir / 'logistic.joblib'
            historical = static_dir / 'historical.keras'
        else:
            dynamic_pca = SHARED_DYNAMIC_PCA
            scaler = artifact_paths['quantile_scaler.joblib']
            logistic = artifact_paths['logistic.joblib']
            historical = artifact_paths['best_model.keras']
        required = (dynamic_pca, scaler, logistic, historical, attempt / 'temporal.pt', attempt / 'line_verifier.joblib')
        missing = [str(path) for path in required if not _resolve_model_artifact(path).is_file()]
        if missing:
            raise FileNotFoundError(f'Missing artifacts for {run_id}: {missing}')
        folds.append({'fold': fold_name, 'run_id': run_id, 'attempt': attempt.name, 'test_group': metadata['test_group'], 'development_group': metadata['development_group'], 'dynamic_pca': _relative(dynamic_pca), 'quantile_scaler': _relative(scaler), 'logistic': _relative(logistic), 'historical_model': _relative(historical), 'temporal_model': _relative(attempt / 'temporal.pt'), 'line_verifier': _relative(attempt / 'line_verifier.joblib'), 'static_alpha': float(metadata['model_metadata']['static_alpha']), 'line_threshold': float(metadata['thresholds']['line']), 'scrum_threshold': float(metadata['thresholds']['scrum']), 'line_verifier_threshold': float(metadata['thresholds']['line_verifier']), 'checkpoint_hash': metadata['checkpoint_hash']})
        temporal = metrics['temporal']
        metric_rows.append({'f1w_macro': float(temporal['weighted_f1_macro']), 'f1w_line': float(temporal['line']['weighted_f1']), 'f1w_scrum': float(temporal['scrum']['weighted_f1']), 'fp_far_hour': float(temporal['line']['far_false_positives_per_hour'] + temporal['scrum']['far_false_positives_per_hour'])})
    manifest = {'schema_version': 1, 'model_name': model_name, 'source_configuration': f'PV-M3-{level}-D2', 'comparison_rank': rank, 'deployment_policy': 'equal_soft_vote_across_six_outer_folds', 'event_verifier_policy': 'majority_vote_of_fold_specific_line_verifiers', 'validation_status': 'components_cross_validated_web_ensemble_not_independently_validated', 'label_order': ['line', 'other', 'scrum'], 'pose_model': 'yolo11n-pose.pt', 'recommended_web_thresholds': {'line': float(np.median([fold['line_threshold'] for fold in folds])), 'scrum': float(np.median([fold['scrum_threshold'] for fold in folds])), 'source': 'median_of_development_selected_fold_thresholds'}, 'cross_validation_metrics': {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}, 'folds': folds}
    export_dir = EXPERIMENT_DIR / 'outputs' / f'web_export_m3_{level.lower()}_d2'
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / 'manifest.json'
    destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return destination

def main() -> None:
    print(_build('H2', rank=2))
    print(_build('H1', rank=3))
if __name__ == '__main__':
    main()
