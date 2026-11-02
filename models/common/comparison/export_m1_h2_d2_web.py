"""Build the manifest consumed by the web M1-H2-D2 cross-fold ensemble."""
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
EXPORT_DIR = EXPERIMENT_DIR / 'outputs' / 'web_export_m1_h2_d2'

def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

def _latest_completed_attempt(run_id: str) -> Path:
    for attempt in sorted(_model_artifact_glob(RUNS_DIR / run_id, 'attempt_*', recursive=False), reverse=True):
        required = ('run.json', 'metrics.json', 'dual.keras', 'geometry_rf.joblib')
        if all((_resolve_model_artifact(attempt / name).is_file() for name in required)):
            return attempt
    raise FileNotFoundError(f'No completed web-compatible attempt for {run_id}')

def main() -> None:
    folds = []
    metric_rows = []
    for fold_number in range(1, 7):
        fold = f'PV{fold_number:02d}'
        run_id = f'PV-M1-H2-D2-{fold}-S42'
        attempt = _latest_completed_attempt(run_id)
        metadata = json.loads(_resolve_model_artifact(attempt / 'run.json').read_text(encoding='utf-8'))
        metrics = json.loads(_resolve_model_artifact(attempt / 'metrics.json').read_text(encoding='utf-8'))
        temporal = metrics['temporal']
        folds.append({'fold': fold, 'run_id': run_id, 'attempt': attempt.name, 'test_group': metadata['test_group'], 'development_group': metadata['development_group'], 'dual_model': _relative(attempt / 'dual.keras'), 'geometry_rf': _relative(attempt / 'geometry_rf.joblib'), 'geometry_alpha': float(metadata['model_metadata']['geometry_alpha']), 'line_threshold': float(metadata['thresholds']['line']), 'scrum_threshold': float(metadata['thresholds']['scrum']), 'checkpoint_hash': metadata['checkpoint_hash']})
        metric_rows.append({'f1w_macro': float(temporal['weighted_f1_macro']), 'f1w_line': float(temporal['line']['weighted_f1']), 'f1w_scrum': float(temporal['scrum']['weighted_f1']), 'fp_far_hour': float(temporal['line']['far_false_positives_per_hour'] + temporal['scrum']['far_false_positives_per_hour'])})
    line_thresholds = np.asarray([item['line_threshold'] for item in folds], dtype=float)
    scrum_thresholds = np.asarray([item['scrum_threshold'] for item in folds], dtype=float)
    manifest = {'schema_version': 1, 'model_name': 'm1_h2_d2_cross_fold_ensemble', 'source_configuration': 'PV-M1-H2-D2', 'deployment_policy': 'equal_soft_vote_across_six_outer_folds', 'validation_status': 'components_cross_validated_web_ensemble_not_independently_validated', 'label_order': ['line', 'other', 'scrum'], 'rgb_sequence_shape': [200, 2048], 'pose_sequence_shape': [200, 62], 'geometry_feature_count': 549, 'pose_model': 'yolo11n-pose.pt', 'recommended_web_thresholds': {'line': float(np.median(line_thresholds)), 'scrum': float(np.median(scrum_thresholds)), 'source': 'median_of_development_selected_fold_thresholds'}, 'cross_validation_metrics': {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}, 'folds': folds}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = EXPORT_DIR / 'manifest.json'
    destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(destination)
if __name__ == '__main__':
    main()
