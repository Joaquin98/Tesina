from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
from pathlib import Path
import pandas as pd
from models.M0.studies.independent_evaluation.audit_protocol import keras_validation_overlap
from models.M0.studies.independent_evaluation.protocol import EXPERIMENT_DIR, OUTPUT_DIR, SPLIT_DIR, assert_disjoint, load_candidates

def test_historical_validation_overlap_is_reproduced():
    overlap, summary = keras_validation_overlap()
    assert summary['overlap_unique_clips'] == 92
    assert summary['overlap_by_class']['scrum'] == 92
    assert overlap.video_name.nunique() == 92

def test_candidates_were_frozen_before_outer_evaluation():
    payload = json.loads(_resolve_model_artifact(EXPERIMENT_DIR / 'candidates.json').read_text(encoding='utf-8'))
    assert payload['frozen_before_outer_evaluation'] is True
    assert len(load_candidates()) >= 4

def test_all_generated_splits_are_group_disjoint():
    for fold_dir in sorted(_model_artifact_glob(SPLIT_DIR, 'outer_*', recursive=False)):
        parts = {name: pd.read_csv(_resolve_model_artifact(fold_dir / f'{name}.csv')) for name in ('train', 'validation', 'evaluation')}
        findings = assert_disjoint(parts)
        assert all((item['clip_overlap'] == 0 for item in findings))
        assert all((item['group_overlap'] == 0 for item in findings))

def test_smoke_run_wrote_required_artifacts():
    root = OUTPUT_DIR / 'integration_smoke' / 'outer_01'
    required = [root / 'inner_candidate_selection.csv', root / 'selected_candidate.json', root / 'outer_predictions.csv', root / 'outer_metrics.json', root / 'confusion_matrix.csv', root / 'fold_summary.json']
    assert all((_resolve_model_artifact(path).exists() for path in required))

def test_completed_pilot_has_two_disjoint_outer_folds():
    root = OUTPUT_DIR / 'nested_evaluation_pilot_2fold_e12_manual'
    summary = json.loads(_resolve_model_artifact(root / 'summary.json').read_text(encoding='utf-8'))
    predictions = pd.read_csv(_resolve_model_artifact(root / 'all_outer_predictions.csv'))
    assert summary['completed_outer_folds'] == [1, 2]
    assert summary['candidate_count'] == 2
    assert summary['evaluated_unique_clips'] == 1634
    assert predictions.video_name.nunique() == len(predictions) == 1634
    group_sets = [set(predictions.loc[predictions.outer_fold == fold, 'source_group']) for fold in (1, 2)]
    assert not group_sets[0].intersection(group_sets[1])
    assert 'pooled_group_bootstrap_95' in summary
