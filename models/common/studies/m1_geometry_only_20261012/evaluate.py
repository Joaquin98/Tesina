"""Paired six-fold held-out ablation: saved M1 geometry RF, no refitting."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import runtime as rt
from runtime import HERE, ROOT, save, sha
import json, sys, time, statistics
import numpy as np, pandas as pd, joblib
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import block_metrics, temporal_metrics, apply_class_thresholds, threshold_search, normalize_probabilities
from models.common.comparison.protocol import LABELS, LABEL_TO_ID, select_run
from models.common.comparison.data import split_for_run
BASE = ROOT / 'data/final_model_comparison_20260827/outputs'

def metric_frame(frame, probabilities, thresholds):
    out = frame.copy()
    out['y_true'] = out['label']
    out['y_final'] = [LABELS[i] for i in apply_class_thresholds(probabilities, line=thresholds['line'], scrum=thresholds['scrum'])]
    for i, label in enumerate(LABELS):
        out['p_' + label] = probabilities[:, i]
    return out

def compute(frame, probabilities, thresholds):
    return {'block': block_metrics(frame.label.map(LABEL_TO_ID).to_numpy(), probabilities), 'temporal': temporal_metrics(metric_frame(frame, probabilities, thresholds))}

def geometry_matrix(frame):
    values = []
    for row in frame.itertuples(index=False):
        path = BASE / 'prepared_final' / f'{row.block_id}.npz'
        assert _resolve_model_artifact(path).is_file(), path
        with _load_model_result_array(path, allow_pickle=False) as data:
            values.append(data['geometry'].astype(np.float32))
    x = np.stack(values)
    assert x.shape == (len(frame), 549)
    return x

def main():
    started = time.perf_counter()
    manifest = json.loads(_resolve_model_artifact(HERE / 'inputs/source_manifest.json').read_text(encoding='utf-8'))
    master = pd.read_csv(_resolve_model_artifact(BASE / 'master_examples.csv'))
    folds = []
    paired = []
    references = {}
    for fold in manifest['folds']:
        print('FOLD_START ' + fold['fold'], flush=True)
        run = ROOT / Path(fold['geometry_rf']).parent
        refs = [ROOT / fold['geometry_rf'], run / 'predictions.csv', run / 'metrics.json', run / 'run.json']
        references.update({str(p): sha(p) for p in refs})
        meta = json.loads(_resolve_model_artifact(run / 'run.json').read_text(encoding='utf-8'))
        assert meta['smoke_limit'] is None
        train, development, test = split_for_run(master, select_run(fold['run_id']))
        assert development.group_id.nunique() == test.group_id.nunique() == 1
        assert not set(test.group_id) & set(development.group_id)
        saved = pd.read_csv(_resolve_model_artifact(run / 'predictions.csv')).set_index('block_id').loc[test.block_id].reset_index()
        assert saved.y_true.tolist() == test.label.tolist()
        forest = joblib.load(_resolve_model_artifact(ROOT / fold['geometry_rf']))
        assert forest.n_features_in_ == 549
        assert list(forest.classes_) == [0, 1, 2]
        geometry_test = forest.predict_proba(geometry_matrix(test))
        cached = saved[['p_geometry_' + label for label in LABELS]].to_numpy()
        error = float(np.abs(geometry_test - cached).max())
        assert np.allclose(geometry_test, cached, atol=1e-12, rtol=0), error
        geometry_dev = forest.predict_proba(geometry_matrix(development))
        dev_frame = development.copy()
        dev_frame['y_true'] = dev_frame['label']
        for i, label in enumerate(LABELS):
            dev_frame['p_' + label] = geometry_dev[:, i]
        t = time.perf_counter()
        selected = threshold_search(dev_frame, geometry_dev)
        selection_seconds = time.perf_counter() - t
        thresholds = {'line': selected['line'], 'scrum': selected['scrum']}
        original = saved[['p_' + label for label in LABELS]].to_numpy()
        full = compute(test, original, meta['thresholds'])
        reference = json.loads(_resolve_model_artifact(run / 'metrics.json').read_text(encoding='utf-8'))
        assert abs(full['temporal']['weighted_f1_macro'] - reference['temporal']['weighted_f1_macro']) < 1e-12
        assert abs(full['block']['f1_macro'] - reference['block']['f1_macro']) < 1e-12
        original_tau = {'line': meta['thresholds']['line'], 'scrum': meta['thresholds']['scrum']}
        record = {'fold': fold['fold'], 'test_group': fold['test_group'], 'development_group': fold['development_group'], 'n_test': len(test), 'n_development': len(development), 'geometry_thresholds_development_only': thresholds, 'threshold_selection_seconds': selection_seconds, 'saved_geometry_probability_max_difference': error, 'full_M1': full, 'geometry_frozen_full_thresholds': compute(test, geometry_test, original_tau), 'geometry_recalibrated_development': compute(test, geometry_test, thresholds)}
        save('results/' + fold['fold'] + '_metrics.json', record)
        folds.append(record)
        out = metric_frame(test, geometry_test, thresholds)
        out['fold'] = fold['fold']
        for label in LABELS:
            out['full_p_' + label] = saved['p_' + label].to_numpy()
        out['full_y_final'] = saved.y_final.to_numpy()
        paired.append(out)
        print('FOLD_COMPLETE ' + json.dumps({'fold': fold['fold'], 'full_f1w': full['temporal']['weighted_f1_macro'], 'geometry_f1w': record['geometry_recalibrated_development']['temporal']['weighted_f1_macro'], 'thresholds': thresholds}), flush=True)
    combined = pd.concat(paired, ignore_index=True)
    assert not combined.block_id.duplicated().any()
    combined.to_csv(HERE / 'results/heldout_predictions.csv', index=False)
    summary = {}
    for name in ['full_M1', 'geometry_frozen_full_thresholds', 'geometry_recalibrated_development']:
        summary[name] = {'temporal_f1_macro_equal_fold_mean': float(np.mean([f[name]['temporal']['weighted_f1_macro'] for f in folds])), 'block_f1_macro_equal_fold_mean': float(np.mean([f[name]['block']['f1_macro'] for f in folds])), 'accuracy_equal_fold_mean': float(np.mean([f[name]['block']['accuracy'] for f in folds])), **{f'temporal_f1_{label}_equal_fold_mean': float(np.mean([f[name]['temporal'][label]['weighted_f1'] for f in folds])) for label in ['line', 'scrum']}, 'far_false_positives_per_hour_equal_fold_mean': float(np.mean([sum((f[name]['temporal'][label]['far_false_positives_per_hour'] for label in ['line', 'scrum'])) for f in folds]))}
    thresholds = {label: statistics.median((f['geometry_thresholds_development_only'][label] for f in folds)) for label in ['line', 'scrum']}
    after = {path: sha(Path(path)) for path in references}
    assert references == after
    output = {'source_configuration': 'M1-H2-D2', 'variant': 'YOLO_geometry_RF_only', 'protocol': 'paired_outer_test_per_fold; branch thresholds selected exclusively on that fold development group', 'no_refitting': True, 'uses_RGB_features': False, 'uses_cached_geometry_for_quality_evaluation': True, 'full_video_timing_must_reextract_YOLO': True, 'fold_count': 6, 'total_heldout_blocks': len(combined), 'summary': summary, 'folds': folds, 'video_thresholds_median_development_folds': thresholds, 'video_ensemble_not_independently_accuracy_validated': True, 'hashes_unchanged': True, 'reference_hashes': after, 'evaluation_seconds': time.perf_counter() - started}
    save('results/evaluation.json', output)
    print('EVALUATION_COMPLETE ' + json.dumps(summary), flush=True)
from pathlib import Path
if __name__ == '__main__':
    main()
