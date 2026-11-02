"""Compare event-verifier feature groups with one fixed logistic head."""
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
from sklearn.base import clone
from models.M3.studies.line_cached_improvement_suite_20260818.run_stability_tests import select_event_policy_fixed_model
from models.M3.studies.line_cached_improvement_suite_20260818.run_suite import SECTIONS, accepted_events, compact_metric, event_dataset, event_estimators, event_metric, load_sections
OUTPUT = experiment_directory(__file__) / 'outputs'
FEATURE_GROUPS = {'official_only': ('official',), 'historical_static': ('historical', 'static'), 'static_temporal': ('static', 'temporal'), 'historical_static_temporal': ('historical', 'static', 'temporal')}

def compact_event_vector(section: dict, event: dict, branches: tuple[str, ...]) -> np.ndarray:
    start, stop = (event['first'], event['stop'])
    probability = np.concatenate([section[name][start:stop] for name in branches], axis=1)
    summaries = [probability.mean(axis=0), probability.std(axis=0), probability.min(axis=0), probability.max(axis=0), np.quantile(probability, 0.25, axis=0), np.quantile(probability, 0.5, axis=0), np.quantile(probability, 0.75, axis=0)]
    before = section['official'][max(0, start - 2):start]
    after = section['official'][stop:min(len(section['official']), stop + 2)]
    context = np.concatenate([before.mean(axis=0) if len(before) else np.zeros(3), after.mean(axis=0) if len(after) else np.zeros(3)])
    curve = section['official'][start:stop, 0]
    shape = np.asarray([stop - start, (stop - start) * 7.0, float(np.mean(np.abs(np.diff(curve)))) if len(curve) > 1 else 0.0, float(curve.max()), float(curve.min()), float(curve.mean())])
    return np.concatenate([shape, context, *summaries]).astype(np.float32)

def feature_dataset(section: dict, branches: tuple[str, ...]) -> dict:
    base = event_dataset(section)
    base['x'] = np.asarray([compact_event_vector(section, event, branches) for event in base['events']], dtype=np.float32)
    return base

def main() -> None:
    data = load_sections()
    definitions = event_estimators()
    definition = definitions['logreg_c0.1']
    datasets = {name: {section: feature_dataset(data[section], branches) for section in SECTIONS} for name, branches in FEATURE_GROUPS.items()}
    datasets['full_208'] = {section: event_dataset(data[section]) for section in SECTIONS}
    results = {}
    fold_rows = []
    for feature_name, event_data in datasets.items():
        values = []
        for holdout in SECTIONS:
            train = [name for name in SECTIONS if name != holdout]
            policy = select_event_policy_fixed_model(data, event_data, train, definition)
            fitted = clone(definition).fit(np.concatenate([event_data[name]['x'] for name in train]), np.concatenate([event_data[name]['y'] for name in train]))
            scores = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
            metric = compact_metric(event_metric(data[holdout]['frame'], accepted_events(event_data[holdout], scores, policy)))
            values.append(metric)
            fold_rows.append({'feature_group': feature_name, 'holdout': holdout, 'policy': {'threshold': policy.threshold, 'protected_anchor': policy.protected_anchor}, 'metrics': metric})
        results[feature_name] = {'feature_count': int(next(iter(event_data.values()))['x'].shape[1]), 'recall_mean': float(np.mean([value['recall'] for value in values])), 'recall_min': float(np.min([value['recall'] for value in values])), 'precision_mean': float(np.mean([value['precision'] for value in values])), 'false_positives_per_hour_mean': float(np.mean([value['false_positives_per_hour'] for value in values])), 'tp_total': int(sum((value['tp'] for value in values))), 'fp_total': int(sum((value['fp'] for value in values))), 'fn_total': int(sum((value['fn'] for value in values)))}
    feasible = [name for name, value in results.items() if value['recall_min'] + 1e-12 >= 0.9]
    best = min(feasible, key=lambda name: (results[name]['false_positives_per_hour_mean'], -results[name]['precision_mean'], results[name]['feature_count']))
    result = {'protocol': {'fixed_head': 'standardized logistic regression C=0.1', 'outer_validation': 'leave-one-section-out', 'purpose': 'post-hoc feature-group comparison', 'sections_from_same_match': True}, 'results': results, 'folds': fold_rows, 'feasible_feature_groups': feasible, 'best_feature_group': best}
    (OUTPUT / 'feature_group_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(result)
    print(json.dumps({'best': best, 'results': results}, indent=2, ensure_ascii=False))

def write_report(result: dict) -> None:
    lines = ['# ComparaciÃ³n de grupos de caracterÃ­sticas', '', '| CaracterÃ­sticas | DimensiÃ³n | Recall medio | Recall mÃ­nimo | PrecisiÃ³n | FP/h | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in sorted(result['results'].items(), key=lambda item: (item[1]['recall_min'] >= 0.9, -item[1]['false_positives_per_hour_mean']), reverse=True):
        lines.append(f"| {name} | {value['feature_count']} | {100 * value['recall_mean']:.1f}% | {100 * value['recall_min']:.1f}% | {100 * value['precision_mean']:.1f}% | {value['false_positives_per_hour_mean']:.1f} | {value['tp_total']} | {value['fp_total']} | {value['fn_total']} |")
    lines.extend(['', f"Mejor grupo bajo la restricciÃ³n de recall: `{result['best_feature_group']}`.", '', 'La comparaciÃ³n es de desarrollo y todavÃ­a requiere otro partido.'])
    (OUTPUT / 'FEATURE_GROUP_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
