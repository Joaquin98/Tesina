"""Stability-focused follow-up for the cached three-section line suite."""
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
from dataclasses import asdict
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from models.M3.studies.line_cached_improvement_suite_20260818.run_suite import EventPolicy, FusionPolicy, SECTIONS, accepted_events, aggregate, baseline_metrics, candidate_events, clone, compact_metric, event_dataset, event_estimators, event_metric, fusion_events, fusion_grid, load_sections
OUTPUT = experiment_directory(__file__) / 'outputs'

def select_fusion_from(data: dict, train_sections: list[str], policies: list[FusionPolicy]) -> FusionPolicy:
    candidates = []
    for policy in policies:
        values = [compact_metric(event_metric(data[name]['frame'], fusion_events(data[name], policy))) for name in train_sections]
        candidates.append((policy, min((value['recall'] for value in values)), float(np.mean([value['false_positives_per_hour'] for value in values])), float(np.mean([value['precision'] for value in values]))))
    feasible = [candidate for candidate in candidates if candidate[1] + 1e-12 >= 0.9]
    pool = feasible or candidates
    return max(pool, key=lambda value: (-value[2] if feasible else value[1], value[3], value[0].threshold))[0]

def policy_families() -> dict[str, list[FusionPolicy]]:
    complete = list(fusion_grid())
    return {'fixed_40_60': [policy for policy in complete if np.isclose(policy.historical_weight, 0.4) and np.isclose(policy.static_weight, 0.6) and np.isclose(policy.temporal_weight, 0.0)], 'constrained_static': [policy for policy in complete if 0.3 <= policy.historical_weight <= 0.5 and policy.temporal_weight <= 0.1 + 1e-12], 'no_temporal': [policy for policy in complete if np.isclose(policy.temporal_weight, 0.0)]}

def select_event_policy_fixed_model(data: dict, event_data: dict, train_sections: list[str], definition) -> EventPolicy:
    oof = {}
    for holdout in train_sections:
        fit_section = next((name for name in train_sections if name != holdout))
        fitted = clone(definition).fit(event_data[fit_section]['x'], event_data[fit_section]['y'])
        oof[holdout] = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
    joined = np.concatenate([oof[name] for name in train_sections])
    thresholds = np.unique(np.r_[np.arange(0.01, 0.951, 0.01), np.quantile(joined, np.linspace(0.02, 0.98, 97))])
    candidates = []
    for threshold in thresholds:
        for anchor in (0.75, 0.85, 0.95, 1.01):
            policy = EventPolicy(float(threshold), float(anchor))
            values = [compact_metric(event_metric(data[name]['frame'], accepted_events(event_data[name], oof[name], policy))) for name in train_sections]
            candidates.append((policy, min((value['recall'] for value in values)), float(np.mean([value['false_positives_per_hour'] for value in values])), float(np.mean([value['precision'] for value in values]))))
    feasible = [candidate for candidate in candidates if candidate[1] + 1e-12 >= 0.9]
    pool = feasible or candidates
    return max(pool, key=lambda value: (-value[2] if feasible else value[1], value[3], -value[0].threshold))[0]

def main() -> None:
    data = load_sections()
    event_data = {name: event_dataset(section) for name, section in data.items()}
    families = policy_families()
    definitions = event_estimators()
    methods: dict[str, list[dict]] = {'baseline': []}
    methods.update({name: [] for name in families})
    for name in definitions:
        methods[f'event_{name}'] = []
    selections = []
    for holdout in SECTIONS:
        train = [name for name in SECTIONS if name != holdout]
        base = baseline_metrics(data[holdout])
        methods['baseline'].append({'recall': base['event_recall'], 'precision': base['event_precision'], 'false_positives_per_hour': base['event_false_positives_per_hour'], 'tp': base['event_tp'], 'fp': base['event_fp'], 'fn': base['event_fn']})
        for family_name, policies in families.items():
            selected = select_fusion_from(data, train, policies)
            value = compact_metric(event_metric(data[holdout]['frame'], fusion_events(data[holdout], selected)))
            methods[family_name].append(value)
            selections.append({'holdout': holdout, 'method': family_name, 'selection': asdict(selected), 'metrics': value})
        for model_name, definition in definitions.items():
            policy = select_event_policy_fixed_model(data, event_data, train, definition)
            fitted = clone(definition).fit(np.concatenate([event_data[name]['x'] for name in train]), np.concatenate([event_data[name]['y'] for name in train]))
            scores = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
            value = compact_metric(event_metric(data[holdout]['frame'], accepted_events(event_data[holdout], scores, policy)))
            methods[f'event_{model_name}'].append(value)
            selections.append({'holdout': holdout, 'method': f'event_{model_name}', 'selection': asdict(policy), 'metrics': value})
    aggregate_rows = {}
    for name, values in methods.items():
        aggregate_rows[name] = {'recall_mean': float(np.mean([value['recall'] for value in values])), 'recall_min': float(np.min([value['recall'] for value in values])), 'precision_mean': float(np.mean([value['precision'] for value in values])), 'false_positives_per_hour_mean': float(np.mean([value['false_positives_per_hour'] for value in values])), 'tp_total': int(sum((value['tp'] for value in values))), 'fp_total': int(sum((value['fp'] for value in values))), 'fn_total': int(sum((value['fn'] for value in values)))}
    feasible = [name for name, value in aggregate_rows.items() if name != 'baseline' and value['recall_min'] + 1e-12 >= 0.9 and (value['false_positives_per_hour_mean'] < aggregate_rows['baseline']['false_positives_per_hour_mean'])]
    best = min(feasible, key=lambda name: (aggregate_rows[name]['false_positives_per_hour_mean'], -aggregate_rows[name]['precision_mean'])) if feasible else None
    stable_model_name = 'event_logreg_c0.1'
    stable_definition = definitions[stable_model_name.removeprefix('event_')]
    stable_policy = EventPolicy(threshold=0.11, protected_anchor=1.01)
    stable_by_section = {}
    for holdout in SECTIONS:
        train = [name for name in SECTIONS if name != holdout]
        fitted = clone(stable_definition).fit(np.concatenate([event_data[name]['x'] for name in train]), np.concatenate([event_data[name]['y'] for name in train]))
        scores = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
        stable_by_section[holdout] = compact_metric(event_metric(data[holdout]['frame'], accepted_events(event_data[holdout], scores, stable_policy)))
    stable_summary = {'model': stable_model_name, 'policy': asdict(stable_policy), 'by_section': stable_by_section, 'recall_min': float(min((value['recall'] for value in stable_by_section.values()))), 'recall_mean': float(np.mean([value['recall'] for value in stable_by_section.values()])), 'precision_mean': float(np.mean([value['precision'] for value in stable_by_section.values()])), 'false_positives_per_hour_mean': float(np.mean([value['false_positives_per_hour'] for value in stable_by_section.values()])), 'status': 'pending_different_match_validation', 'must_not_deploy_yet': True, 'selection_note': 'Architecture fixed after stability comparison; threshold is the minimum selected in the three outer folds.'}
    stable_final = clone(stable_definition).fit(np.concatenate([event_data[name]['x'] for name in SECTIONS]), np.concatenate([event_data[name]['y'] for name in SECTIONS]))
    joblib.dump(stable_final, OUTPUT / 'stable_event_logreg_c0_1_pending_external_match.joblib')
    (OUTPUT / 'stable_event_logreg_c0_1_pending_external_match.json').write_text(json.dumps(stable_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    result = {'protocol': {'purpose': 'post-hoc stability comparison after the unrestricted suite', 'outer_validation': 'leave-one-section-out', 'sections_from_same_match': True, 'winner_requires': 'recall >= 90% in every section and fewer FP/h than baseline'}, 'aggregate': aggregate_rows, 'selections': selections, 'feasible_methods': feasible, 'best_feasible_method': best, 'stable_candidate_pending_external_match': stable_summary}
    (OUTPUT / 'stability_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(result)
    print(json.dumps({'best': best, 'feasible': feasible, 'aggregate': aggregate_rows}, indent=2, ensure_ascii=False))

def write_report(result: dict) -> None:
    ranked = sorted(result['aggregate'], key=lambda name: (result['aggregate'][name]['recall_min'] >= 0.9, -result['aggregate'][name]['false_positives_per_hour_mean']), reverse=True)
    lines = ['# ComparaciÃ³n de estabilidad', '', '| MÃ©todo | Recall medio | Recall mÃ­nimo | PrecisiÃ³n | FP/h | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ranked:
        value = result['aggregate'][name]
        lines.append(f"| {name} | {100 * value['recall_mean']:.1f}% | {100 * value['recall_min']:.1f}% | {100 * value['precision_mean']:.1f}% | {value['false_positives_per_hour_mean']:.1f} | {value['tp_total']} | {value['fp_total']} | {value['fn_total']} |")
    lines.extend(['', f"Mejor mÃ©todo que cumple la restricciÃ³n: `{result['best_feasible_method']}`.", '', 'Esta comparaciÃ³n sirve para orientar el desarrollo. El mÃ©todo elegido todavÃ­a requiere un partido diferente porque la familia de modelos fue comparada usando estos tres tramos.'])
    (OUTPUT / 'STABILITY_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
