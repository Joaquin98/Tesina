"""Cached line-improvement suite over three fully tagged match sections.

Tests two directions without decoding video again:
1. robust historical/static/temporal fusion selected on two sections;
2. a verifier that classifies complete baseline events instead of windows.
All reported comparison metrics are outer leave-one-section-out results.
"""
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
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
OUTPUT = HERE / 'outputs'
PREVIOUS = ROOT / 'data/line_verifier_three_sections_20260818'
if str(PREVIOUS) not in sys.path:
    sys.path.insert(0, str(source_directory(PREVIOUS)))
from run_experiment import SECTIONS, baseline_metrics, load_sections, match_indices, metrics, predicted_events, runs, truth_events

@dataclass(frozen=True)
class FusionPolicy:
    historical_weight: float
    static_weight: float
    temporal_weight: float
    threshold: float
    margin: float
    require_winner: bool
    min_support_blocks: int

@dataclass(frozen=True)
class EventPolicy:
    threshold: float
    protected_anchor: float

def normalize(values: np.ndarray) -> np.ndarray:
    output = np.clip(np.asarray(values, dtype=float), 1e-09, None)
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)

def event_metric(frame: pd.DataFrame, events: list[dict]) -> dict:
    duration = float(frame['end_time'].max() - frame['start_time'].min())
    return metrics(truth_events(frame, 'line'), events, duration)

def compact_metric(value: dict) -> dict:
    return {'recall': float(value['recall']), 'precision': float(value['precision']), 'false_positives_per_hour': float(value['false_positives_per_hour']), 'tp': int(value['true_positive']), 'fp': int(value['false_positive']), 'fn': int(value['false_negative'])}

def aggregate(folds: list[dict], key: str) -> dict:
    values = [fold[key] for fold in folds]
    return {'recall_mean': float(np.mean([value['recall'] for value in values])), 'recall_min': float(np.min([value['recall'] for value in values])), 'precision_mean': float(np.mean([value['precision'] for value in values])), 'false_positives_per_hour_mean': float(np.mean([value['false_positives_per_hour'] for value in values])), 'tp_total': int(sum((value['tp'] for value in values))), 'fp_total': int(sum((value['fp'] for value in values))), 'fn_total': int(sum((value['fn'] for value in values)))}

def fusion_grid():
    weights = np.round(np.arange(0.0, 1.001, 0.1), 2)
    thresholds = np.round(np.arange(0.2, 0.901, 0.05), 2)
    for historical in weights:
        for static in weights:
            temporal = round(1.0 - float(historical) - float(static), 2)
            if temporal < -1e-09 or temporal > 1.0 + 1e-09:
                continue
            for threshold in thresholds:
                for margin in (-0.1, 0.0, 0.1):
                    for require_winner in (True, False):
                        if require_winner and margin < 0.0:
                            continue
                        for support in (1, 2):
                            yield FusionPolicy(float(historical), float(static), float(temporal), float(threshold), float(margin), require_winner, support)

def fusion_events(section: dict, policy: FusionPolicy) -> list[dict]:
    probability = normalize(policy.historical_weight * section['historical'] + policy.static_weight * section['static'] + policy.temporal_weight * section['temporal'])
    score = probability[:, 0]
    competitor = np.max(probability[:, 1:], axis=1)
    evidence = (score >= policy.threshold) & (score - competitor >= policy.margin)
    if policy.require_winner:
        evidence &= probability.argmax(axis=1) == 0
    if policy.min_support_blocks > 1:
        filtered = np.zeros(len(evidence), dtype=bool)
        for start, stop in runs(evidence):
            if stop - start >= policy.min_support_blocks:
                filtered[start:stop] = True
        evidence = filtered
    return predicted_events(section['frame'], evidence)

def select_fusion(data: dict[str, dict], train_sections: list[str]) -> tuple[FusionPolicy, dict]:
    candidates = []
    for policy in fusion_grid():
        by_section = {name: compact_metric(event_metric(data[name]['frame'], fusion_events(data[name], policy))) for name in train_sections}
        min_recall = min((value['recall'] for value in by_section.values()))
        mean_fp = float(np.mean([value['false_positives_per_hour'] for value in by_section.values()]))
        mean_precision = float(np.mean([value['precision'] for value in by_section.values()]))
        candidates.append((policy, by_section, min_recall, mean_fp, mean_precision))
    feasible = [candidate for candidate in candidates if candidate[2] + 1e-12 >= 0.9]
    pool = feasible or candidates
    selected = max(pool, key=lambda value: (-value[3] if feasible else value[2], value[4], value[0].threshold, -value[0].temporal_weight))
    return (selected[0], {'feasible': bool(feasible), 'by_section': selected[1], 'min_recall': selected[2], 'mean_false_positives_per_hour': selected[3]})

def candidate_events(section: dict) -> list[dict]:
    base_line = section['official'].argmax(axis=1) == 0
    starts = section['frame']['start_time'].to_numpy(float)
    ends = section['frame']['end_time'].to_numpy(float)
    return [{'start': float(starts[start]), 'end': float(ends[stop - 1]), 'first': int(start), 'stop': int(stop)} for start, stop in runs(base_line)]

def event_vector(section: dict, event: dict) -> np.ndarray:
    start, stop = (event['first'], event['stop'])
    values = section['x'][start:stop]
    probability = np.concatenate([section['historical'][start:stop], section['static'][start:stop], section['official'][start:stop], section['temporal'][start:stop]], axis=1)
    before = section['official'][max(0, start - 2):start]
    after = section['official'][stop:min(len(section['official']), stop + 2)]
    context = []
    for part in (before, after):
        context.extend((part.mean(axis=0) if len(part) else np.zeros(3, dtype=float)).tolist())
    line_curve = section['official'][start:stop, 0]
    return np.concatenate([np.asarray([stop - start, (stop - start) * 7.0], dtype=float), values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0), np.quantile(probability, 0.25, axis=0), np.quantile(probability, 0.5, axis=0), np.quantile(probability, 0.75, axis=0), np.asarray(context, dtype=float), np.asarray([float(np.mean(np.abs(np.diff(line_curve)))) if len(line_curve) > 1 else 0.0, float(np.max(line_curve)), float(np.min(line_curve)), float(np.mean(line_curve))])]).astype(np.float32)

def event_dataset(section: dict) -> dict:
    events = candidate_events(section)
    actual = truth_events(section['frame'], 'line')
    _, matched_predictions = match_indices(actual, events)
    return {'events': events, 'x': np.asarray([event_vector(section, event) for event in events], dtype=np.float32), 'y': np.asarray([index in matched_predictions for index in range(len(events))], dtype=np.int32), 'anchor': np.asarray([section['official'][event['first']:event['stop'], 0].max() for event in events], dtype=float)}

def event_estimators() -> dict[str, object]:
    result = {}
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        result[f'logreg_c{c:g}'] = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=5000, random_state=20260818))
    for leaf in (1, 2, 4, 8):
        result[f'extra_trees_leaf{leaf}'] = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=leaf, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=20260818)
    return result

def accepted_events(dataset: dict, scores: np.ndarray, policy: EventPolicy) -> list[dict]:
    keep = (scores >= policy.threshold) | (dataset['anchor'] >= policy.protected_anchor)
    return [{'start': event['start'], 'end': event['end']} for event, accepted in zip(dataset['events'], keep) if accepted]

def select_event_verifier(data: dict[str, dict], event_data: dict[str, dict], train_sections: list[str], definitions: dict[str, object]) -> tuple[str, EventPolicy, dict]:
    candidates = []
    for model_name, definition in definitions.items():
        oof_scores = {}
        for holdout in train_sections:
            fit_section = next((name for name in train_sections if name != holdout))
            fitted = clone(definition).fit(event_data[fit_section]['x'], event_data[fit_section]['y'])
            oof_scores[holdout] = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
        joined = np.concatenate([oof_scores[name] for name in train_sections])
        thresholds = np.unique(np.r_[np.arange(0.05, 0.951, 0.025), np.quantile(joined, np.linspace(0.05, 0.95, 37))])
        for threshold in thresholds:
            for anchor in (0.75, 0.85, 0.95, 1.01):
                policy = EventPolicy(float(threshold), float(anchor))
                by_section = {}
                for section in train_sections:
                    events = accepted_events(event_data[section], oof_scores[section], policy)
                    by_section[section] = compact_metric(event_metric(data[section]['frame'], events))
                min_recall = min((value['recall'] for value in by_section.values()))
                mean_fp = float(np.mean([value['false_positives_per_hour'] for value in by_section.values()]))
                mean_precision = float(np.mean([value['precision'] for value in by_section.values()]))
                candidates.append((model_name, policy, by_section, min_recall, mean_fp, mean_precision))
    feasible = [candidate for candidate in candidates if candidate[3] + 1e-12 >= 0.9]
    pool = feasible or candidates
    selected = max(pool, key=lambda value: (-value[4] if feasible else value[3], value[5], value[1].threshold))
    return (selected[0], selected[1], {'feasible': bool(feasible), 'by_section': selected[2], 'min_recall': selected[3], 'mean_false_positives_per_hour': selected[4]})

def run_outer(data: dict[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    definitions = event_estimators()
    event_data = {name: event_dataset(section) for name, section in data.items()}
    folds = []
    for holdout in SECTIONS:
        train = [name for name in SECTIONS if name != holdout]
        fusion_policy, fusion_inner = select_fusion(data, train)
        fusion_value = compact_metric(event_metric(data[holdout]['frame'], fusion_events(data[holdout], fusion_policy)))
        model_name, event_policy, event_inner = select_event_verifier(data, event_data, train, definitions)
        fitted = clone(definitions[model_name]).fit(np.concatenate([event_data[name]['x'] for name in train]), np.concatenate([event_data[name]['y'] for name in train]))
        scores = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
        verified = accepted_events(event_data[holdout], scores, event_policy)
        verifier_value = compact_metric(event_metric(data[holdout]['frame'], verified))
        base = baseline_metrics(data[holdout])
        folds.append({'holdout': holdout, 'baseline': {'recall': base['event_recall'], 'precision': base['event_precision'], 'false_positives_per_hour': base['event_false_positives_per_hour'], 'tp': base['event_tp'], 'fp': base['event_fp'], 'fn': base['event_fn']}, 'robust_fusion': fusion_value, 'fusion_policy': asdict(fusion_policy), 'fusion_inner': fusion_inner, 'event_verifier': verifier_value, 'event_verifier_model': model_name, 'event_verifier_policy': asdict(event_policy), 'event_verifier_inner': event_inner})
    return (folds, event_data)

def pooled_fusion(data: dict[str, dict]) -> tuple[FusionPolicy, dict]:
    return select_fusion(data, list(SECTIONS))

def pooled_event_verifier(data: dict[str, dict], event_data: dict[str, dict]) -> tuple[object, dict]:
    definitions = event_estimators()
    candidates = []
    for model_name, definition in definitions.items():
        scores_by_section = {}
        for holdout in SECTIONS:
            train = [name for name in SECTIONS if name != holdout]
            fitted = clone(definition).fit(np.concatenate([event_data[name]['x'] for name in train]), np.concatenate([event_data[name]['y'] for name in train]))
            scores_by_section[holdout] = fitted.predict_proba(event_data[holdout]['x'])[:, 1]
        joined = np.concatenate([scores_by_section[name] for name in SECTIONS])
        thresholds = np.unique(np.r_[np.arange(0.05, 0.951, 0.025), np.quantile(joined, np.linspace(0.05, 0.95, 37))])
        for threshold in thresholds:
            for anchor in (0.75, 0.85, 0.95, 1.01):
                policy = EventPolicy(float(threshold), float(anchor))
                by_section = {name: compact_metric(event_metric(data[name]['frame'], accepted_events(event_data[name], scores_by_section[name], policy))) for name in SECTIONS}
                min_recall = min((value['recall'] for value in by_section.values()))
                mean_fp = float(np.mean([value['false_positives_per_hour'] for value in by_section.values()]))
                mean_precision = float(np.mean([value['precision'] for value in by_section.values()]))
                candidates.append((model_name, policy, by_section, min_recall, mean_fp, mean_precision))
    feasible = [candidate for candidate in candidates if candidate[3] + 1e-12 >= 0.9]
    pool = feasible or candidates
    selected = max(pool, key=lambda value: (-value[4] if feasible else value[3], value[5]))
    estimator = clone(definitions[selected[0]]).fit(np.concatenate([event_data[name]['x'] for name in SECTIONS]), np.concatenate([event_data[name]['y'] for name in SECTIONS]))
    metadata = {'model': selected[0], 'policy': asdict(selected[1]), 'oof_by_section': selected[2], 'min_recall': selected[3], 'mean_false_positives_per_hour': selected[4], 'event_feature_count': int(next(iter(event_data.values()))['x'].shape[1]), 'status': 'pending_different_match_validation', 'must_not_deploy_yet': True}
    return (estimator, metadata)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_sections()
    folds, event_data = run_outer(data)
    fusion_policy, fusion_development = pooled_fusion(data)
    event_estimator, event_development = pooled_event_verifier(data, event_data)
    result = {'protocol': {'outer_validation': 'leave-one-section-out with selection only on the other two sections', 'sections_from_same_match': True, 'video_reprocessing': False, 'baseline_candidate_events': int(sum((len(value['events']) for value in event_data.values()))), 'positive_candidate_events': int(sum((value['y'].sum() for value in event_data.values())))}, 'folds': folds, 'aggregate': {'baseline': aggregate(folds, 'baseline'), 'robust_fusion': aggregate(folds, 'robust_fusion'), 'event_verifier': aggregate(folds, 'event_verifier')}, 'development_candidates_pending_external_match': {'fusion': {'policy': asdict(fusion_policy), **fusion_development}, 'event_verifier': event_development}}
    result['decision'] = {name: {'passes_90_percent_recall_every_fold': value['recall_min'] + 1e-12 >= 0.9, 'reduces_false_positives': value['false_positives_per_hour_mean'] < result['aggregate']['baseline']['false_positives_per_hour_mean']} for name, value in result['aggregate'].items() if name != 'baseline'}
    (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    joblib.dump(event_estimator, OUTPUT / 'event_verifier_pending_external_match.joblib')
    (OUTPUT / 'event_verifier_pending_external_match.json').write_text(json.dumps(event_development, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUTPUT / 'fusion_pending_external_match.json').write_text(json.dumps({'policy': asdict(fusion_policy), **fusion_development}, indent=2, ensure_ascii=False), encoding='utf-8')
    write_report(result)
    print(json.dumps({'aggregate': result['aggregate'], 'decision': result['decision']}, indent=2, ensure_ascii=False))

def write_report(result: dict) -> None:
    lines = ['# Pruebas adicionales para mejorar line', '', 'Se comparÃ³ una bÃºsqueda robusta de fusiÃ³n y un verificador de eventos completos. NingÃºn video fue reprocesado.', '', '| MÃ©todo | Recall medio | Recall mÃ­nimo | PrecisiÃ³n media | FP/h medio | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = {'baseline': 'Modelo 40/60', 'robust_fusion': 'FusiÃ³n robusta', 'event_verifier': 'Verificador por evento'}
    for key in ('baseline', 'robust_fusion', 'event_verifier'):
        value = result['aggregate'][key]
        lines.append(f"| {labels[key]} | {100 * value['recall_mean']:.1f}% | {100 * value['recall_min']:.1f}% | {100 * value['precision_mean']:.1f}% | {value['false_positives_per_hour_mean']:.1f} | {value['tp_total']} | {value['fp_total']} | {value['fn_total']} |")
    lines.extend(['', '## Resultados por tramo', ''])
    for fold in result['folds']:
        lines.append(f"- `{fold['holdout']}`: fusiÃ³n {100 * fold['robust_fusion']['recall']:.1f}% recall y {fold['robust_fusion']['false_positives_per_hour']:.1f} FP/h; verificador {100 * fold['event_verifier']['recall']:.1f}% y {fold['event_verifier']['false_positives_per_hour']:.1f} FP/h.")
    lines.extend(['', '## DecisiÃ³n', '', 'Solo debe promoverse una variante que mantenga al menos 90% de recall en cada tramo y reduzca los falsos positivos frente al modelo 40/60.', '', 'Los candidatos ajustados con los tres tramos quedan marcados como pendientes de un partido diferente.'])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
