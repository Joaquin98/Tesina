"""CLI for preflight, preparation, execution and aggregation of the final matrix."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import OUTPUTS, assert_cache_ready, build_master_manifest, labels_for, load_master_manifest, prepare_final_record, preflight_report, split_for_run
from models.M3.event_verifier_training import apply_verifier, fit_verifier, load_verifier, select_verifier_threshold
from .metrics import apply_class_thresholds, block_metrics, temporal_metrics, threshold_search
from .protocol import CACHE_SIGNATURE, LABELS, MODELS, all_runs, panel_v_exclusion_reason, select_run
RUNS = OUTPUTS / 'runs'
REGISTRY = OUTPUTS / 'run_registry.csv'
RUN_MATRIX = OUTPUTS / 'run_matrix.csv'
PREFLIGHT = OUTPUTS / 'preflight.json'

@lru_cache(maxsize=None)
def _is_file(value: str) -> bool:
    return _resolve_model_artifact(Path(value)).is_file()

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def artifact_hash(paths: list[Path]) -> tuple[str, dict[str, str]]:
    individual = {}
    for path in paths:
        path = Path(path)
        if not _resolve_model_artifact(path).is_file():
            individual[str(path)] = 'missing'
            continue
        digest = hashlib.sha256()
        with _resolve_model_artifact(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        individual[str(path.resolve())] = digest.hexdigest()
    combined = hashlib.sha256(json.dumps(individual, sort_keys=True).encode('utf-8')).hexdigest()
    return (combined, individual)

def required_rows(master: pd.DataFrame, spec):
    train, development, test = split_for_run(master, spec)
    return (train, development, test, pd.concat([train, development, test], ignore_index=True))

def readiness(master: pd.DataFrame, spec) -> tuple[str, str, tuple[int, int, int]]:
    train, development, test, requested = required_rows(master, spec)
    status_ready = requested.resnet_status.astype(str).str.startswith('ready') & (requested.yolo_status.astype(str) == 'ready')
    paths_ready = requested.resnet_path.astype(str).map(_is_file) & requested.yolo_path.astype(str).map(_is_file)
    pending = int((~status_ready).sum())
    missing = int((~paths_ready).sum())
    if pending or missing:
        return ('esperando_cache', f'estados_pendientes={pending}; rutas_faltantes={missing}', (len(train), len(development), len(test)))
    if spec.model == 'M1' and spec.level in {'B', 'H0'}:
        m1_ready = requested.m1_pose32_status.astype(str).eq('ready')
        m1_paths = requested.m1_pose32_path.astype(str).map(_is_file)
        m1_pending = int((~m1_ready).sum())
        m1_missing = int((~m1_paths).sum())
        if m1_pending or m1_missing:
            return ('esperando_m1_pose32', f'estados_pendientes={m1_pending}; rutas_faltantes={m1_missing}', (len(train), len(development), len(test)))
    return ('lista', '', (len(train), len(development), len(test)))

def preflight(*, strict: bool=False) -> dict:
    master = build_master_manifest(write=True)
    base = preflight_report(master)
    rows = []
    for spec in all_runs():
        state, reason, sizes = readiness(master, spec)
        rows.append({**spec.as_dict(), 'state': state, 'reason': reason, 'n_train': sizes[0], 'n_dev': sizes[1], 'n_test': sizes[2]})
    matrix = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(RUN_MATRIX, index=False)
    result = {**base, 'runs_total': int(len(matrix)), 'runs_by_state': matrix.state.value_counts().to_dict(), 'run_matrix': str(RUN_MATRIX.resolve()), 'm1_pose32_pending_runs': int((matrix.state == 'esperando_m1_pose32').sum())}
    write_json(PREFLIGHT, result)
    if strict and (base['pending_rows'] or (matrix.state != 'lista').any()):
        raise SystemExit(2)
    return result

def prepare_final(*, overwrite: bool=False, limit: int | None=None) -> dict:
    master = build_master_manifest(write=True)
    final = master[master.domain == 'CV'].copy()
    ready = final.resnet_status.astype(str).str.startswith('ready') & (final.yolo_status == 'ready')
    final = final[ready]
    if limit is not None:
        final = final.iloc[:limit]
    completed, failed = (0, [])
    for row in final.itertuples(index=False):
        try:
            prepare_final_record(row, overwrite=overwrite)
            completed += 1
        except Exception as error:
            failed.append({'block_id': str(row.block_id), 'error': repr(error)})
    result = {'selected': int(len(final)), 'completed': completed, 'failed': failed}
    write_json(OUTPUTS / 'prepare_final_report.json', result)
    return result

def _prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray, run_id: str) -> pd.DataFrame:
    output = frame[['block_id', 'domain', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'label']].copy()
    output.insert(0, 'run_id', run_id)
    output = output.rename(columns={'group_id': 'test_group', 'label': 'y_true'})
    for index, label in enumerate(LABELS):
        output[f'p_{label}'] = probabilities[:, index]
    output['y_argmax'] = [LABELS[index] for index in probabilities.argmax(axis=1)]
    return output

def _augment_for_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result['y_true'] = result['label']
    for index, label in enumerate(LABELS):
        result[f'p_{label}'] = probabilities[:, index]
    return result

def _event_ids(frame: pd.DataFrame) -> np.ndarray:
    result = np.full(len(frame), '', dtype=object)
    counter = 0
    for (_group, _section), rows in frame.groupby(['test_group', 'section'], sort=False):
        ordered = rows.sort_values('block_index', kind='mergesort')
        previous_label, previous_block = (None, None)
        current_id = ''
        for index, row in ordered.iterrows():
            label, block = (str(row['y_final']), int(row['block_index']))
            if label not in {'line', 'scrum'}:
                previous_label, previous_block = (None, None)
                continue
            if label != previous_label or previous_block is None or block != previous_block + 1:
                counter += 1
                current_id = f'{label}_{counter:06d}'
            result[index] = current_id
            previous_label, previous_block = (label, block)
    return result

def _append_registry(row: dict) -> int:
    frame = pd.read_csv(_resolve_model_artifact(REGISTRY)) if _resolve_model_artifact(REGISTRY).is_file() else pd.DataFrame()
    index = len(frame)
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REGISTRY, index=False)
    return index

def _update_registry(index: int, values: dict) -> None:
    frame = pd.read_csv(_resolve_model_artifact(REGISTRY))
    for key, value in values.items():
        if key not in frame.columns:
            frame[key] = ''
        frame.loc[index, key] = value
    frame.to_csv(REGISTRY, index=False)

def _next_attempt(run_id: str) -> tuple[int, Path]:
    root = RUNS / run_id
    existing = sorted(_model_artifact_glob(root, 'attempt_*', recursive=False)) if _resolve_model_artifact(root).is_dir() else []
    attempt = len(existing) + 1
    return (attempt, root / f'attempt_{attempt:03d}')

def execute(run_id: str, *, epochs: int, patience: int, batch_size: int, smoke_limit: int | None=None) -> dict:
    spec = select_run(run_id)
    master = build_master_manifest(write=True)
    state, reason, sizes = readiness(master, spec)
    if state != 'lista':
        raise RuntimeError(f'{run_id}: {state}: {reason}')
    train, development, test, requested = required_rows(master, spec)
    if smoke_limit is not None:
        if len(train):
            train = train.groupby('label', group_keys=False).head(smoke_limit).reset_index(drop=True)
        development = development.groupby('label', group_keys=False).head(smoke_limit).reset_index(drop=True)
        test = test.groupby('label', group_keys=False).head(smoke_limit).reset_index(drop=True)
        run_id = f'{run_id}-SMOKE{smoke_limit}'
    assert_cache_ready(pd.concat([train, development, test], ignore_index=True))
    for row in pd.concat([train, development, test], ignore_index=True).query("domain == 'CV'").itertuples(index=False):
        prepare_final_record(row)
    attempt, run_dir = _next_attempt(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    registry_index = _append_registry({'run_id': run_id, 'attempt': attempt, 'date': datetime.now(timezone.utc).isoformat(), 'status': 'ejecutando', 'panel': spec.panel, 'model': spec.model, 'level': spec.level, 'data_regime': spec.data_regime, 'fold': spec.fold, 'seed': spec.seed, 'n_train': len(train), 'n_dev': len(development), 'n_test': len(test), 'run_dir': str(run_dir.resolve())})
    try:
        from .models import fit_predict
        bundle = fit_predict(spec.model, train, development, test, level=spec.level, seed=spec.seed, run_dir=run_dir, epochs=epochs, patience=patience, batch_size=batch_size)
        if bundle.development_probabilities is None:
            raise RuntimeError('Model adapter did not return development probabilities')
        dev_metric_frame = _augment_for_metrics(development, bundle.development_probabilities)
        if spec.level == 'B':
            thresholds = {'line': None, 'scrum': None, 'source': 'published_argmax'}
            dev_prediction = bundle.development_probabilities.argmax(axis=1)
            test_prediction = bundle.probabilities.argmax(axis=1)
        else:
            selected = threshold_search(dev_metric_frame, bundle.development_probabilities)
            thresholds = {'line': selected['line'], 'scrum': selected['scrum'], 'source': 'development'}
            dev_prediction = apply_class_thresholds(bundle.development_probabilities, line=selected['line'], scrum=selected['scrum'])
            test_prediction = apply_class_thresholds(bundle.probabilities, line=selected['line'], scrum=selected['scrum'])
        verifier_scores = np.full(len(test), np.nan)
        verifier_event_ids = np.full(len(test), '', dtype=object)
        if spec.model in {'M3', 'M4'}:
            weight = float(bundle.metadata['verifier_historical_weight'])
            if spec.level == 'B':
                verifier = load_verifier(Path(bundle.metadata['verifier_path']))
                verifier_threshold = float(bundle.metadata['verifier_threshold'])
            else:
                if not bundle.training_branches:
                    raise RuntimeError('Trainable verifier requires training branch probabilities')
                verifier_path = run_dir / 'line_verifier.joblib'
                verifier = fit_verifier(train, bundle.training_branches, historical_weight=weight, seed=spec.seed, output=verifier_path)
                bundle.artifacts.append(verifier_path)
                verifier_threshold = select_verifier_threshold(dev_metric_frame, bundle.development_branches, model=verifier, historical_weight=weight, base_prediction=dev_prediction)
            test_prediction, verifier_scores, verifier_event_ids = apply_verifier(test, bundle.branches, model=verifier, threshold=verifier_threshold, historical_weight=weight, base_prediction=test_prediction)
            thresholds['line_verifier'] = verifier_threshold
        predictions = _prediction_frame(test, bundle.probabilities, run_id)
        predictions['y_final'] = [LABELS[index] for index in test_prediction]
        predictions['line_verifier_score'] = verifier_scores
        predictions['predicted_event_id'] = _event_ids(predictions)
        accepted = verifier_event_ids != ''
        predictions.loc[accepted, 'predicted_event_id'] = verifier_event_ids[accepted]
        for branch_name, branch in bundle.branches.items():
            for index, label in enumerate(LABELS):
                predictions[f'p_{branch_name}_{label}'] = branch[:, index]
        combined_hash, hashes = artifact_hash(bundle.artifacts)
        predictions['checkpoint_hash'] = combined_hash
        predictions['cache_signature'] = CACHE_SIGNATURE
        predictions.to_csv(run_dir / 'predictions.csv', index=False)
        metric_frame = test.reset_index(drop=True).copy()
        metric_frame['y_true'] = metric_frame['label']
        metric_frame['y_final'] = predictions['y_final'].to_numpy()
        for label in LABELS:
            metric_frame[f'p_{label}'] = predictions[f'p_{label}'].to_numpy()
        metrics = {'block': block_metrics(labels_for(test), bundle.probabilities), 'temporal': temporal_metrics(metric_frame)}
        elapsed = time.perf_counter() - started
        metadata = {**spec.as_dict(), 'effective_run_id': run_id, 'attempt': attempt, 'smoke_limit': smoke_limit, 'n_train': len(train), 'n_dev': len(development), 'n_test': len(test), 'epochs_requested': epochs, 'patience': patience, 'batch_size': batch_size, 'thresholds': thresholds, 'model_metadata': bundle.metadata, 'artifacts': hashes, 'checkpoint_hash': combined_hash, 'cache_signature': CACHE_SIGNATURE, 'elapsed_seconds': elapsed, 'python': sys.version, 'platform': platform.platform()}
        write_json(run_dir / 'metrics.json', metrics)
        write_json(run_dir / 'run.json', metadata)
        _update_registry(registry_index, {'status': 'completa', 'elapsed_seconds': elapsed, 'checkpoint_hash': combined_hash})
        return {'run_dir': str(run_dir.resolve()), 'metrics': metrics, 'metadata': metadata}
    except Exception as error:
        (run_dir / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        _update_registry(registry_index, {'status': 'fallida', 'elapsed_seconds': time.perf_counter() - started, 'error': repr(error)})
        raise

def selected_specs(args) -> list:
    result = []
    for spec in all_runs():
        if args.panel and spec.panel != args.panel:
            continue
        if args.model and spec.model != args.model:
            continue
        if args.level and spec.level != args.level:
            continue
        if args.regime and spec.data_regime != args.regime:
            continue
        if args.fold and spec.fold != args.fold:
            continue
        result.append(spec)
    return result

def run_matrix(args) -> dict:
    master = build_master_manifest(write=True)
    selected = selected_specs(args)
    if args.max_runs is not None:
        selected = selected[:args.max_runs]
    rows = []
    completed_ids = set()
    if args.resume and _resolve_model_artifact(REGISTRY).is_file():
        registry = pd.read_csv(_resolve_model_artifact(REGISTRY))
        completed_ids = set(registry.loc[registry.status == 'completa', 'run_id'].astype(str))
    for spec in selected:
        state, reason, sizes = readiness(master, spec)
        exclusion_reason = panel_v_exclusion_reason(spec)
        action = 'run'
        if spec.run_id in completed_ids:
            action = 'skip_complete'
        elif exclusion_reason:
            action = 'skip_excluded'
            reason = exclusion_reason
        elif state != 'lista':
            action = f'skip_{state}'
        rows.append({'run_id': spec.run_id, 'action': action, 'reason': reason, 'sizes': sizes})
        if not args.dry_run and action == 'run':
            execute(spec.run_id, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, smoke_limit=args.smoke_limit)
    return {'selected': len(selected), 'actions': rows}

def summarize() -> dict:
    completed_attempts = None
    if _resolve_model_artifact(REGISTRY).is_file():
        registry = pd.read_csv(_resolve_model_artifact(REGISTRY))
        completed = registry.loc[registry.status == 'completa', ['run_id', 'attempt']]
        completed_attempts = {(str(row.run_id), int(row.attempt)) for row in completed.itertuples(index=False)}
    rows = []
    for path in _model_artifact_glob(RUNS, '*/attempt_*/metrics.json', recursive=False):
        run_path = path.with_name('run.json')
        if not _resolve_model_artifact(run_path).is_file():
            continue
        metrics = json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8'))
        run = json.loads(_resolve_model_artifact(run_path).read_text(encoding='utf-8'))
        if run.get('smoke_limit') is not None:
            continue
        attempt_key = (str(run['run_id']), int(run['attempt']))
        if completed_attempts is not None and attempt_key not in completed_attempts:
            continue
        rows.append({'run_id': run['run_id'], 'panel': run['panel'], 'model': run['model'], 'level': run['level'], 'data_regime': run['data_regime'], 'fold': run['fold'], 'f1_macro': metrics['block']['f1_macro'], 'balanced_accuracy': metrics['block']['balanced_accuracy'], 'f1w_macro': metrics['temporal']['weighted_f1_macro'], 'f1w_line': metrics['temporal']['line']['weighted_f1'], 'f1w_scrum': metrics['temporal']['scrum']['weighted_f1'], 'fp_far_hour': metrics['temporal']['line']['far_false_positives_per_hour'] + metrics['temporal']['scrum']['far_false_positives_per_hour']})
    detail = pd.DataFrame(rows)
    detail_path = OUTPUTS / 'completed_fold_results.csv'
    detail.to_csv(detail_path, index=False)
    if detail.empty:
        summary = pd.DataFrame(columns=['panel', 'model', 'level', 'data_regime', 'folds', 'f1w_macro', 'f1_macro'])
    else:
        summary = detail.groupby(['panel', 'model', 'level', 'data_regime'], dropna=False).agg(folds=('fold', 'nunique'), f1w_macro=('f1w_macro', 'mean'), f1w_line=('f1w_line', 'mean'), f1w_scrum=('f1w_scrum', 'mean'), f1_macro=('f1_macro', 'mean'), balanced_accuracy=('balanced_accuracy', 'mean'), fp_far_hour=('fp_far_hour', 'mean')).reset_index()
    summary_path = OUTPUTS / 'comparison_summary.csv'
    summary.to_csv(summary_path, index=False)
    return {'completed_folds': len(detail), 'detail': str(detail_path.resolve()), 'summary': str(summary_path.resolve())}

def audit_panel_v() -> dict:
    """Verify persisted Panel-V attempts against their frozen run specifications."""
    master = build_master_manifest(write=False)
    registry = pd.read_csv(_resolve_model_artifact(REGISTRY)) if _resolve_model_artifact(REGISTRY).is_file() else pd.DataFrame()
    rows = []
    for spec in (value for value in all_runs() if value.panel == 'PV'):
        issues = []
        registered = registry.loc[registry.run_id.astype(str) == spec.run_id].copy() if len(registry) and 'run_id' in registry else pd.DataFrame()
        completed = registered.loc[registered.status == 'completa'].sort_values('attempt') if len(registered) and 'status' in registered else pd.DataFrame()
        state, reason, sizes = readiness(master, spec)
        exclusion_reason = panel_v_exclusion_reason(spec)
        audit_state = state if state != 'lista' else 'pendiente'
        attempt = None
        run_dir = None
        if len(completed):
            selected = completed.iloc[-1]
            attempt = int(selected.attempt)
            run_dir = Path(str(selected.run_dir))
            required = {'run': run_dir / 'run.json', 'metrics': run_dir / 'metrics.json', 'predictions': run_dir / 'predictions.csv'}
            missing = [name for name, path in required.items() if not _resolve_model_artifact(path).is_file()]
            if missing:
                issues.append(f"artefactos_faltantes={','.join(missing)}")
            else:
                try:
                    run = json.loads(_resolve_model_artifact(required['run']).read_text(encoding='utf-8'))
                    metrics = json.loads(_resolve_model_artifact(required['metrics']).read_text(encoding='utf-8'))
                    predictions = pd.read_csv(_resolve_model_artifact(required['predictions']))
                    probability_columns = [f'p_{label}' for label in LABELS]
                    if run.get('run_id') != spec.run_id:
                        issues.append('run_id_inconsistente')
                    if run.get('test_group') != spec.test_group:
                        issues.append('test_group_metadata_inconsistente')
                    if run.get('cache_signature') != CACHE_SIGNATURE:
                        issues.append('firma_cache_inconsistente')
                    if len(predictions) != sizes[2] or int(run.get('n_test', -1)) != sizes[2]:
                        issues.append(f'filas_test={len(predictions)}_esperadas={sizes[2]}')
                    observed_groups = set(predictions.get('test_group', pd.Series(dtype=str)).astype(str))
                    if observed_groups != {spec.test_group}:
                        issues.append(f'grupos_test={sorted(observed_groups)}')
                    if not set(probability_columns).issubset(predictions.columns):
                        issues.append('columnas_probabilidad_faltantes')
                    else:
                        probabilities = predictions[probability_columns].to_numpy(np.float64)
                        if not np.isfinite(probabilities).all():
                            issues.append('probabilidades_no_finitas')
                        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-05):
                            issues.append('probabilidades_no_normalizadas')
                        if ((probabilities < -1e-07) | (probabilities > 1.0 + 1e-07)).any():
                            issues.append('probabilidades_fuera_de_rango')
                    if 'block' not in metrics or 'temporal' not in metrics:
                        issues.append('metricas_incompletas')
                    train, development, test = split_for_run(master, spec)
                    sets = [set(value.block_id.astype(str)) for value in (train, development, test)]
                    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                        issues.append('solapamiento_conjuntos')
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    issues.append(f'error_lectura={error!r}')
            audit_state = 'verificada' if not issues else 'invalida'
        elif exclusion_reason:
            audit_state = 'excluida'
            reason = exclusion_reason
        elif len(registered) and (registered.status == 'ejecutando').any():
            audit_state = 'ejecutando'
        elif len(registered) and (registered.status == 'fallida').any():
            audit_state = 'fallida'
        rows.append({'run_id': spec.run_id, 'model': spec.model, 'level': spec.level, 'data_regime': spec.data_regime, 'fold': spec.fold, 'test_group': spec.test_group, 'state': audit_state, 'attempt': attempt, 'n_train': sizes[0], 'n_dev': sizes[1], 'n_test': sizes[2], 'reason': '; '.join(issues) if issues else reason, 'run_dir': '' if run_dir is None else str(run_dir.resolve())})
    detail = pd.DataFrame(rows)
    detail_path = OUTPUTS / 'panel_v_completion_audit.csv'
    report_path = OUTPUTS / 'panel_v_completion_audit.json'
    detail.to_csv(detail_path, index=False)
    required = detail.state != 'excluida'
    all_required_verified = bool((detail.loc[required, 'state'] == 'verificada').all())
    result = {'runs_total': int(len(detail)), 'runs_required': int(required.sum()), 'runs_excluded': int((~required).sum()), 'states': {str(key): int(value) for key, value in detail.state.value_counts().items()}, 'all_required_verified': all_required_verified, 'all_resolved': bool(detail.state.isin({'verificada', 'excluida'}).all()), 'all_verified': all_required_verified, 'detail': str(detail_path.resolve())}
    write_json(report_path, result)
    result['report'] = str(report_path.resolve())
    return result

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    command = commands.add_parser('preflight')
    command.add_argument('--strict', action='store_true')
    command = commands.add_parser('prepare-final')
    command.add_argument('--overwrite', action='store_true')
    command.add_argument('--limit', type=int)
    command = commands.add_parser('run')
    command.add_argument('run_id')
    command.add_argument('--epochs', type=int, default=40)
    command.add_argument('--patience', type=int, default=7)
    command.add_argument('--batch-size', type=int, default=16)
    command.add_argument('--smoke-limit', type=int)
    command = commands.add_parser('run-matrix')
    command.add_argument('--panel', choices=('PV', 'PH'))
    command.add_argument('--model', choices=MODELS)
    command.add_argument('--level', choices=('B', 'H0', 'H1', 'H2'))
    command.add_argument('--regime', choices=('D0', 'D1', 'D2'))
    command.add_argument('--fold')
    command.add_argument('--max-runs', type=int)
    command.add_argument('--dry-run', action='store_true')
    command.add_argument('--resume', action='store_true')
    command.add_argument('--epochs', type=int, default=40)
    command.add_argument('--patience', type=int, default=7)
    command.add_argument('--batch-size', type=int, default=16)
    command.add_argument('--smoke-limit', type=int)
    commands.add_parser('summarize')
    commands.add_parser('audit-pv')
    return result

def main(argv: list[str] | None=None) -> int:
    args = parser().parse_args(argv)
    if args.command == 'preflight':
        value = preflight(strict=args.strict)
    elif args.command == 'prepare-final':
        value = prepare_final(overwrite=args.overwrite, limit=args.limit)
    elif args.command == 'run':
        value = execute(args.run_id, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, smoke_limit=args.smoke_limit)
    elif args.command == 'run-matrix':
        value = run_matrix(args)
    elif args.command == 'audit-pv':
        value = audit_panel_v()
    else:
        value = summarize()
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
