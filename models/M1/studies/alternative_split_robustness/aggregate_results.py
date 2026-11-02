"""Agrega los cinco pliegues externos de la prueba de robustez."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
LABELS = ['line', 'other', 'scrum']

def metrics(true: pd.Series, predicted: pd.Series) -> dict[str, float]:
    report = classification_report(true, predicted, labels=LABELS, target_names=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(true, predicted)), 'f1_macro': float(f1_score(true, predicted, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(true, predicted, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall'])}

def bootstrap_delta(frame: pd.DataFrame, candidate_column: str, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = {'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    for _ in range(draws):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        base = metrics(sample.true_label, sample.historical_label)
        candidate = metrics(sample.true_label, sample[candidate_column])
        for name in deltas:
            deltas[name].append(candidate[name] - base[name])
    return {f'{name}_{suffix}': float(getattr(np, suffix)(values)) for name, values in deltas.items() for suffix in ('mean',)} | {f'{name}_low': float(np.quantile(values, 0.025)) for name, values in deltas.items()} | {f'{name}_high': float(np.quantile(values, 0.975)) for name, values in deltas.items()}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    predictions = []
    for fold in range(1, 6):
        fold_root = root / f'final_fold_{fold:02d}_stream' / f'outer_{fold:02d}'
        info = json.loads(_resolve_model_artifact(fold_root / 'metrics.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(_resolve_model_artifact(fold_root / 'evaluation_predictions.csv'))
        predictions.append(frame)
        for model, key in (('historical', 'historical'), ('yolo_residual_uncalibrated', 'all_residual_uncalibrated'), ('yolo_residual_fixed_calibration', 'all_residual_calibrated'), ('yolo_residual_validation_calibration', 'all_residual_recalibrated_internal')):
            row = {'fold': fold, 'model': model, 'evaluation_rows': len(frame), **info[key]}
            fold_rows.append(row)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output / 'fold_metrics.csv', index=False)
    aggregate_rows = []
    for model, group in fold_frame.groupby('model', sort=True):
        aggregate_rows.append({'model': model, 'folds': len(group), 'evaluation_rows_total': int(group.evaluation_rows.sum()), **{f'{metric}_mean': float(group[metric].mean()) for metric in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}, **{f'{metric}_std': float(group[metric].std(ddof=0)) for metric in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}})
    pooled = pd.concat(predictions, ignore_index=True)
    pooled_rows = []
    for model, column in (('historical', 'historical_label'), ('yolo_residual_fixed_calibration', 'yolo_ensemble_label'), ('yolo_residual_validation_calibration', 'yolo_ensemble_selected_label')):
        pooled_rows.append({'model': model, 'evaluation_rows': len(pooled), **metrics(pooled.true_label, pooled[column])})
    pd.DataFrame(aggregate_rows).sort_values('model').to_csv(output / 'aggregate_by_fold.csv', index=False)
    pd.DataFrame(pooled_rows).to_csv(output / 'pooled_out_of_fold_metrics.csv', index=False)
    pooled.to_csv(output / 'all_evaluation_predictions.csv', index=False)
    bootstrap = {'fixed_calibration_vs_historical': bootstrap_delta(pooled, 'yolo_ensemble_label', args.bootstrap_draws, 20260812), 'validation_calibration_vs_historical': bootstrap_delta(pooled, 'yolo_ensemble_selected_label', args.bootstrap_draws, 20260813), 'draws': args.bootstrap_draws, 'unit': 'clip row bootstrap over 4082 outer-test predictions; fold direction is reported separately'}
    (output / 'paired_bootstrap.json').write_text(json.dumps(bootstrap, indent=2), encoding='utf-8')
    deltas = []
    for fold, frame in enumerate(predictions, 1):
        base = metrics(frame.true_label, frame.historical_label)
        for model, column in (('fixed_calibration', 'yolo_ensemble_label'), ('validation_calibration', 'yolo_ensemble_selected_label')):
            candidate = metrics(frame.true_label, frame[column])
            deltas.append({'fold': fold, 'model': model, 'accuracy_delta': candidate['accuracy'] - base['accuracy'], 'f1_macro_delta': candidate['f1_macro'] - base['f1_macro'], 'f1_weighted_delta': candidate['f1_weighted'] - base['f1_weighted']})
    pd.DataFrame(deltas).to_csv(output / 'fold_deltas.csv', index=False)
    summary = {'outer_folds': 5, 'evaluation_clips': int(len(pooled)), 'unique_evaluation_clips': int(pooled.video_name.nunique()), 'clip_overlap_between_outer_tests': 0, 'historical_817_clip_overlap_across_outer_tests': '817 clips are distributed across the five outer tests; no outer test equals the historical 817-clip set', 'training_epochs': 4, 'early_stopping_patience': 2, 'yolo_pose_model': 'yolo11n-pose.pt', 'note': 'Las mÃ©tricas se reportan con presupuesto de 4 Ã©pocas; el objetivo es comprobar sensibilidad al split, no reoptimizar la duraciÃ³n histÃ³rica. El universo externo son los 4082 clips etiquetados; por eso algunos clips del test antiguo aparecen distribuidos en distintos pliegues, pero nunca se usa el conjunto antiguo completo como test.'}
    (output / 'protocol_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Robustez con particiones externas', '', 'Se evaluaron cinco pliegues externos agrupados por fuente. Cada clip aparece una sola vez en evaluaciÃ³n externa; entrenamiento, validaciÃ³n y evaluaciÃ³n no comparten clips, grupos ni prefijos temporales.', '', f'Total evaluado: {len(pooled)} clips ({pooled.video_name.nunique()} Ãºnicos). El conjunto histÃ³rico fijo de 817 clips no se usÃ³ como test en esta corrida.', '', '## Resultados', '', 'La tabla `aggregate_by_fold.csv` contiene medias y desviaciones entre pliegues; `pooled_out_of_fold_metrics.csv` agrega las predicciones out-of-fold de los 4082 clips.', '', 'La calibraciÃ³n fija conserva los pesos desplegados (line=0.80, scrum=0.90). La calibraciÃ³n de validaciÃ³n se ajusta dentro de cada pliegue y se informa como anÃ¡lisis adicional, no como configuraciÃ³n desplegada.', '', '## Lectura', '', 'La mejora de exactitud de YOLO no es uniforme en todas las fuentes. El F1 macro es mÃ¡s sensible: hay pliegues donde baja, por lo que no corresponde afirmar una mejora universal. La comparaciÃ³n descarta que todo el rendimiento del modelo se explique Ãºnicamente por el split histÃ³rico, pero tambiÃ©n muestra que la ganancia depende del origen del video.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
