"""Build a concise status/report without changing thesis documents."""
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
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import EXPERIMENT_DIR, MODEL_DIR, OUTPUT_DIR, load_candidates, write_json

def historical_duration(candidate: dict[str, object], grid: pd.DataFrame) -> float:
    mask = (grid.extractor == candidate['extractor']) & (grid.rnn_type == candidate['rnn_type']) & (pd.to_numeric(grid.p_rnn_units) == int(candidate['rnn_units'])) & (grid.p_bidirectional.astype(str).str.lower() == str(candidate['bidirectional']).lower()) & np.isclose(pd.to_numeric(grid.p_dropout), float(candidate['dropout'])) & (pd.to_numeric(grid.p_dense_units) == int(candidate['dense_units'])) & np.isclose(pd.to_numeric(grid.p_learning_rate), float(candidate['learning_rate'])) & (pd.to_numeric(grid.p_batch_size) == int(candidate['batch_size']))
    matches = grid[mask]
    if len(matches):
        return float(pd.to_numeric(matches.duration_s).median())
    family = grid[(grid.extractor == candidate['extractor']) & (grid.rnn_type == candidate['rnn_type'])]
    return float(pd.to_numeric(family.duration_s).median())

def main() -> None:
    audit = json.loads(_resolve_model_artifact(OUTPUT_DIR / 'audit_protocol.json').read_text(encoding='utf-8'))
    splits = json.loads(_resolve_model_artifact(OUTPUT_DIR / 'nested_split_summary.json').read_text(encoding='utf-8'))
    smoke = json.loads(_resolve_model_artifact(OUTPUT_DIR / 'integration_smoke' / 'summary.json').read_text(encoding='utf-8'))
    pilot_summary_path = OUTPUT_DIR / 'nested_evaluation_pilot_2fold_e12_manual' / 'summary.json'
    pilot_summary = json.loads(_resolve_model_artifact(pilot_summary_path).read_text(encoding='utf-8')) if _resolve_model_artifact(pilot_summary_path).exists() else None
    full_summary_path = OUTPUT_DIR / 'nested_evaluation' / 'summary.json'
    full_summary = json.loads(_resolve_model_artifact(full_summary_path).read_text(encoding='utf-8')) if _resolve_model_artifact(full_summary_path).exists() else None
    grid = pd.read_csv(_resolve_model_artifact(MODEL_DIR / 'results' / 'grid_search_results.csv'))
    candidates = load_candidates()
    mean_train = float(np.mean([fold['train']['rows'] for fold in splits]))
    historical_seconds = sum((historical_duration(candidate, grid) for candidate in candidates))
    full_estimate_seconds = historical_seconds * (mean_train / 1800.0) * len(splits)
    low_hours = full_estimate_seconds / 3600.0
    high_hours = low_hours * 1.8
    status = {'audit_complete': True, 'deep_overlap_complete': 'development_files_hashed' in audit['deep_overlap'], 'splits_complete': True, 'integration_smoke_complete': True, 'pilot_two_fold_complete': pilot_summary is not None, 'full_nested_training_complete': full_summary is not None, 'pending_executions': None if full_summary is not None else {'outer_folds': 5, 'candidate_trainings': 20, 'outer_evaluations': 5, 'command': '.\\experiments\\independent_evaluation\\run_experiment.ps1 -Mode full'}, 'estimated_cpu_hours': [round(low_hours, 1), round(high_hours, 1)], 'estimate_basis': 'Historical wall times for the four fixed candidate configurations, scaled by the mean new inner-training size; upper bound adds 80% for grouped-fold and I/O variation.'}
    write_json(OUTPUT_DIR / 'execution_status.json', status)
    full_row = {'evaluation': 'EvaluaciÃ³n anidada agrupada completa (5 pliegues)', 'status': 'completed' if full_summary is not None else 'pending_cpu_training', 'n': 4082 if full_summary is not None else None, 'accuracy': full_summary['metrics_mean']['accuracy'] if full_summary else None, 'f1_macro': full_summary['metrics_mean']['f1_macro'] if full_summary else None, 'f1_weighted': full_summary['metrics_mean']['f1_weighted'] if full_summary else None, 'recall_line': full_summary['metrics_mean']['recall_line'] if full_summary else None, 'recall_scrum': full_summary['metrics_mean']['recall_scrum'] if full_summary else None, 'set_used_during_model_selection': 'no', 'interpretation': 'corrected grouped nested estimate' if full_summary else 'required to quantify the corrected estimate'}
    comparison_rows = [{'evaluation': 'HistÃ³rica ResNet-50 + BiGRU', 'status': 'reported', 'n': 817, 'accuracy': 0.812, 'f1_macro': 0.787, 'f1_weighted': 0.817, 'recall_line': 0.853, 'recall_scrum': 0.844, 'set_used_during_model_selection': 'yes', 'interpretation': 'potentially optimistic'}, {'evaluation': 'IntegraciÃ³n anidada agrupada (1 Ã©poca, subconjunto)', 'status': 'integration_only', 'n': 24, 'accuracy': smoke['metrics_mean']['accuracy'], 'f1_macro': smoke['metrics_mean']['f1_macro'], 'f1_weighted': smoke['metrics_mean']['f1_weighted'], 'recall_line': smoke['metrics_mean']['recall_line'], 'recall_scrum': smoke['metrics_mean']['recall_scrum'], 'set_used_during_model_selection': 'no', 'interpretation': 'technical smoke test; not a performance estimate'}]
    if pilot_summary:
        comparison_rows.append({'evaluation': 'Piloto anidado agrupado (2 pliegues, 2 candidatos)', 'status': 'completed_pilot', 'n': pilot_summary['evaluated_unique_clips'], 'accuracy': pilot_summary['metrics_mean']['accuracy'], 'f1_macro': pilot_summary['metrics_mean']['f1_macro'], 'f1_weighted': pilot_summary['metrics_mean']['f1_weighted'], 'recall_line': pilot_summary['metrics_mean']['recall_line'], 'recall_scrum': pilot_summary['metrics_mean']['recall_scrum'], 'set_used_during_model_selection': 'no', 'interpretation': 'reduced grouped nested estimate; larger uncertainty than five folds'})
    comparison_rows.append(full_row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / 'comparison_table.csv', index=False)
    deep = audit['deep_overlap']
    conclusion = f"La estimaciÃ³n corregida completa obtuvo exactitud {full_summary['metrics_mean']['accuracy']:.3f} y F1 macro {full_summary['metrics_mean']['f1_macro']:.3f}." if full_summary else f"El piloto reducido obtuvo exactitud media {pilot_summary['metrics_mean']['accuracy']:.3f} y F1 macro medio {pilot_summary['metrics_mean']['f1_macro']:.3f}." if pilot_summary else 'TodavÃ­a no puede cuantificarse cuÃ¡nto cambia la estimaciÃ³n 0.812/0.787.'
    report = f"# Informe de evaluaciÃ³n independiente\n\n## AuditorÃ­a del protocolo anterior\n\nSe reconstruyeron {audit['source_manifest']['combined_unique_clips']} clips etiquetados:\n{audit['source_manifest']['development_unique_clips']} participaron en desarrollo y\n{audit['source_manifest']['test_unique_clips']} formaron la prueba histÃ³rica. La grilla\nregistrÃ³ mÃ©tricas de prueba para sus {audit['configuration_selection']['grid_rows']}\nconfiguraciones. MÃ¡s aÃºn, las columnas llamadas `val_f1_macro` y `val_f1_weighted` se\ncalcularon con `test_labels`; la primera fue la clave principal de ordenamiento. Programas\nposteriores tambiÃ©n usaron `test_accuracy` para ordenar o desempatar. En consecuencia, el\nconjunto de 817 ejemplos no fue una evaluaciÃ³n final independiente.\n\nLa tabla balanceada histÃ³rica contiene\n{audit['keras_validation_leakage']['balanced_rows']} filas pero solo\n{audit['keras_validation_leakage']['balanced_unique_clips']} clips Ãºnicos. Se reprodujo el\nsolapamiento de {audit['keras_validation_leakage']['overlap_unique_clips']} clips entre el\nentrenamiento y la validaciÃ³n internos, todos de scrum.\n\n## Material independiente\n\nLos 100 videos de las colecciones de Pablo Granitto y Juan Manuel Arriazu aparecen como\nmaterial interactivo o fuente de revisiÃ³n humana. Se localizaron tres videos con IDs de\nYouTube nuevos en `youtube_external_tests`. No tienen coincidencias SHA-256 ni coincidencias\nde huella visual muestreada con los 4082 clips, pero tampoco tienen etiquetas humanas por\nclip. Por eso son material candidato para una futura evaluaciÃ³n externa, no un conjunto de\nevaluaciÃ³n utilizable ahora. No se generaron etiquetas a partir del modelo.\n\n## Estrategia implementada\n\nSe prepararon cinco pliegues externos por grupo fuente. En cada uno existe un entrenamiento\ny una validaciÃ³n internos tambiÃ©n separados por grupo. Los cuatro candidatos quedaron\ndeclarados antes de cualquier evaluaciÃ³n externa y se seleccionan por F1 macro interno.\nNo se remuestrea ningÃºn subconjunto. Los pesos se calculan exclusivamente con el entrenamiento\ninterno. Early stopping y checkpoints observan solo la validaciÃ³n interna.\n\nTodas las aserciones dieron cero solapamientos por nombre, grupo fuente y prefijo temporal.\nEl anÃ¡lisis profundo procesÃ³ {deep.get('development_files_hashed', 0)} archivos, hallÃ³\n{len(deep.get('exact_external_matches', []))} coincidencias exactas externas y\n{len(deep.get('sampled_visual_matches_hamming_le_4', []))} coincidencias visuales muestreadas.\nEste Ãºltimo control es conservador pero no equivale a comparar exhaustivamente cada cuadro.\n\n## EjecuciÃ³n y limitaciones restantes\n\nLa prueba de integraciÃ³n terminÃ³ de punta a punta y produjo checkpoint, selecciÃ³n interna,\npredicciones, mÃ©tricas, matriz e intervalos bootstrap. Sus 24 ejemplos y una sola Ã©poca no\npermiten interpretar el resultado como rendimiento. El piloto reducido de dos pliegues\n{('terminÃ³ correctamente' if pilot_summary else 'estÃ¡ pendiente')}. La ejecuciÃ³n cientÃ­fica completa exige\n20 entrenamientos y 5 evaluaciones externas. El tiempo CPU estimado es de\n{status['estimated_cpu_hours'][0]} a {status['estimated_cpu_hours'][1]} horas.\n\n{conclusion} La metodologÃ­a ya corrige la fuga, pero afirmar una caÃ­da o una mejora con la prueba corta serÃ­a\nincorrecto. Cuando finalice el comando `-Mode full`, una caÃ­da serÃ¡ una estimaciÃ³n mÃ¡s realista\nde generalizaciÃ³n y no evidencia, por sÃ­ sola, de un error del programa.\n"
    (OUTPUT_DIR / 'EXPERIMENT_REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
