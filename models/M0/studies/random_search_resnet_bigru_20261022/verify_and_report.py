"""Recompute saved metrics and verify immutable inputs after the run completes."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
LABELS = ['line', 'other', 'scrum']

def read(name):
    return json.loads(_resolve_model_artifact(HERE / name).read_text(encoding='utf-8'))

def main():
    completion = read('completion.json')
    assert completion['status'] == 'complete'
    audit = read('split_audit.json')
    for name, expected in audit['source_hashes'].items():
        assert hashlib.sha256(_resolve_model_artifact(ROOT / name).read_bytes()).hexdigest() == expected, name
    parts = {name: pd.read_csv(_resolve_model_artifact(HERE / (name + '.csv'))) for name in ['train', 'validation', 'test']}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(parts[a].video_name) & set(parts[b].video_name)
    results = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
    selected = read('selected_configuration.json')
    assert int(results.loc[results.val_f1_macro.idxmax(), 'trial']) == selected['trial']
    assert len(results) <= 12
    assert results.epochs.max() <= read('budget_plan.json')['epochs_cap'] <= 10
    assert not results[['units', 'dropout', 'dense', 'learning_rate']].duplicated().any()
    val_y = np.array([LABELS.index(x) for x in parts['validation'].tag])
    for row in results.to_dict('records'):
        d = HERE / f"trial_{int(row['trial']):02d}"
        assert _resolve_model_artifact(d / 'best.keras').exists() and _resolve_model_artifact(d / 'best.weights.h5').exists()
        pred = _load_model_result_array(d / 'validation_probabilities.npy').argmax(axis=1)
        score = f1_score(val_y, pred, average='macro', labels=[0, 1, 2])
        assert abs(score - row['val_f1_macro']) < 1e-12
        history = pd.read_csv(_resolve_model_artifact(d / 'history.csv'))
        assert abs(history.val_f1_macro.max() - score) < 1e-12
        assert int(history.loc[history.val_f1_macro.idxmax(), 'epoch']) == row['best_epoch']
    test_pred = pd.read_csv(_resolve_model_artifact(HERE / 'historical_test_predictions.csv'))
    assert list(test_pred.video_name) == list(parts['test'].video_name)
    test_y = [LABELS.index(x) for x in test_pred.tag]
    pred = [LABELS.index(x) for x in test_pred.prediction]
    test = read('historical_test_metrics.json')
    assert abs(f1_score(test_y, pred, average='macro', labels=[0, 1, 2]) - test['f1_macro']) < 1e-12
    recalls = recall_score(test_y, pred, labels=[0, 1, 2], average=None)
    assert abs(recalls[0] - test['line_recall']) < 1e-12
    assert abs(recalls[2] - test['scrum_recall']) < 1e-12
    summary = results[['trial', 'units', 'dropout', 'dense', 'learning_rate', 'val_f1_macro', 'val_line_f1', 'val_line_recall', 'val_scrum_f1', 'val_scrum_recall', 'epochs', 'best_epoch', 'duration_s']].copy()
    summary.to_csv(HERE / 'summary.csv', index=False)
    repetition = None
    if _resolve_model_artifact(HERE / 'second_seed_status.json').exists():
        repetition = read('second_seed_status.json')
        if repetition['status'] == 'complete':
            d = HERE / f"repetition_seed_{repetition['seed']}"
            pred = _load_model_result_array(d / 'validation_probabilities.npy').argmax(axis=1)
            score = f1_score(val_y, pred, average='macro', labels=[0, 1, 2])
            assert abs(score - repetition['val_f1_macro']) < 1e-12
            assert repetition['no_reselection'] and (not repetition['test_evaluated'])
    repeat_text = 'No se ejecuto repeticion con segunda semilla.' if not repetition or repetition['status'] != 'complete' else f"Repeticion separada con semilla {repetition['seed']}: F1 macro de validacion {repetition['val_f1_macro']:.4f}, diferencia {repetition['delta_val_f1_macro']:+.4f}. Sin reseleccion ni nueva evaluacion de prueba."
    text = ['BUSQUEDA ALEATORIA REDUCIDA: RESNET-50 + BiGRU', '', summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'), '', f"Configuracion seleccionada exclusivamente por F1 macro de validacion: ensayo {selected['trial']}.", f"Prueba historica: F1 macro {test['f1_macro']:.4f}, exactitud {test['accuracy']:.4f}.", f"Lineout: F1 {test['line_f1']:.4f}, sensibilidad {test['line_recall']:.4f}.", f"Scrum: F1 {test['scrum_f1']:.4f}, sensibilidad {test['scrum_recall']:.4f}.", f"Tiempo del piloto, busqueda y evaluacion: {completion['search_and_evaluation_elapsed_s'] / 60:.2f} min.", repeat_text, f"Tiempo total incluyendo repeticion si corresponde: {completion.get('overall_elapsed_s', completion['search_and_evaluation_elapsed_s']) / 60:.2f} min.", '', 'Protocolo: secuencias originales de 200 x 2048, lote 32, Adam, dos capas BiGRU', 'con u y u/2 unidades, dropout y densa opcional. Semilla de muestreo e inicio', '20261022. Division estratificada unica 70/30 con semilla 42 antes del', 'remuestreo. Entrenamiento mixto a 420 filas/clase (1260 filas, 1093 clips', 'unicos); validacion natural de 980 clips. Pesos de clase omitidos porque', 'el entrenamiento esta balanceado. Cero intersecciones de identificadores', 'de clip entre entrenamiento, validacion y prueba. No se verifico independencia', 'por partido: no se encontro una correspondencia completa fiable. Posible', 'dependencia entre clips del mismo partido y clips similares.', '', 'Se generaron hasta 12 candidatos unicos con unidades 32/64/128, dropout', 'uniforme [0.2,0.5], densa 0/64 y tasa log-uniforme [1e-4,1e-3].', 'El piloto de 96 ejemplos de entrenamiento y 96 de validacion con 128', 'unidades estimo el costo por epoca. El plan presupuestado redujo candidatos', 'y epocas ANTES de observar sus metricas. El piloto no compite en la seleccion.', 'Seleccion de checkpoint y configuracion por F1 macro de validacion con', 'paciencia 2. Se restaura el mejor checkpoint de cada ensayo.', '', 'LIMITACIONES: presupuesto pequeno, pocas epocas y una sola particion de desarrollo.', 'No demuestra convergencia, optimalidad ni estabilidad entre semillas.', 'Los 817 clips de prueba ya fueron consultados en experimentos anteriores:', 'esta es una evaluacion historica, no una prueba externa nueva. Solo se', 'evaluo en prueba la configuracion seleccionada despues de congelar la seleccion.', 'La division corregida y el presupuesto reducido impiden comparar este', 'resultado con la grilla historica como comparacion controlada de estrategias.', 'Ademas, el programa de grilla historico asigna a val_f1_macro un F1 calculado', 'sobre prueba. La nueva seleccion usa exclusivamente validacion real.', 'Ver historical_code_audit.json para el hallazgo y el protocolo aplicado.', 'Demuestra que la busqueda aleatoria acotada puede ejecutarse y dejar', 'resultados reproducibles sin repetir los 576 ensayos ni extraer caracteristicas.', '', 'Verificacion: metricas recalculadas a partir de predicciones guardadas,', 'seleccion y epoca optima contrastadas con historias, ausencia de intersecciones', 'de clips y hashes intactos de los CSV y programas historicos leidos.', 'Artefactos: configuration.json, environment.json, split_audit.json, particiones', 'CSV, pilot.json, budget_plan.json, run.log, results.csv, summary.csv, trial_*/', '(parametros, historia, probabilidades y metricas de validacion, best.keras y', 'best.weights.h5), selected_configuration.json, historical_test_metrics.json,', 'historical_test_predictions.csv, completion.json y verification.json.']
    (HERE / 'RESULTS.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    (HERE / 'verification.json').write_text(json.dumps({'status': 'passed', 'trials_verified': len(results), 'historical_inputs_unchanged': True, 'metrics_recomputed': True, 'clip_intersections': 0}, indent=2), encoding='utf-8')
    print('\n'.join(text[:12]))
if __name__ == '__main__':
    main()
