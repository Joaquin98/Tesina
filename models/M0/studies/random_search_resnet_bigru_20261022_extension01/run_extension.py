"""Complete remaining frozen random candidates using validation only.

Preserves the original run. Limits combined active compute to the original hour.
Uses a compiled inference graph, checking parity against eager inference first.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
HERE = experiment_directory(__file__)
BASE = HERE.parent / 'random_search_resnet_bigru_20261022'
sys.path.insert(0, str(source_directory(BASE)))
import run_search as rs
import numpy as np
import pandas as pd
import tensorflow as tf

def read(path):
    return json.loads(_resolve_model_artifact(study_csv_path(path)).read_text(encoding='utf-8'))

def fast_predict(model, data):
    """One graph per model, fixed input signature including variable last batch."""
    if not hasattr(model, '_extension_infer'):
        model._extension_infer = tf.function(lambda features, masks: model([features, masks], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)])
    return np.concatenate([model._extension_infer(*data[i][0]).numpy() for i in range(len(data))])

def main():
    if _resolve_model_artifact(study_csv_path(HERE / 'configuration.json')).exists():
        raise RuntimeError('Preserve this extension; use a new directory for another run.')
    original = read(BASE / 'configuration.json')
    previous = pd.read_csv(_resolve_model_artifact(study_csv_path(BASE / 'results.csv')))
    completion = read(BASE / 'completion.json')
    budget = original['budget_s'] - completion['overall_elapsed_s']
    candidates = original['candidates_in_order'][len(previous):]
    assert len(candidates) + len(previous) <= 12
    for p in candidates:
        assert not any((all((row[k] == p[k] for k in p)) for row in previous.to_dict('records')))
    source_hashes = read(BASE / 'split_audit.json')['source_hashes']
    for relative, expected in source_hashes.items():
        assert rs.digest(rs.ROOT / relative) == expected
    train = pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'train.csv')))
    val = pd.read_csv(_resolve_model_artifact(study_csv_path(BASE / 'validation.csv')))
    assert not set(train.video_name) & set(val.video_name)
    assert not val.video_name.duplicated().any()
    store = rs.Store()
    for name in pd.concat([train, val]).video_name:
        assert name in store.index
    rs.save(HERE / 'configuration.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Validation-only extension over the remaining frozen candidates', 'original_directory': str(BASE), 'original_active_compute_s': completion['overall_elapsed_s'], 'extension_budget_s': budget, 'combined_compute_cap_s': original['budget_s'], 'epochs_cap': 3, 'patience': 2, 'batch_size': 32, 'sequence_length': 200, 'sampling_seed': original['sampling_seed'], 'initialization_seed': rs.SEED, 'candidates': candidates, 'split_hashes': {name: rs.digest(BASE / name) for name in ['train.csv', 'validation.csv']}, 'test_policy': 'No test manifest, features, predictions or scores read by extension. No test evaluation.', 'script_sha256': rs.digest(__file__), 'reused_module_sha256': rs.digest(_resolve_model_artifact(study_csv_path(BASE / 'run_search.py'))), 'python': sys.executable, 'tensorflow': tf.__version__, 'keras': rs.keras.__version__})
    start = time.monotonic()
    deadline = start + budget
    winner = read(BASE / 'selected_configuration.json')
    model = rs.keras.models.load_model(_resolve_model_artifact(study_csv_path(BASE / f"trial_{winner['trial']:02d}" / 'best.keras')))
    probe = rs.Data(val.iloc[:64], store)
    t = time.monotonic()
    eager = rs.predict(model, probe)
    eager_s = time.monotonic() - t
    t = time.monotonic()
    graph = fast_predict(model, probe)
    first_graph_s = time.monotonic() - t
    t = time.monotonic()
    graph_warm = fast_predict(model, probe)
    warm_graph_s = time.monotonic() - t
    np.testing.assert_allclose(graph, eager, rtol=1e-05, atol=1e-06)
    np.testing.assert_allclose(graph_warm, eager, rtol=1e-05, atol=1e-06)
    assert np.array_equal(eager.argmax(1), graph.argmax(1))
    rs.save(HERE / 'inference_parity.json', {'validation_probe_rows': 64, 'eager_s': eager_s, 'first_graph_s': first_graph_s, 'warm_graph_s': warm_graph_s, 'max_probability_difference': float(np.max(np.abs(eager - graph))), 'argmax_equal': True, 'rtol': 1e-05, 'atol': 1e-06})
    print('INFERENCE PARITY ' + json.dumps(read(HERE / 'inference_parity.json')), flush=True)
    del model, probe
    rs.keras.backend.clear_session()
    gc.collect()
    rs.predict = fast_predict
    results = []
    stop_reason = 'all_candidates_completed'
    for i, p in enumerate(candidates, len(previous) + 1):
        if results:
            estimate = max((r['duration_s'] / (1 + r['units'] / 128) for r in results)) * (1 + p['units'] / 128) * 1.5
            estimate = max(90.0, estimate)
        else:
            estimate = previous.duration_s.max() * 1.2
        remaining = deadline - time.monotonic()
        if remaining < estimate + 30:
            stop_reason = 'budget_admission'
            print(f'STOP: remaining={remaining:.1f}s, estimated_trial={estimate:.1f}s', flush=True)
            break
        directory = HERE / f'trial_{i:02d}'
        directory.mkdir()
        rs.save(directory / 'parameters.json', p)
        rs.keras.backend.clear_session()
        rs.keras.utils.set_random_seed(rs.SEED)
        model = rs.build(p)
        train_data = rs.Data(train, store, True)
        val_data = rs.Data(val, store)
        x, _ = val_data[0]
        eager = model(x, training=False).numpy()
        probe = rs.Data(val.iloc[:32], store)
        graph = fast_predict(model, probe)
        np.testing.assert_allclose(eager, graph, rtol=1e-05, atol=1e-06)
        assert np.array_equal(eager.argmax(1), graph.argmax(1))
        monitor = rs.Monitor(val_data, directory, deadline - 30)
        t = time.monotonic()
        print(f'TRIAL {i}/12 {p}', flush=True)
        model.fit(train_data, epochs=3, callbacks=[monitor], verbose=0, shuffle=False)
        model.load_weights(_resolve_model_artifact(study_csv_path(directory / 'best.weights.h5')))
        if not results:
            slow = np.concatenate([model(val_data[j][0], training=False).numpy() for j in range(len(val_data))])
            quick = fast_predict(model, val_data)
            np.testing.assert_allclose(slow, quick, rtol=1e-05, atol=1e-06)
            assert np.array_equal(slow.argmax(1), quick.argmax(1))
            rs.save(HERE / 'trained_full_validation_parity.json', {'rows': len(val), 'trial': i, 'max_probability_difference': float(np.max(np.abs(slow - quick))), 'argmax_equal': True})
        model.save(directory / 'best.keras')
        scores = read(directory / 'validation_metrics.json')
        row = {'trial': i, **p, 'seed': rs.SEED, 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'duration_s': time.monotonic() - t, **{f'val_{k}': v for k, v in scores.items() if isinstance(v, float)}}
        results.append(row)
        pd.DataFrame(results).to_csv(HERE / 'results.csv', index=False)
        combined = pd.concat([previous, pd.DataFrame(results)], ignore_index=True)
        combined.to_csv(HERE / 'combined_results.csv', index=False)
        print('TRIAL RESULT ' + json.dumps(row), flush=True)
        del model, monitor, train_data, val_data, probe
        rs.keras.backend.clear_session()
        gc.collect()
    elapsed = time.monotonic() - start
    combined = pd.concat([previous, pd.DataFrame(results)], ignore_index=True)
    selected = combined.loc[combined.val_f1_macro.idxmax()].to_dict()
    selected['trial'] = int(selected['trial'])
    selected['checkpoint'] = str((BASE if selected['trial'] <= len(previous) else HERE) / f"trial_{selected['trial']:02d}" / 'best.keras')
    selected['criterion'] = 'Validation macro F1 only, earliest trial breaks ties. No new test evaluation.'
    rs.save(HERE / 'selected_configuration.json', selected)
    y = np.array([rs.LABELS.index(tag) for tag in val.tag])
    for row in combined.to_dict('records'):
        directory = (BASE if row['trial'] <= len(previous) else HERE) / f"trial_{int(row['trial']):02d}"
        prob = _load_model_result_array(directory / 'validation_probabilities.npy')
        score = rs.metrics(y, prob)['f1_macro']
        assert abs(score - row['val_f1_macro']) < 1e-12
        history = pd.read_csv(_resolve_model_artifact(study_csv_path(directory / 'history.csv')))
        assert abs(history.val_f1_macro.max() - score) < 1e-12
        assert _resolve_model_artifact(study_csv_path(directory / 'best.keras')).exists()
    for relative, expected in source_hashes.items():
        assert rs.digest(rs.ROOT / relative) == expected
    rs.save(HERE / 'completion.json', {'status': 'complete', 'stop_reason': stop_reason, 'new_trials_completed': len(results), 'total_configurations': len(combined), 'extension_elapsed_s': elapsed, 'combined_active_compute_s': completion['overall_elapsed_s'] + elapsed, 'test_evaluated': False, 'validation_metrics_recomputed': True, 'historical_inputs_unchanged': True, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    text = ['AMPLIACION DE BUSQUEDA ALEATORIA RESNET-50 + BiGRU', '', combined.to_string(index=False), '', f"Seleccion por validacion: ensayo {selected['trial']}, F1 macro {selected['val_f1_macro']:.4f}.", f"Tiempo adicional: {elapsed / 60:.2f} min. Computo acumulado: {(completion['overall_elapsed_s'] + elapsed) / 60:.2f} min.", 'No se consulto ni se evaluo prueba en esta ampliacion.', 'Los resultados de prueba de la busqueda original corresponden exclusivamente a su antigua seleccion.', 'La validacion ahora fue usada para seleccionar entre mas configuraciones; posible sobreajuste de seleccion.', 'Se conservan las limitaciones: pocas epocas, una particion por clips, procedencia por partido incompleta.', 'No demuestra superioridad frente a la grilla historica ni generalizacion a partidos nuevos.', 'Verificacion: paridad de inferencia eager/grafo, F1 recalculados desde probabilidades, hashes intactos.', 'La ejecucion como grafo cambia el costo de validacion, no la arquitectura ni el protocolo de entrenamiento.']
    (HERE / 'RESULTS.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('FINAL ' + json.dumps(read(HERE / 'completion.json')), flush=True)
    print('SELECTED ' + json.dumps(selected), flush=True)
if __name__ == '__main__':
    main()
