"""Twelve additional frozen random candidates, validation-only, 30-minute cap."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
HERE = experiment_directory(__file__)
PREVIOUS = HERE.parent / 'random_search_resnet_bigru_20261022_extension01'
sys.path.insert(0, str(source_directory(PREVIOUS)))
import run_extension as shared
rs = shared.rs
import numpy as np
import pandas as pd
BUDGET = 1800
N = 12
SAMPLING_SEED = 20261024

def location(trial):
    parent = rs.HERE if trial <= 4 else PREVIOUS if trial <= 12 else HERE
    return parent / f'trial_{int(trial):02d}'

def draw(previous):
    rng = np.random.default_rng(SAMPLING_SEED)
    seen = {tuple((row[k] for k in ['units', 'dropout', 'dense', 'learning_rate'])) for row in previous.to_dict('records')}
    candidates = []
    while len(candidates) < N:
        p = {'units': int(rng.choice([32, 64, 128])), 'dropout': float(rng.uniform(0.2, 0.5)), 'dense': int(rng.choice([0, 64])), 'learning_rate': float(10 ** rng.uniform(-4, -3))}
        key = tuple(p.values())
        if key not in seen:
            seen.add(key)
            candidates.append(p)
    return candidates

def main():
    if _resolve_model_artifact(HERE / 'configuration.json').exists():
        raise RuntimeError('Existing run preserved. Use a new directory to rerun.')
    start = time.monotonic()
    deadline = start + BUDGET
    previous = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'combined_results.csv'))
    assert len(previous) == 12
    previous_time = shared.read(PREVIOUS / 'completion.json')['combined_active_compute_s']
    audit = shared.read(rs.HERE / 'split_audit.json')
    for path, expected in audit['source_hashes'].items():
        assert rs.digest(rs.ROOT / path) == expected
    train = pd.read_csv(_resolve_model_artifact(rs.HERE / 'train.csv'))
    val = pd.read_csv(_resolve_model_artifact(rs.HERE / 'validation.csv'))
    assert not set(train.video_name) & set(val.video_name)
    assert not val.video_name.duplicated().any()
    split_hashes = {name: rs.digest(rs.HERE / name) for name in ['train.csv', 'validation.csv']}
    candidates = draw(previous)
    rs.save(HERE / 'configuration.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Additional search candidates following the initial 12 configurations', 'sampling_seed': SAMPLING_SEED, 'initialization_seed': rs.SEED, 'new_candidates': candidates, 'budget_s': BUDGET, 'epochs_cap': 3, 'patience': 2, 'batch_size': 32, 'sequence_length': 200, 'extractor': 'resnet50', 'architecture': 'Two stacked BiGRU: units and units/2', 'selection': 'Validation macro F1; earliest trial breaks exact ties', 'resampling': 'Same frozen training manifest; validation unmodified', 'split_hashes': split_hashes, 'prior_result_hash': rs.digest(PREVIOUS / 'combined_results.csv'), 'script_sha256': rs.digest(__file__), 'shared_module_sha256': rs.digest(_resolve_model_artifact(PREVIOUS / 'run_extension.py')), 'test_policy': 'Only byte hashes of historical source CSVs for integrity. No test labels parsed, no test features/predictions/scores accessed or test evaluation.', 'python': sys.executable, 'tensorflow': shared.tf.__version__, 'keras': rs.keras.__version__, 'intra_threads': 4, 'inter_threads': 1, 'previous_active_compute_s': previous_time})
    store = rs.Store()
    rs.predict = shared.fast_predict
    results = []
    stop_reason = 'all_candidates_completed'
    for trial, p in enumerate(candidates, 13):
        estimate = max((r['duration_s'] / (1 + r['units'] / 128) for r in results)) * (1 + p['units'] / 128) * 1.5 if results else 180.0
        if deadline - time.monotonic() < max(90.0, estimate) + 45:
            stop_reason = 'budget_admission'
            print(f'STOP before trial {trial}: remaining={deadline - time.monotonic():.1f}s estimated={estimate:.1f}s', flush=True)
            break
        directory = location(trial)
        directory.mkdir()
        rs.save(directory / 'parameters.json', p)
        rs.keras.backend.clear_session()
        rs.keras.utils.set_random_seed(rs.SEED)
        model = rs.build(p)
        train_data = rs.Data(train, store, True)
        val_data = rs.Data(val, store)
        t = time.monotonic()
        x, _ = val_data[0]
        eager = model(x, training=False).numpy()
        graph = shared.fast_predict(model, rs.Data(val.iloc[:32], store))
        np.testing.assert_allclose(eager, graph, rtol=1e-05, atol=1e-06)
        assert np.array_equal(eager.argmax(1), graph.argmax(1))
        rs.save(directory / 'inference_parity.json', {'rows': 32, 'argmax_equal': True, 'max_probability_difference': float(np.max(np.abs(eager - graph)))})
        monitor = rs.Monitor(val_data, directory, deadline - 45)
        print(f'TRIAL {trial}/24 {p}', flush=True)
        model.fit(train_data, epochs=3, callbacks=[monitor], verbose=0, shuffle=False)
        model.load_weights(_resolve_model_artifact(directory / 'best.weights.h5'))
        model.save(directory / 'best.keras')
        scores = shared.read(directory / 'validation_metrics.json')
        row = {'trial': trial, **p, 'seed': rs.SEED, 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'duration_s': time.monotonic() - t, **{f'val_{k}': v for k, v in scores.items() if isinstance(v, float)}}
        results.append(row)
        pd.DataFrame(results).to_csv(HERE / 'results.csv', index=False)
        pd.concat([previous, pd.DataFrame(results)], ignore_index=True).to_csv(HERE / 'combined_results.csv', index=False)
        print('TRIAL RESULT ' + json.dumps(row), flush=True)
        del model, monitor, train_data, val_data
        rs.keras.backend.clear_session()
        gc.collect()
    if not results:
        rs.save(HERE / 'completion.json', {'status': 'no_trial_admitted', 'elapsed_s': time.monotonic() - start})
        return
    combined = pd.concat([previous, pd.DataFrame(results)], ignore_index=True)
    combined.to_csv(HERE / 'summary.csv', index=False)
    winner = combined.loc[combined.val_f1_macro.idxmax()].to_dict()
    for key in ['trial', 'units', 'dense', 'seed', 'epochs', 'best_epoch']:
        winner[key] = int(winner[key])
    winner['checkpoint'] = str(location(winner['trial']) / 'best.keras')
    winner['criterion'] = 'Validation macro F1 only; no new test evaluation'
    rs.save(HERE / 'selected_configuration.json', winner)
    y = np.array([rs.LABELS.index(tag) for tag in val.tag])
    for row in combined.to_dict('records'):
        d = location(row['trial'])
        prob = _load_model_result_array(d / 'validation_probabilities.npy')
        assert abs(rs.metrics(y, prob)['f1_macro'] - row['val_f1_macro']) < 1e-12
        history = pd.read_csv(_resolve_model_artifact(d / 'history.csv'))
        assert abs(history.val_f1_macro.max() - row['val_f1_macro']) < 1e-12
        assert int(history.loc[history.val_f1_macro.idxmax(), 'epoch']) == row['best_epoch']
        assert _resolve_model_artifact(d / 'best.keras').exists()
    pd.testing.assert_frame_equal(combined.iloc[:12].reset_index(drop=True), previous, check_dtype=False)
    assert not combined[['units', 'dropout', 'dense', 'learning_rate']].duplicated().any()
    for path, expected in audit['source_hashes'].items():
        assert rs.digest(rs.ROOT / path) == expected
    for name, expected in split_hashes.items():
        assert rs.digest(rs.HERE / name) == expected
    model = rs.keras.models.load_model(_resolve_model_artifact(winner['checkpoint']), compile=False)
    prob = shared.fast_predict(model, rs.Data(val, store))
    saved = _load_model_result_array(location(winner['trial']) / 'validation_probabilities.npy')
    np.testing.assert_allclose(prob, saved, rtol=1e-05, atol=1e-06)
    assert np.array_equal(prob.argmax(1), saved.argmax(1))
    rs.save(HERE / 'verification.json', {'status': 'passed', 'trials_verified': len(combined), 'prior_rows_unchanged': True, 'split_hashes_unchanged': True, 'source_hashes_unchanged': True, 'selected_checkpoint_reloaded': True, 'validation_rows': len(val), 'max_checkpoint_probability_difference': float(np.max(np.abs(prob - saved))), 'selected_val_f1_macro': rs.metrics(y, prob)['f1_macro'], 'test_evaluated': False})
    elapsed = time.monotonic() - start
    status = {'status': 'complete', 'stop_reason': stop_reason, 'new_trials_completed': len(results), 'total_configurations': len(combined), 'batch_elapsed_s': elapsed, 'batch_budget_s': BUDGET, 'cumulative_active_compute_s': previous_time + elapsed, 'test_evaluated': False, 'selected_trial': winner['trial'], 'finished_utc': datetime.now(timezone.utc).isoformat()}
    rs.save(HERE / 'completion.json', status)
    before = float(previous.val_f1_macro.max())
    text = ['NUEVA TANDA DE BUSQUEDA ALEATORIA RESNET-50 + BiGRU', '', pd.DataFrame(results).to_string(index=False), '', f'Configuraciones acumuladas: {len(combined)}. Nuevas: {len(results)}.', f"Mejor F1 macro de validacion: {winner['val_f1_macro']:.4f} (ensayo {winner['trial']}).", f"Mejor previo: {before:.4f}. Diferencia: {winner['val_f1_macro'] - before:+.4f}.", f"F1/sensibilidad scrum: {winner['val_scrum_f1']:.4f}/{winner['val_scrum_recall']:.4f}.", f"F1/sensibilidad lineout: {winner['val_line_f1']:.4f}/{winner['val_line_recall']:.4f}.", f'Tiempo de esta tanda: {elapsed / 60:.2f} min; acumulado: {(previous_time + elapsed) / 60:.2f} min.', 'Seleccion exclusivamente por F1 macro de validacion. Ninguna evaluacion nueva de prueba.', 'Se mantienen tres epocas, paciencia 2, lote 32 y secuencias originales de 200 fotogramas.', 'Semilla nueva de muestreo 20261024; semilla de inicializacion/orden 20261022 como antes.', 'Mismas particiones: clips unicos antes de balancear solo entrenamiento. Procedencia', 'por partido incompleta: no se garantiza independencia entre partidos.', 'Al aumentar las configuraciones seleccionadas sobre la misma validacion aumenta', 'el riesgo de sobreajuste de seleccion. No es una prueba de generalizacion externa,', 'ni de convergencia en tres epocas ni de superioridad sobre la grilla historica.', 'Verificados los F1 desde probabilidades, mejores epocas, ausencia de configuraciones', 'repetidas y hashes. El checkpoint elegido se recargo y reprodujo sus predicciones.', 'Los resultados anteriores y la tesina se conservan.']
    (HERE / 'RESULTS.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('FINAL ' + json.dumps(status), flush=True)
    print('WINNER ' + json.dumps(winner), flush=True)
if __name__ == '__main__':
    main()
