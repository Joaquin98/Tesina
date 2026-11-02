"""200 sequential, resumable random trials; TensorFlow restarts every 20 trials."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
BASE = HERE.parent / 'random_search_resnet_bigru_20261022'
PREVIOUS = HERE.parent / 'random_search_resnet_bigru_20261022_extension03'
N, FIRST, CHUNK, BUDGET, SAMPLE_SEED = (200, 65, 20, 8 * 3600, 20261027)

def read(path):
    return json.loads(_resolve_model_artifact(Path(path)).read_text(encoding='utf-8'))

def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(_resolve_model_artifact(Path(path)).read_bytes()).hexdigest()

def old_directory(trial):
    trial = int(trial)
    suffix = '' if trial <= 4 else '_extension01' if trial <= 12 else '_extension02' if trial <= 24 else '_extension03'
    return HERE.parent / ('random_search_resnet_bigru_20261022' + suffix) / f'trial_{trial:02d}'

def complete_rows():
    return [read(p)['row'] for p in sorted(_model_artifact_glob(HERE, 'trial_*/complete.json', recursive=False))]

def tables():
    old = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'combined_results.csv'))
    rows = sorted(complete_rows(), key=lambda r: r['trial'])
    if rows:
        new = pd.DataFrame(rows)
        combined = pd.concat([old, new], ignore_index=True)
        for name, frame in [('results.csv', new), ('combined_results.csv', combined)]:
            temp = HERE / (name + '.tmp')
            frame.to_csv(temp, index=False)
            temp.replace(HERE / name)
    return (old, rows)

def engine():
    import models.M0.studies.random_search_resnet_bigru_20261022_extension04.frozen_core as rs
    import tensorflow as tf

    def predict(model, data):
        if not hasattr(model, '_batch_infer'):
            model._batch_infer = tf.function(lambda f, m: model([f, m], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)])
        return np.concatenate([model._batch_infer(*data[i][0]).numpy() for i in range(len(data))])
    rs.predict = predict
    return rs

def initialize():
    config_path = HERE / 'configuration.json'
    if _resolve_model_artifact(config_path).exists():
        config = read(config_path)
        assert config['script_sha256'] == digest(__file__)
        assert config['frozen_core_sha256'] == digest(_resolve_model_artifact(HERE / 'frozen_core.py'))
        return config
    old = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'combined_results.csv'))
    assert len(old) == 64
    seen = {tuple((row[k] for k in ['units', 'dropout', 'dense', 'learning_rate'])) for row in old.to_dict('records')}
    rng = np.random.default_rng(SAMPLE_SEED)
    candidates = []
    while len(candidates) < N:
        p = {'units': int(rng.choice([32, 64, 128])), 'dropout': float(rng.uniform(0.2, 0.5)), 'dense': int(rng.choice([0, 64])), 'learning_rate': float(10 ** rng.uniform(-4, -3))}
        key = tuple(p.values())
        if key not in seen:
            seen.add(key)
            candidates.append(p)
    train = pd.read_csv(_resolve_model_artifact(HERE / 'inputs/train.csv'))
    val = pd.read_csv(_resolve_model_artifact(HERE / 'inputs/validation.csv'))
    normalize = lambda v: str(v).strip().casefold()
    assert not set(train.video_name.map(normalize)) & set(val.video_name.map(normalize))
    assert val.video_name.nunique() == len(val) == 980
    assert len(train) == 1260
    hashes = {name: digest(HERE / 'inputs' / name) for name in ['train.csv', 'validation.csv']}
    for name, expected in hashes.items():
        assert digest(BASE / name) == expected
    config = {'started_utc': datetime.now(timezone.utc).isoformat(), 'request': '200 more configurations', 'new_trials': N, 'first_trial': FIRST, 'last_trial': FIRST + N - 1, 'candidates': candidates, 'sampling_seed': SAMPLE_SEED, 'initialization_seed': 20261022, 'split_seed': 42, 'epochs_cap': 3, 'patience': 2, 'batch_size': 32, 'sequence_length': 200, 'extractor': 'resnet50', 'architecture': 'BiGRU(u, return_sequences=True), BiGRU(u/2), Dropout, optional Dense, softmax', 'optimizer': 'Adam', 'selection': 'validation macro F1 only; earliest exact tie', 'budget_s': BUDGET, 'worker_chunk_size': CHUNK, 'intra_threads': 4, 'inter_threads': 1, 'split_hashes': hashes, 'prior_csv_sha256': digest(PREVIOUS / 'combined_results.csv'), 'script_sha256': digest(__file__), 'frozen_core_sha256': digest(_resolve_model_artifact(HERE / 'frozen_core.py')), 'frozen_core_original_sha256': digest(_resolve_model_artifact(BASE / 'run_search.py')), 'python': sys.executable, 'previous_active_compute_s': read(PREVIOUS / 'completion.json')['cumulative_active_compute_s'], 'test_policy': 'No test data or scores read or evaluated. Frozen train/validation only.', 'group_limitation': 'No complete verified match provenance; same clip split as prior runs.', 'resampling': 'Same frozen balanced training manifest; validation untouched.'}
    assert config['frozen_core_sha256'] == config['frozen_core_original_sha256']
    save(config_path, config)
    return config

def worker(first, last, deadline_unix):
    config = read(HERE / 'configuration.json')
    assert config['frozen_core_sha256'] == digest(_resolve_model_artifact(HERE / 'frozen_core.py'))
    rs = engine()
    train = pd.read_csv(_resolve_model_artifact(HERE / 'inputs/train.csv'))
    val = pd.read_csv(_resolve_model_artifact(HERE / 'inputs/validation.csv'))
    store = rs.Store()
    deadline = time.monotonic() + max(0, deadline_unix - time.time())
    save(HERE / 'environment.json', {'tensorflow': rs.tf.__version__, 'keras': rs.keras.__version__, 'numpy': np.__version__, 'python': sys.executable, 'platform': rs.platform.platform(), 'devices': [str(d) for d in rs.tf.config.list_physical_devices()], 'cache': str(rs.STORE), 'cache_shape': list(store.features.shape)})

    class LiveMonitor(rs.Monitor):

        def on_epoch_end(self, epoch, logs=None):
            super().on_epoch_end(epoch, logs)
            save(HERE / 'live_status.json', {'state': 'training', 'trial': trial, 'new_index': trial - FIRST + 1, 'epoch': epoch + 1, 'val_f1_macro': self.rows[-1]['val_f1_macro'], 'best_val_f1_macro': self.best, 'updated_utc': datetime.now(timezone.utc).isoformat()})
    for trial in range(first, last + 1):
        parent = HERE / f'trial_{trial:03d}'
        parent.mkdir(exist_ok=True)
        if _resolve_model_artifact(parent / 'complete.json').exists():
            continue
        if time.monotonic() + 240 > deadline:
            raise RuntimeError('Insufficient remaining budget to admit another full trial')
        p = config['candidates'][trial - FIRST]
        attempts = list(_model_artifact_glob(parent, 'attempt_*', recursive=False))
        directory = parent / f'attempt_{len(attempts) + 1:02d}'
        directory.mkdir()
        save(directory / 'parameters.json', p)
        save(HERE / 'live_status.json', {'state': 'starting_trial', 'trial': trial, 'new_index': trial - FIRST + 1, 'updated_utc': datetime.now(timezone.utc).isoformat()})
        rs.keras.backend.clear_session()
        rs.keras.utils.set_random_seed(rs.SEED)
        model = rs.build(p)
        train_data = rs.Data(train, store, True)
        val_data = rs.Data(val, store)
        start = time.monotonic()
        x, _ = val_data[0]
        eager = model(x, training=False).numpy()
        graph = rs.predict(model, rs.Data(val.iloc[:32], store))
        np.testing.assert_allclose(eager, graph, rtol=1e-05, atol=1e-06)
        assert np.array_equal(eager.argmax(1), graph.argmax(1))
        save(directory / 'inference_parity.json', {'rows': 32, 'argmax_equal': True, 'max_probability_difference': float(np.max(np.abs(eager - graph)))})
        monitor = LiveMonitor(val_data, directory, deadline - 30)
        print(f'TRIAL {trial}/{FIRST + N - 1} {p}', flush=True)
        model.fit(train_data, epochs=3, callbacks=[monitor], verbose=0, shuffle=False)
        model.load_weights(_resolve_model_artifact(directory / 'best.weights.h5'))
        model.save(directory / 'best.keras')
        scores = read(directory / 'validation_metrics.json')
        row = {'trial': trial, **p, 'seed': rs.SEED, 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'duration_s': time.monotonic() - start, **{f'val_{k}': v for k, v in scores.items() if isinstance(v, float)}}
        save(parent / 'complete.json', {'status': 'complete', 'row': row, 'artifact_dir': str(directory), 'checkpoint': str(directory / 'best.keras'), 'finished_utc': datetime.now(timezone.utc).isoformat()})
        tables()
        save(HERE / 'live_status.json', {'state': 'trial_complete', 'trial': trial, 'new_index': trial - FIRST + 1, 'val_f1_macro': row['val_f1_macro'], 'updated_utc': datetime.now(timezone.utc).isoformat()})
        print('TRIAL RESULT ' + json.dumps(row), flush=True)
        del model, monitor, train_data, val_data
        rs.keras.backend.clear_session()
        gc.collect()

def finalize(config, elapsed):
    old, rows = tables()
    assert len(rows) == N
    combined = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    assert not combined[['units', 'dropout', 'dense', 'learning_rate']].duplicated().any()
    pd.testing.assert_frame_equal(combined.iloc[:64].reset_index(drop=True), old, check_dtype=False)
    y = pd.read_csv(_resolve_model_artifact(HERE / 'inputs/validation.csv')).tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
    from sklearn.metrics import f1_score
    for row in combined.to_dict('records'):
        trial = int(row['trial'])
        directory = old_directory(trial) if trial < FIRST else Path(read(HERE / f'trial_{trial:03d}/complete.json')['artifact_dir'])
        pred = _load_model_result_array(directory / 'validation_probabilities.npy').argmax(1)
        assert abs(f1_score(y, pred, average='macro', labels=[0, 1, 2]) - row['val_f1_macro']) < 1e-12
        history = pd.read_csv(_resolve_model_artifact(directory / 'history.csv'))
        assert abs(history.val_f1_macro.max() - row['val_f1_macro']) < 1e-12
        assert int(history.loc[history.val_f1_macro.idxmax(), 'epoch']) == row['best_epoch']
        assert _resolve_model_artifact(directory / 'best.keras').exists()
    for name, expected in config['split_hashes'].items():
        assert digest(HERE / 'inputs' / name) == expected
    assert digest(PREVIOUS / 'combined_results.csv') == config['prior_csv_sha256']
    winner = combined.loc[combined.val_f1_macro.idxmax()].to_dict()
    for k in ['trial', 'units', 'dense', 'seed', 'epochs', 'best_epoch']:
        winner[k] = int(winner[k])
    d = old_directory(winner['trial']) if winner['trial'] < FIRST else Path(read(HERE / f"trial_{winner['trial']:03d}/complete.json")['artifact_dir'])
    winner['checkpoint'] = str(d / 'best.keras')
    winner['artifact_dir'] = str(d)
    winner['criterion'] = 'Validation macro F1 only; no test evaluation'
    save(HERE / 'selected_configuration.json', winner)
    rs = engine()
    model = rs.keras.models.load_model(_resolve_model_artifact(winner['checkpoint']), compile=False)
    prob = rs.predict(model, rs.Data(pd.read_csv(_resolve_model_artifact(HERE / 'inputs/validation.csv')), rs.Store()))
    saved = _load_model_result_array(d / 'validation_probabilities.npy')
    np.testing.assert_allclose(prob, saved, rtol=1e-05, atol=1e-06)
    assert np.array_equal(prob.argmax(1), saved.argmax(1))
    save(HERE / 'verification.json', {'status': 'passed', 'trials_verified': len(combined), 'prior_rows_unchanged': True, 'prior_csv_hash_unchanged': True, 'frozen_split_hashes_unchanged': True, 'metrics_recomputed_independently': True, 'selected_checkpoint_reloaded': True, 'max_probability_difference': float(np.max(np.abs(prob - saved))), 'test_evaluated': False})
    combined.to_csv(HERE / 'summary.csv', index=False)
    status = {'status': 'complete', 'new_trials_completed': N, 'total_configurations': len(combined), 'batch_elapsed_s': elapsed, 'batch_budget_s': BUDGET, 'selected_trial': winner['trial'], 'cumulative_active_compute_s': config['previous_active_compute_s'] + elapsed, 'test_evaluated': False, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    save(HERE / 'completion.json', status)
    text = ['200 CONFIGURACIONES ADICIONALES: RESNET-50 + BiGRU', '', pd.DataFrame(rows).sort_values('val_f1_macro', ascending=False).to_string(index=False), '', f"Seleccion final: ensayo {winner['trial']}, F1 macro {winner['val_f1_macro']:.4f}.", f"Mejor previo: {old.val_f1_macro.max():.4f}; diferencia {winner['val_f1_macro'] - old.val_f1_macro.max():+.4f}.", f'Tiempo de tanda: {elapsed / 3600:.2f} horas.', 'Todas las configuraciones usan el protocolo previo: tres epocas, lote 32, 200 fotogramas, paciencia 2.', 'Las particiones y caracteristicas permanecen congeladas. Solo se balancea entrenamiento.', 'No se evalua prueba. La prueba historica ya fue consultada antes y sus resultados antiguos no pertenecen a esta nueva seleccion.', 'Mas candidatos sobre la misma validacion aumentan el riesgo de sobreajuste de seleccion.', 'Tres epocas no demuestran convergencia; dos semillas tampoco demostrarian estabilidad estadistica.', 'No hay independencia verificada por partido ni prueba externa nueva ni comparacion controlada con la grilla historica.', 'Resultados previos preservados; metricas recalculadas y checkpoint seleccionado recargado.']
    (HERE / 'RESULTS.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('FINAL ' + json.dumps(status), flush=True)

def coordinator():
    config = initialize()
    if _resolve_model_artifact(HERE / 'completion.json').exists() and read(HERE / 'completion.json')['status'] == 'complete':
        print('Already complete; artifacts preserved.', flush=True)
        return
    start = time.monotonic()
    spent = sum((r['duration_s'] for r in complete_rows()))
    if _resolve_model_artifact(HERE / 'interruption_audit.json').exists():
        spent += read(HERE / 'interruption_audit.json')['prior_overhead_s']
    save(HERE / 'coordinator_status.json', {'state': 'running', 'pid': os.getpid(), 'resumed_completed': len(complete_rows()), 'started_utc': datetime.now(timezone.utc).isoformat()})
    for first in range(FIRST, FIRST + N, CHUNK):
        last = min(first + CHUNK - 1, FIRST + N - 1)
        if all((_resolve_model_artifact(HERE / f'trial_{t:03d}/complete.json').exists() for t in range(first, last + 1))):
            continue
        remaining = BUDGET - spent - (time.monotonic() - start)
        if remaining < 240:
            save(HERE / 'completion.json', {'status': 'budget_exhausted', 'completed': len(complete_rows())})
            return
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(first), str(last), '--deadline-unix', str(time.time() + remaining)]
        for attempt in range(3):
            print(f'CHUNK {first}-{last}, process attempt {attempt + 1}', flush=True)
            log_path = HERE / f'worker_{first:03d}_{last:03d}_{time.time_ns()}.log'
            with _resolve_model_artifact(log_path).open('w', encoding='utf-8') as worker_log:
                process = subprocess.Popen(command, stdout=worker_log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                save(HERE / 'coordinator_status.json', {'state': 'running', 'pid': os.getpid(), 'worker_pid': process.pid, 'worker_log': str(log_path), 'chunk': [first, last], 'completed': len(complete_rows()), 'updated_utc': datetime.now(timezone.utc).isoformat()})
                returncode = process.wait()
            if returncode == 0:
                break
            print(f'Worker exited {returncode}; completed trials preserved. See {log_path.name}.', flush=True)
        else:
            save(HERE / 'completion.json', {'status': 'worker_failed', 'completed': len(complete_rows()), 'failed_chunk': [first, last]})
            return
        save(HERE / 'coordinator_status.json', {'state': 'running', 'pid': os.getpid(), 'completed': len(complete_rows()), 'elapsed_s': spent + time.monotonic() - start, 'updated_utc': datetime.now(timezone.utc).isoformat()})
    elapsed = spent + time.monotonic() - start
    finalize(config, elapsed)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2, type=int)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, args.deadline_unix)
    else:
        coordinator()
