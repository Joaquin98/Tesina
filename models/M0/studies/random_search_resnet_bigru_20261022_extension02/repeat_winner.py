"""Repeat a newly selected winner without changing the frozen selection."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import time
import pandas as pd
import models.M0.studies.random_search_resnet_bigru_20261022_extension02.run_batch as batch
def main():
    start = time.monotonic()
    rs, shared, here = (batch.rs, batch.shared, batch.HERE)
    done = shared.read(here / 'completion.json')
    assert done['status'] == 'complete'
    winner = shared.read(here / 'selected_configuration.json')
    remaining = batch.BUDGET - done['batch_elapsed_s']
    if winner['trial'] <= 12 or remaining < winner['duration_s'] * 1.5 + 60:
        rs.save(here / 'second_seed_status.json', {'status': 'skipped', 'reason': 'Previous winner retained or insufficient budget', 'new_training': False})
        return
    seed = rs.SEED + 1
    directory = here / f'repetition_seed_{seed}'
    directory.mkdir()
    params = {k: winner[k] for k in ['units', 'dropout', 'dense', 'learning_rate']}
    rs.save(directory / 'parameters.json', {'params': params, 'seed': seed, 'original_trial': winner['trial'], 'epochs_cap': 3, 'role': 'Validation-only repeat; no reselection or test evaluation'})
    rs.predict = shared.fast_predict
    rs.keras.utils.set_random_seed(seed)
    model = rs.build(params)
    store = rs.Store()
    train = pd.read_csv(_resolve_model_artifact(rs.HERE / 'train.csv'))
    val = pd.read_csv(_resolve_model_artifact(rs.HERE / 'validation.csv'))
    train_data = rs.Data(train, store, True, seed)
    val_data = rs.Data(val, store)
    monitor = rs.Monitor(val_data, directory, start + remaining - 30)
    print(f"REPEAT trial={winner['trial']} seed={seed}", flush=True)
    model.fit(train_data, epochs=3, callbacks=[monitor], verbose=0, shuffle=False)
    model.load_weights(_resolve_model_artifact(directory / 'best.weights.h5'))
    model.save(directory / 'best.keras')
    prob = batch.np.load(_resolve_model_artifact(directory / 'validation_probabilities.npy'))
    scores = rs.metrics(val_data.y, prob)
    saved = shared.read(directory / 'validation_metrics.json')
    assert abs(scores['f1_macro'] - saved['f1_macro']) < 1e-12
    elapsed = time.monotonic() - start
    result = {'status': 'complete', 'seed': seed, 'original_trial': winner['trial'], 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'val_f1_macro': scores['f1_macro'], 'delta_val_f1_macro': scores['f1_macro'] - winner['val_f1_macro'], 'elapsed_s': elapsed, 'no_reselection': True, 'test_evaluated': False, 'metrics_recomputed': True}
    rs.save(here / 'second_seed_status.json', result)
    done['new_winner_second_seed'] = result
    done['phase_elapsed_s'] = done['batch_elapsed_s'] + elapsed
    done['cumulative_active_compute_s'] += elapsed
    rs.save(here / 'completion.json', done)
    with _resolve_model_artifact(here / 'RESULTS.txt').open('a', encoding='utf-8') as f:
        f.write(f"\nRepeticion separada con semilla {seed}: F1 macro {scores['f1_macro']:.4f}; diferencia {result['delta_val_f1_macro']:+.4f}.\n")
        f.write(f"Tiempo de esta tanda incluida la repeticion: {done['phase_elapsed_s'] / 60:.2f} min.\n")
        f.write(f"Computo acumulado: {done['cumulative_active_compute_s'] / 60:.2f} min.\n")
        f.write('La repeticion no modifica la seleccion y no incluye prueba.\n')
    print('REPEAT RESULT ' + str(result), flush=True)
if __name__ == '__main__':
    main()
