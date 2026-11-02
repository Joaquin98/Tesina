"""Second seed for a new winner, within remaining original compute budget."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import time
import pandas as pd
import models.M0.studies.random_search_resnet_bigru_20261022_extension01.run_extension as ext
def main():
    start = time.monotonic()
    complete = ext.read(ext.HERE / 'completion.json')
    assert complete['status'] == 'complete'
    selected = ext.read(ext.HERE / 'selected_configuration.json')
    if selected['trial'] <= 4:
        ext.rs.save(ext.HERE / 'second_seed_status.json', {'status': 'original_winner_retained', 'new_training': False})
        return
    remaining = 3600 - complete['combined_active_compute_s']
    required = selected['duration_s'] * 1.5 + 60
    if remaining < required:
        ext.rs.save(ext.HERE / 'second_seed_status.json', {'status': 'skipped_budget', 'remaining_s': remaining, 'required_s': required})
        return
    seed = ext.rs.SEED + 1
    directory = ext.HERE / f'repetition_seed_{seed}'
    directory.mkdir()
    params = {k: selected[k] for k in ['units', 'dropout', 'dense', 'learning_rate']}
    params['units'], params['dense'] = (int(params['units']), int(params['dense']))
    ext.rs.save(directory / 'parameters.json', {'params': params, 'seed': seed, 'original_trial': selected['trial'], 'purpose': 'Validation-only repetition; frozen selection; no test evaluation', 'epochs_cap': 3})
    ext.rs.predict = ext.fast_predict
    ext.rs.keras.utils.set_random_seed(seed)
    model = ext.rs.build(params)
    store = ext.rs.Store()
    train = pd.read_csv(_resolve_model_artifact(study_csv_path(ext.CURRENT_DATASET / 'train.csv')))
    val = pd.read_csv(_resolve_model_artifact(study_csv_path(ext.BASE / 'validation.csv')))
    train_data = ext.rs.Data(train, store, True, seed)
    val_data = ext.rs.Data(val, store)
    monitor = ext.rs.Monitor(val_data, directory, start + remaining - 30)
    print(f"SECOND SEED {seed}: trial {selected['trial']}", flush=True)
    model.fit(train_data, epochs=3, callbacks=[monitor], verbose=0, shuffle=False)
    model.load_weights(_resolve_model_artifact(study_csv_path(directory / 'best.weights.h5')))
    model.save(directory / 'best.keras')
    scores = ext.read(directory / 'validation_metrics.json')
    elapsed = time.monotonic() - start
    status = {'status': 'complete', 'seed': seed, 'original_trial': selected['trial'], 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'val_f1_macro': scores['f1_macro'], 'delta_val_f1_macro': scores['f1_macro'] - selected['val_f1_macro'], 'elapsed_s': elapsed, 'no_reselection': True, 'test_evaluated': False, 'original_winner_second_seed_f1': ext.read(ext.BASE / 'second_seed_status.json')['val_f1_macro']}
    ext.rs.save(ext.HERE / 'second_seed_status.json', status)
    complete['combined_active_compute_s'] += elapsed
    complete['new_winner_second_seed'] = status
    ext.rs.save(ext.HERE / 'completion.json', complete)
    with _resolve_model_artifact(study_csv_path(ext.HERE / 'RESULTS.txt')).open('a', encoding='utf-8') as f:
        f.write(f"\nRepeticion de la nueva ganadora con semilla {seed}: F1 macro {scores['f1_macro']:.4f}, diferencia {status['delta_val_f1_macro']:+.4f}.\n")
        f.write(f"Computo acumulado final incluyendo esta repeticion: {complete['combined_active_compute_s'] / 60:.2f} min.\n")
        f.write('La repeticion no modifica la seleccion. No se evalua prueba.\n')
    print('SECOND SEED RESULT ' + str(status), flush=True)
if __name__ == '__main__':
    main()
