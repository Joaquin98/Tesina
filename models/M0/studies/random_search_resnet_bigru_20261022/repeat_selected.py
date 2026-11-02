"""Optional second-seed validation repeat; never selects again or reads test."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
import time
from datetime import datetime, timezone
import pandas as pd
import models.M0.studies.random_search_resnet_bigru_20261022.run_search as rs
def main():
    env = json.loads(_resolve_model_artifact(rs.HERE / 'environment.json').read_text())
    completion = json.loads(_resolve_model_artifact(rs.HERE / 'completion.json').read_text())
    assert completion['status'] == 'complete'
    winner = json.loads(_resolve_model_artifact(rs.HERE / 'selected_configuration.json').read_text())
    plan = json.loads(_resolve_model_artifact(rs.HERE / 'budget_plan.json').read_text())
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(env['started_utc'])).total_seconds()
    remaining = rs.BUDGET - elapsed
    required = winner['duration_s'] * 1.3 + 60
    if remaining < required:
        rs.save(rs.HERE / 'second_seed_status.json', {'status': 'skipped_budget', 'remaining_s': remaining, 'required_s': required})
        print('Second seed skipped: insufficient budget', flush=True)
        return
    deadline = time.monotonic() + remaining - 30
    seed = rs.SEED + 1
    directory = rs.HERE / f'repetition_seed_{seed}'
    directory.mkdir()
    params = {key: winner[key] for key in ['units', 'dropout', 'dense', 'learning_rate']}
    rs.save(directory / 'parameters.json', {'params': params, 'seed': seed, 'role': 'Validation stability repeat only; no reselection, no test evaluation', 'original_trial': winner['trial']})
    store = rs.Store()
    train = pd.read_csv(_resolve_model_artifact(rs.HERE / 'train.csv'))
    val = pd.read_csv(_resolve_model_artifact(rs.HERE / 'validation.csv'))
    rs.keras.utils.set_random_seed(seed)
    model = rs.build(params)
    train_data = rs.Data(train, store, True, seed)
    val_data = rs.Data(val, store)
    monitor = rs.Monitor(val_data, directory, deadline)
    start = time.monotonic()
    print(f"SECOND SEED {seed}, original trial {winner['trial']}", flush=True)
    model.fit(train_data, epochs=plan['epochs_cap'], callbacks=[monitor], verbose=0, shuffle=False)
    model.load_weights(_resolve_model_artifact(directory / 'best.weights.h5'))
    model.save(directory / 'best.keras')
    scores = json.loads(_resolve_model_artifact(directory / 'validation_metrics.json').read_text())
    duration = time.monotonic() - start
    rs.save(rs.HERE / 'second_seed_status.json', {'status': 'complete', 'seed': seed, 'original_trial': winner['trial'], 'duration_s': duration, 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'val_f1_macro': scores['f1_macro'], 'delta_val_f1_macro': scores['f1_macro'] - winner['val_f1_macro'], 'no_reselection': True, 'test_evaluated': False})
    completion['second_seed_run'] = True
    completion['overall_elapsed_s'] = (datetime.now(timezone.utc) - datetime.fromisoformat(env['started_utc'])).total_seconds()
    completion['second_seed_duration_s'] = duration
    rs.save(rs.HERE / 'completion.json', completion)
    print('SECOND SEED RESULT ' + _resolve_model_artifact(rs.HERE / 'second_seed_status.json').read_text(), flush=True)
if __name__ == '__main__':
    main()
