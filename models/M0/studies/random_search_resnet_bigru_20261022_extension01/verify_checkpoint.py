"""Validate that the selected saved Keras checkpoint reproduces validation scores."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import time
import numpy as np
import pandas as pd
import models.M0.studies.random_search_resnet_bigru_20261022_extension01.run_extension as ext
def main():
    started = time.monotonic()
    completion = ext.read(ext.HERE / 'completion.json')
    assert completion['status'] == 'complete'
    selected = ext.read(ext.HERE / 'selected_configuration.json')
    config = ext.read(ext.HERE / 'configuration.json')
    assert config['script_sha256'] == ext.rs.digest(_resolve_model_artifact(ext.HERE / 'run_extension.py'))
    assert config['reused_module_sha256'] == ext.rs.digest(_resolve_model_artifact(ext.BASE / 'run_search.py'))
    for name, expected in config['split_hashes'].items():
        assert ext.rs.digest(ext.BASE / name) == expected
    combined = pd.read_csv(_resolve_model_artifact(ext.HERE / 'combined_results.csv'))
    previous = pd.read_csv(_resolve_model_artifact(ext.BASE / 'results.csv'))
    assert len(combined) <= 12
    assert not combined[['units', 'dropout', 'dense', 'learning_rate']].duplicated().any()
    assert combined.epochs.max() <= 3
    pd.testing.assert_frame_equal(combined.iloc[:len(previous)].reset_index(drop=True), previous, check_dtype=False)
    assert int(combined.loc[combined.val_f1_macro.idxmax(), 'trial']) == selected['trial']
    model = ext.rs.keras.models.load_model(_resolve_model_artifact(selected['checkpoint']), compile=False)
    val = pd.read_csv(_resolve_model_artifact(ext.BASE / 'validation.csv'))
    data = ext.rs.Data(val, ext.rs.Store())
    prob = ext.fast_predict(model, data)
    directory = (ext.BASE if selected['trial'] <= len(previous) else ext.HERE) / f"trial_{selected['trial']:02d}"
    saved = _load_model_result_array(directory / 'validation_probabilities.npy')
    np.testing.assert_allclose(prob, saved, rtol=1e-05, atol=1e-06)
    assert np.array_equal(prob.argmax(1), saved.argmax(1))
    metrics = ext.rs.metrics(data.y, prob)
    assert abs(metrics['f1_macro'] - selected['val_f1_macro']) < 1e-12
    repeat_checked = False
    if _resolve_model_artifact(ext.HERE / 'second_seed_status.json').exists():
        repeat = ext.read(ext.HERE / 'second_seed_status.json')
        if repeat['status'] == 'complete':
            repeat_prob = _load_model_result_array(ext.HERE / f"repetition_seed_{repeat['seed']}" / 'validation_probabilities.npy')
            assert abs(ext.rs.metrics(data.y, repeat_prob)['f1_macro'] - repeat['val_f1_macro']) < 1e-12
            assert repeat['no_reselection'] and (not repeat['test_evaluated'])
            repeat_checked = True
    result = {'status': 'passed', 'selected_trial': selected['trial'], 'validation_rows': len(val), 'f1_macro': metrics['f1_macro'], 'max_probability_difference': float(np.max(np.abs(prob - saved))), 'argmax_equal': True, 'old_result_rows_unchanged': True, 'split_hashes_unchanged': True, 'selected_checkpoint_reloaded': True, 'test_evaluated': False, 'second_seed_metrics_recomputed': repeat_checked, 'verification_elapsed_s': time.monotonic() - started}
    ext.rs.save(ext.HERE / 'checkpoint_verification.json', result)
    print(json.dumps(result), flush=True)
if __name__ == '__main__':
    main()
