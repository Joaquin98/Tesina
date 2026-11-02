"""Evaluate a frozen, completed arm early; remaining fixed training is unchanged."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, time, traceback, json
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT)]
import models.M1.studies.m1_line_only_costs_20261028.evaluate as e
from models.M1.studies.m1_line_only_costs_20261028.run import read, save, sha
START = time.perf_counter()

class ResourcePressure(Exception):
    pass

def progress(stage, **kw):
    import psutil
    available = psutil.virtual_memory().available / 1000000000.0
    save(HERE / 'progress_partial_line2.json', dict(stage=stage, elapsed_s=time.perf_counter() - START, updated_unix=time.time(), available_ram_gb=available, **kw))
    if available < 0.65:
        raise ResourcePressure('Partial evaluation paused to preserve memory; main pipeline will resume cached rows')

def initialize():
    import tensorflow as tf, psutil
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    return tf

def main():
    model = read(HERE / 'models/line2_scrum1/complete.json')
    assert model['epochs'] == 5 and model['line_cost'] == 2 and (model['scrum_cost'] == 1)
    assert all((sha(p) == h for p, h in model['model_hashes'].items()))
    save(HERE / 'evaluation_amendment.json', dict(reason='Partial evaluation of completed training arms; remaining arms are pending', partial_line_costs=[2], frozen_checkpoint=model['model_hashes'], other_training_hyperparameters_unchanged=True, no_evaluation_based_checkpoint_selection=True, all_checkpoints_before_evaluation_applies_to_main_scheduled_run_only=True, inference_threads=4, priority='idle', started_unix=time.time()))
    e.progress = progress
    e.initialize = initialize
    frozen = dict(status='complete', models=[model], all_models_saved_before_evaluation_unix=time.time(), scope='completed line2 arm only')
    e.main(costs=(2,), training_record=frozen, completion_file='partial_line2_complete.json')
    progress('partial_complete')
if __name__ == '__main__':
    try:
        main()
    except ResourcePressure as exc:
        save(HERE / 'progress_partial_line2.json', dict(stage='paused_for_memory', reason=str(exc), updated_unix=time.time()))
    except Exception:
        save(HERE / 'partial_line2_error.json', dict(error=traceback.format_exc()))
        raise
