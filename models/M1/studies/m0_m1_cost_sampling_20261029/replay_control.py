"""Replay distributed M1 x2 professional windows, without feature extraction."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import numpy as np
import models.M1.studies.m0_m1_cost_sampling_20261029.evaluate as ev
def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    job = JOBS[0]
    policy, _ = ev.policy_for(job)
    proofs = {}
    for stride in [1, 5]:
        source = ev.existing_m1(job, 'x2', stride)
        saved = read(source)
        rows = saved['window_predictions']
        cache, stats = ev.load_geometry(job, stride)
        stats[str(job['rgb'])] = stat(job['rgb'])
        ids = np.unique(np.linspace(0, len(rows) - 1, 30, dtype=int))
        predict, hashes = ev.predictor('m1', 'x2', tf)
        actual = []
        for off in range(0, len(ids), 8):
            items = []
            for i in ids[off:off + 8]:
                rgb, pose, geom = ev.old.input_window(cache, int(rows[i]['frame_start']), ev.align)
                items.append((rgb, pose, np.arange(200) < 175, geom))
            actual.extend(predict(items))
        expected = np.array([[rows[i]['probabilities'][k] for k in ev.LABELS] for i in ids])
        actual = np.asarray(actual)
        error = float(np.abs(actual - expected).max())
        assert error < 2e-05 and np.array_equal(actual.argmax(1), expected.argmax(1)), (stride, error)
        assert ev.post(rows, job, policy) == saved['detections']
        assert stats == {p: stat(p) for p in stats} and hashes == {p: sha(p) for p in hashes}
        proofs[str(stride)] = dict(windows=len(ids), max_probability_error=error, all_events_reproduced=True, model_hashes=hashes, source_result=str(source), source_result_sha256=sha(source))
        del cache, predict
        tf.keras.backend.clear_session()
    save(HERE / 'professional_m1_replay_control.json', dict(status='PASS', controls=proofs, extractors_executed=0))
    print(json.dumps(proofs), flush=True)
if __name__ == '__main__':
    main()
