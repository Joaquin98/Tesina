"""Frozen official M2 heads, complete RGB cache, new raw YOLO cache."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, time, json, hashlib, copy, gc
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='1').items():
    os.environ[k] = v
import numpy as np, joblib, tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
from core import evaluate_events
from models.M2.predictor import RGBYoloHighresQuantileFusionDetector as M2
saved_config = sys.modules.pop('config', None)
sys.path.insert(0, str(source_directory(ROOT / 'data/player_geometry_fusion')))
try:
    from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import make_variant
finally:
    sys.path.pop(0)
    if saved_config is not None:
        sys.modules['config'] = saved_config
START = time.perf_counter()

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(p, v):
    p.parent.mkdir(exist_ok=True, parents=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def progress(stage, **kw):
    save(HERE / 'progress.json', dict(stage=stage, elapsed_s=time.perf_counter() - START, updated=time.time(), **kw))

def main():
    base = ROOT / 'data/professional_rugby_m1_game1_20261024'
    deploy = ROOT / 'data/player_geometry_highres_pose/outputs/deployment_quantile_100frames'
    hist = ROOT / 'models/M0/checkpoints/best_model/best_model.keras'
    pca = ROOT / 'data/joint_multimodal_20260813/rgb_dynamics_seed42/dynamic_pca.joblib'
    reference = ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json'
    ref = read(reference)
    source = [hist, pca, deploy / 'quantile_scaler.joblib', deploy / 'logistic.joblib', reference, Path(__file__), _resolve_model_artifact(ROOT / 'models/M2/predictor.py'), _resolve_model_artifact(ROOT / 'models/M2/studies/yolo_resnet_improvement_suite_20260811/run_suite.py')]
    hashes = {str(p): sha(p) for p in source}
    for s in ref['provenance']:
        assert sha(Path(s['path'])) == s['sha256']
    d = M2.__new__(M2)
    d.dynamic_pca = joblib.load(_resolve_model_artifact(pca))
    d.quantile_scaler = joblib.load(_resolve_model_artifact(deploy / 'quantile_scaler.joblib'))
    d.auxiliary_model = joblib.load(_resolve_model_artifact(deploy / 'logistic.joblib'))
    d._make_variant = make_variant
    d.historical_weight = 0.4
    d.auxiliary_weight = 0.6
    d._window_branch_predictions = []
    chunks = []
    paths = sorted(_model_artifact_glob(HERE / 'raw_pose', '*.npz', recursive=False))
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    for p in paths:
        with _load_model_result_array(p) as f:
            chunks.append({k: f[k] for k in f.files})
    indices = np.concatenate([x['indices'] for x in chunks])
    expected = np.unique(np.rint(np.arange(0.0, 154161, 1.75)).astype(np.int32).clip(0, 154160))
    assert np.array_equal(indices, expected)
    d._yolo_source_indices = indices
    d._raw_geometry = {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0] if k != 'indices'}
    del chunks
    gc.collect()
    rgbpath = base / 'results/rgb_full.npy'
    rgb = _load_model_result_array(rgbpath, mmap_mode='r')
    assert rgb.shape == (154161, 2048)
    stats[str(rgbpath)] = [rgbpath.stat().st_size, rgbpath.stat().st_mtime_ns]
    model = tf.keras.models.load_model(_resolve_model_artifact(hist), compile=False)
    graph = tf.function(lambda a, b: model([a, b], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)], autograph=False)

    class Proxy:

        def predict(self, values, verbose=0):
            return graph(*[tf.convert_to_tensor(x) for x in values]).numpy()
    d.sequence_model = Proxy()
    starts = np.arange(0, 154161 - 174, 25)
    assert len(starts) == 6160

    def inp(a):
        pad = np.zeros((200, 2048), np.float32)
        pad[:175] = rgb[a:a + 175]
        return pad

    def auxiliary(a, pad):
        return np.concatenate([d._rgb_dynamic_vector(pad, 175), d._yolo_vector(a, a + 175)])
    controls = [0, 1260, 2695, 4942, 6000]
    error = 0
    for wi in controls:
        a = wi * 25
        pad = inp(a)
        mask = np.arange(200) < 175
        official = d._predict_sequence_window(pad, mask, a, a + 175)[0]
        ph = d._normalize(graph(pad[None], mask[None]).numpy())[0]
        pa = d._normalize(d.auxiliary_model.predict_proba(d.quantile_scaler.transform(auxiliary(a, pad)[None])))[0]
        own = d._normalize(0.4 * ph + 0.6 * pa)
        error = max(error, float(np.abs(official - own).max()))
    assert error < 1e-06
    save(HERE / 'baseline_validation.json', dict(status='PASS', official_m2_method_reproduced=True, controls=controls, max_probability_error=error))
    d._window_branch_predictions = []
    wp = HERE / 'windows.jsonl'
    rows = [json.loads(x) for x in _resolve_model_artifact(wp).read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(wp).exists() else []
    assert len(rows) <= 6160 and all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
    last = 0
    with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
        for wi in range(len(rows), len(starts), 8):
            batch = [inp(int(a)) for a in starts[wi:wi + 8]]
            masks = np.tile(np.arange(200) < 175, (len(batch), 1))
            ph = d._normalize(graph(np.stack(batch), masks).numpy())
            x = np.stack([auxiliary(int(a), pad) for a, pad in zip(starts[wi:wi + 8], batch)])
            assert x.shape[1] == 1686 and np.isfinite(x).all()
            pa = d._normalize(d.auxiliary_model.predict_proba(d.quantile_scaler.transform(x)))
            probs = d._normalize(0.4 * ph + 0.6 * pa)
            for j, p in enumerate(probs):
                a = int(starts[wi + j])
                r = dict(play_type=['line', 'other', 'scrum'][int(p.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(p.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(['line', 'other', 'scrum'], map(float, p))))
                rows.append(r)
                f.write(json.dumps(r) + '\n')
            f.flush()
            if time.perf_counter() - last > 30:
                progress('classifying_m2', windows=len(rows), total=6160)
                last = time.perf_counter()
    original = read(ROOT / 'data/results/result_20261025_argentina_allblacks_20250816_argmax.json')
    for k, v in original['event_post_processing'].items():
        setattr(d, k, v)
    events = [e for e in d._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0}) if e['end_time'] > 188.2]
    metrics = evaluate_events(ref['reference_segments'], events)
    assert hashes == {p: sha(Path(p)) for p in hashes}
    assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    for s in ref['provenance']:
        assert sha(Path(s['path'])) == s['sha256']
    save(HERE / 'events.json', events)
    save(HERE / 'metrics.json', metrics)
    save(HERE / 'completion.json', dict(status='complete', windows=len(rows), metrics=metrics, model='M2-B highres quantile100 official saved deployment', source_hashes=hashes, sources_unchanged=True, cached_rgb=True, resnet_executed=False, new_yolo_observations=len(indices), reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], elapsed_evaluation_s=time.perf_counter() - START, policy='Common M1 argmax temporal policy, not M2 default thresholds/grouping', blend=dict(historical=0.4, auxiliary=0.6)))
    progress('evaluation_complete', metrics=metrics['metrics']['total'])
if __name__ == '__main__':
    main()
