"""Isolated cost pilot using the compiled extractor used in full extraction."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, time
from pathlib import Path
import numpy as np
from models.M0.studies.resnet_resolution_20261008.experiment import CONFIG, HERE, OUT, Peak, extractor, cv2_resize_rgb, write_json
from models.M0.studies.resnet_resolution_20261008.preflight import sources, probe, sample_indices, native_frames
size = int(sys.argv[1])
h, f = sources()
examples = [('CH', HERE.parents[1] / 'data/7seg unique' / h.iloc[0].video_name, None), ('CV', Path(f.iloc[0].source_video), f.iloc[0])]
rois = []
for domain, path, row in examples:
    indices = sample_indices(probe(path), domain, row)[:8]
    rois += native_frames(path, indices, domain)
x = np.asarray([cv2_resize_rgb(v, size) for v in rois], np.float32)
t = time.perf_counter()
with Peak() as peak:
    model, infer = extractor(size)
    results = []
    times = []
    for first in range(0, len(x), CONFIG['feature_batch_size']):
        tick = time.perf_counter()
        results.append(infer(x[first:first + CONFIG['feature_batch_size']]).numpy())
        times.append(time.perf_counter() - tick)
    values = np.concatenate(results)
    with _load_model_result_array(OUT / 'smoke' / f'features_{size}.npz') as original:
        maximum = float(np.max(np.abs(values - original['features'])))
        np.testing.assert_allclose(values, original['features'], rtol=0.0001, atol=0.002)
r = dict(status='passed', resolution=size, frames=len(x), inference_s=sum(times), s_per_frame=sum(times) / len(x), elapsed_with_model_init_s=time.perf_counter() - t, peak_rss_bytes=peak.value, max_abs_vs_eager=maximum, note='Separate process per resolution.16 source frames and equal batch8 warmup. This is a small cost pilot, not a scientific accuracy comparison.')
write_json(OUT / 'compiled_pilot' / f'R{size}.json', r)
print(r, flush=True)
