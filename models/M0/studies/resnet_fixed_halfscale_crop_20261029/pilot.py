"""Small paired source feature/latency validation, not a quality evaluation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys, time, json, platform
from pathlib import Path
import numpy as np, psutil
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment import CONFIG, HERE, ROOT, OUT, Peak, extractor, cv2_resize_rgb, write_json, manifest
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.preflight import probe, sample_indices, native_frames
size = int(sys.argv[1])
m = manifest()
examples = []
for domain in ('CH', 'CV'):
    row = m[m.domain == domain].iloc[0]
    info = probe(row.source_video)
    timing = type('Times', (), dict(start_time=row.start_s, end_time=row.end_s))()
    indices = sample_indices(info, domain, timing)[:8]
    examples.extend(native_frames(Path(row.source_video), indices, domain))
x = np.asarray([cv2_resize_rgb(frame, size) for frame in examples], np.float32)
assert x.shape == (16, size, size, 3)
with Peak() as peak:
    model, infer = extractor(size)
    times = []
    values = []
    for first in range(0, len(x), CONFIG['feature_batch_size']):
        tick = time.perf_counter()
        values.append(infer(x[first:first + CONFIG['feature_batch_size']]).numpy())
        times.append(time.perf_counter() - tick)
    features = np.concatenate(values)
assert features.shape == (16, 2048) and np.isfinite(features).all()
folder = OUT / 'smoke'
folder.mkdir(exist_ok=True)
np.savez_compressed(folder / f'features_{size}.npz', features=features, condition=size)
import tensorflow as tf, keras, cv2
result = dict(status='passed', size=size, frames=16, feature_shape=list(features.shape), inference_s=sum(times), s_per_frame=sum(times) / 16, peak_rss_bytes=peak.value, hardware=dict(platform=platform.platform(), cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total), runtime=dict(python=sys.version, tensorflow=tf.__version__, keras=keras.__version__, opencv=cv2.__version__, physical_devices=[str(d) for d in tf.config.list_physical_devices()]), tensorflow_threads=CONFIG['tensorflow_threads'], note='Separate process per crop. Full original x0.5 then crop. Equal batch8 warmup excluded from inference timer. Small engineering benchmark, not F1.')
write_json(OUT / 'pilot' / f'R{size}.json', result)
print(json.dumps(result, indent=2), flush=True)
