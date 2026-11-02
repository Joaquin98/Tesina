"""File-backed exact training tensors, produced solely from immutable caches."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import hashlib, json, time
import numpy as np

def prepare(frame, data, folder, progress):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    n = len(frame)
    identity = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    manifest = folder / 'manifest.json'
    if _resolve_model_artifact(manifest).exists():
        info = json.loads(_resolve_model_artifact(manifest).read_text())
        assert info['frame_sha256'] == identity and info['complete']
    else:
        specs = {'rgb': ((n, 200, 2048), np.float32), 'pose': ((n, 200, 62), np.float32), 'mask': ((n, 200), bool), 'labels': ((n,), np.int64)}
        arrays = {}
        for key, (shape, dtype) in specs.items():
            path = folder / (key + '.npy')
            arrays[key] = np.lib.format.open_memmap(path, mode='r+' if _resolve_model_artifact(path).exists() else 'w+', dtype=dtype, shape=shape)
            assert arrays[key].shape == shape and arrays[key].dtype == np.dtype(dtype)
        donepath = folder / 'done.npy'
        done = np.lib.format.open_memmap(donepath, mode='r+' if _resolve_model_artifact(donepath).exists() else 'w+', dtype=bool, shape=(n,))
        labels = data.labels_for(frame)
        start = time.perf_counter()
        last = 0
        for i, row in enumerate(frame.itertuples(index=False)):
            if done[i]:
                continue
            rgb, mask, length = data._load_resnet_path.__wrapped__(str(row.resnet_path), str(row.block_id))
            pose = data.geometry_sequence(data.load_yolo(row))
            arrays['rgb'][i] = rgb
            arrays['pose'][i] = pose
            arrays['mask'][i] = mask
            arrays['labels'][i] = labels[i]
            done[i] = True
            if time.perf_counter() - last >= 30 or i == n - 1:
                for a in arrays.values():
                    a.flush()
                done.flush()
                progress('preparing_file_backed_training_inputs', blocks=i + 1, total=n, preparation_s=time.perf_counter() - start)
                last = time.perf_counter()
        assert done.all()
        for a in arrays.values():
            a.flush()
        done.flush()
        info = dict(complete=True, frame_sha256=identity, blocks=n, seconds=time.perf_counter() - start, shapes={k: list(a.shape) for k, a in arrays.items()}, values='float32 exact outputs of original load_resnet and geometry_sequence; no quantization')
        manifest.write_text(json.dumps(info, indent=2))
        del arrays, done
    result = {k: _load_model_result_array(folder / (k + '.npy'), mmap_mode='r') for k in ('rgb', 'pose', 'mask', 'labels')}
    for i in (0, 10, 4081, 4082, 5000, n - 1):
        row = frame.iloc[i]
        rgb, mask, _ = data._load_resnet_path.__wrapped__(str(row.resnet_path), str(row.block_id))
        pose = data.geometry_sequence(data.load_yolo(row))
        np.testing.assert_array_equal(result['rgb'][i], rgb)
        np.testing.assert_array_equal(result['mask'][i], mask)
        np.testing.assert_array_equal(result['pose'][i], pose)
    np.testing.assert_array_equal(result['labels'], data.labels_for(frame))
    (folder / 'equivalence.json').write_text(json.dumps(dict(passed=True, blocks=[0, 10, 4081, 4082, 5000, n - 1], both_domains=True, labels_all=True), indent=2))
    return result

def make_sequence(models, arrays):

    class FileSequence(models.ResNetCacheSequence):

        def __getitem__(self, batch_index):
            selected = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            return ({'rgb': arrays['rgb'][selected], 'pose': arrays['pose'][selected], 'sequence_mask': arrays['mask'][selected]}, arrays['labels'][selected])
    return FileSequence
