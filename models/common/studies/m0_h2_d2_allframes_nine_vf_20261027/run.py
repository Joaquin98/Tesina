"""Six frozen thesis M0 folds; actual 25fps frames; nine crops; resumable by block."""
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
import sys, os, json, time, hashlib, msvcrt, argparse
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2, numpy as np, pandas as pd, keras, openvino as ov
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
H = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(source_directory(ROOT / 'data/multicrop_d2_stride25_vf_20261026')))
from common import views
from models.common.spatial import normalize_training_frame
from models.common.comparison.metrics import events_from_blocks, match_events
from sklearn.metrics import precision_recall_fscore_support
LABELS = ['line', 'other', 'scrum']
SOURCE = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
XML = ROOT / 'data/yolo_crop_classification_20261023/runtime/resnet50.xml'
BIN = XML.with_suffix('.bin')
MODELS = [ROOT / f'data/final_model_comparison_20260827/outputs/runs/PV-M0-H2-D2-PV{i:02d}-S42/attempt_001/model.keras' for i in range(1, 7)]

def sha(p):
    h = hashlib.sha256()
    with open(_resolve_model_artifact(p), 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, p)

def lock():
    f = _resolve_model_artifact(H / 'run.lock').open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    return f

def settings():
    rows = [r for r in json.loads(_resolve_model_artifact(SOURCE).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test']
    assert len(rows) == 688
    files = [SOURCE, XML, BIN, Path(__file__).resolve(), _resolve_model_artifact(ROOT / 'data/multicrop_d2_stride25_vf_20261026/common.py'), _resolve_model_artifact(ROOT / 'models/common/spatial.py')] + MODELS
    paths = {str(p): sha(p) for p in files}
    cfg = dict(samples=688, models=[str(p) for p in MODELS], model_sha256={str(p): paths[str(p)] for p in MODELS}, hashes=paths, views=9, sequence_length=200, real_frames='all 25 FPS positions, up to 175 per seven second block; masks exclude padding', decision='central argmax; only other to line when two contexts vote line and >=1 p_line>=0.90; central scrum protected', source='VF previously used; exploratory', folds='six distinct checkpoints seed42, not six seeds or independent tests')
    return (rows, cfg)

def setup():
    rows, cfg = settings()
    p = H / 'protocol.json'
    if _resolve_model_artifact(p).exists():
        if json.loads(_resolve_model_artifact(p).read_text(encoding='utf8')) != cfg:
            raise RuntimeError('frozen protocol or hashes changed')
    else:
        save(p, cfg)
    return rows

def resources():
    encoder = ov.Core().compile_model(str(XML), 'CPU', {'INFERENCE_NUM_THREADS': 6, 'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32'})
    models = [keras.models.load_model(_resolve_model_artifact(p), compile=False) for p in MODELS]
    for m in models:
        assert [(tuple(x.shape[1:]), str(x.dtype)) for x in m.inputs] == [((200, 2048), 'float32'), ((200,), 'bool')]
    return (encoder, models)

def extract(row, encoder):
    cap = cv2.VideoCapture(row['source'])
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(row['start_s'] * fps)
    stop = min(total, max(start + 1, int(row['end_s'] * fps)))
    length = min(200, int(np.floor((row['end_s'] - row['start_s']) * 25)))
    assert length > 0 and length == row['frames']
    indices = np.clip(np.rint(start + np.arange(length) * fps / 25).astype(int), start, stop - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    decoded = int(indices[0]) - 1
    last = None
    batch = []
    encoded = []
    first = None
    for i in indices:
        while decoded < i:
            if not cap.grab():
                raise RuntimeError('incomplete source')
            decoded += 1
            ok, last = cap.retrieve()
            assert ok
        crops, _ = views(last)
        central = normalize_training_frame(last)
        assert np.array_equal(crops[0], central)
        if first is None:
            first = crops[0].copy()
        batch.append(crops)
        if len(batch) == 8 or i == indices[-1]:
            images = np.concatenate(batch)
            out = []
            for off in range(0, len(images), 16):
                out.append(encoder(np.ascontiguousarray(images[off:off + 16, :, :, ::-1], dtype=np.float32))[0].copy())
            encoded.append(np.concatenate(out).reshape(len(batch), 9, 2048))
            batch = []
    cap.release()
    encoded = np.concatenate(encoded)
    assert encoded.shape == (length, 9, 2048) and np.isfinite(encoded).all()
    return (encoded, first, indices)

def infer(z, models):
    features = np.zeros((9, 200, 2048), np.float32)
    features[:, :len(z)] = z.transpose(1, 0, 2)
    mask = np.broadcast_to(np.arange(200) < len(z), (9, 200))
    probs = np.stack([m([features, mask], training=False).numpy() for m in models])
    assert probs.shape == (6, 9, 3) and np.isfinite(probs).all() and np.allclose(probs.sum(2), 1, atol=1e-05)
    return probs

def pilot(rows, encoder, models):
    z, first, indices = extract(rows[0], encoder)
    p = infer(z, models)
    full = keras.models.load_model(_resolve_model_artifact(ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras'), compile=False)
    ref = full(np.asarray(first[None, :, :, ::-1], dtype=np.float32), training=False).numpy()
    np.testing.assert_allclose(ref[0], z[0, 0], atol=0.0001, rtol=0.0001)
    save(H / 'pilot.json', dict(passed=True, clip_id=rows[0]['clip_id'], valid_frames=len(indices), center_exact=True, keras_openvino_max_diff=float(abs(ref[0] - z[0, 0]).max()), model_shapes=list(p.shape), probabilities_finite=True))
    print('PILOT OK', flush=True)

def report(rows):
    probs = np.stack([_load_model_result_array(H / 'blocks' / f'{i:04}.npy') for i in range(len(rows))])
    assert probs.shape == (688, 6, 9, 3)
    frame = pd.DataFrame(rows)
    y = np.array([LABELS.index(t) for t in frame.tag])
    out = []
    event = []
    allpred = []
    for fi in range(6):
        p = probs[:, fi]
        base = p[:, 0].argmax(1)
        votes = p[:, 1:].argmax(2) == 0
        high = votes & (p[:, 1:, 0] >= 0.9)
        combined = np.where((base == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1), 0, base)
        assert np.array_equal(base == 2, combined == 2)
        d = frame[['clip_id', 'section', 'start_s', 'end_s', 'tag']].copy()
        d['fold'] = fi + 1
        for name, pred in [('central', base), ('nine', combined)]:
            d[name] = pred
            pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
            entry = {'fold': fi + 1, 'model': name}
            for i, l in [(0, 'line'), (2, 'scrum')]:
                entry.update({l + '_precision': pr[i], l + '_recall': re[i], l + '_f1': f[i], l + '_tp': int(sum((y == i) & (pred == i))), l + '_fp': int(sum((y != i) & (pred == i)))})
            out.append(entry)
            f = frame.copy()
            f['y_true'] = f.tag
            f['y_final'] = [LABELS[i] for i in pred]
            f['p_line'] = p[:, 0, 0]
            f['p_scrum'] = p[:, 0, 2]
            if name == 'nine':
                strength = np.sort(np.where(votes, p[:, 1:, 0], 0), axis=1)[:, -2]
                f.loc[pred != base, 'p_line'] = strength[pred != base]
            for l in ['line', 'scrum']:
                t = events_from_blocks(f, l, prediction_column='y_true', probability_column='p_' + l)
                a = events_from_blocks(f, l, prediction_column='y_final', probability_column='p_' + l)
                n = len(match_events(t, a))
                event.append(dict(fold=fi + 1, model=name, label=l, real=len(t), predicted=len(a), matched=n, unmatched=len(a) - n, precision=n / len(a) if a else 0, recall=n / len(t) if t else 0))
        allpred.append(d)
    pd.DataFrame(out).to_csv(H / 'results.csv', index=False)
    pd.DataFrame(event).to_csv(H / 'events.csv', index=False)
    pd.concat(allpred).to_csv(H / 'predictions.csv', index=False)
    save(H / 'completion.json', {'complete': True, 'blocks': 688, 'fold_models': 6, 'views': 9, 'mean_is_not_ensemble': True, 'vf_reused': True})
    save(H / 'status.json', {'phase': 'complete', 'blocks': 688})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    a = parser.parse_args()
    with lock():
        rows = setup()
        cv2.setNumThreads(1)
        encoder, models = resources()
        if a.pilot:
            pilot(rows, encoder, models)
            return
        if not json.loads(_resolve_model_artifact(H / 'pilot.json').read_text())['passed']:
            raise RuntimeError('pilot not passed')
        folder = H / 'blocks'
        folder.mkdir(exist_ok=True)
        verified = set()
        started = time.time()
        count = 0
        for i, row in enumerate(rows):
            target = folder / f'{i:04}.npy'
            if _resolve_model_artifact(target).exists():
                p = _load_model_result_array(target)
                assert p.shape == (6, 9, 3) and np.isfinite(p).all() and np.allclose(p.sum(2), 1, atol=1e-05)
                count += 1
                continue
            if row['source'] not in verified:
                if sha(Path(row['source'])) != row['source_sha256']:
                    raise RuntimeError('VF source changed')
                verified.add(row['source'])
            t = time.time()
            z, _, _ = extract(row, encoder)
            p = infer(z, models)
            tmp = target.with_suffix('.tmp')
            with _resolve_model_artifact(tmp).open('wb') as f:
                np.save(f, p)
            os.replace(tmp, target)
            count += 1
            save(H / 'status.json', {'phase': 'inference', 'complete': count, 'total': 688, 'last_clip': row['clip_id'], 'seconds_last': time.time() - t, 'started_unix': started})
            print(f"{count}/688 {row['clip_id']} seconds={time.time() - t:.1f}", flush=True)
        report(rows)
if __name__ == '__main__':
    main()
