"""Audit and geometry for a fixed x0.5 resize followed by a center crop.
Writes only within this new experiment. No previous experiment is modified.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse, collections, hashlib, json, os, platform, sys, threading, time
from pathlib import Path
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'data/resnet_resolution_20261008'
sys.path.insert(0, str(ROOT))
CONFIG = json.loads(_resolve_model_artifact(HERE / 'config.json').read_text(encoding='utf-8-sig'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(CONFIG['tensorflow_threads']))
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import cv2, numpy as np, pandas as pd, psutil
from models.common.studies.final_tests_feature_cache_20260826.build_cache import sampled_block_indices

def digest(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(path)).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

def probe(path):
    cap = cv2.VideoCapture(str(path))
    r = dict(opened=cap.isOpened(), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=float(cap.get(cv2.CAP_PROP_FPS)), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return r

def halfscale(frame):
    result = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    if min(result.shape[:2]) < max(CONFIG['resolutions']):
        raise ValueError(f'Fixed x0.5 image {result.shape[:2]} cannot support both crops; no padding or enlargement allowed')
    return result

def center_crop(half, size):
    h, w = half.shape[:2]
    if min(h, w) < size:
        raise ValueError('Crop does not fit the half-scale frame')
    x, y = ((w - size) // 2, (h - size) // 2)
    return half[y:y + size, x:x + size].copy()

def sample_indices(info, domain, row=None):
    if domain == 'CV':
        return sampled_block_indices(info['fps'], info['frames'], float(row.start_time), float(row.end_time))[0][:200]
    n = max(1, int(np.floor(info['frames'] / info['fps'] * 25.0)))
    ts = np.arange(n, dtype=np.float32) / 25.0
    return np.clip(np.rint(ts * float(info['fps'])).astype(np.int32), 0, info['frames'] - 1)[:200]

def native_frames(path, indices, domain):
    cap = cv2.VideoCapture(str(path))
    wanted = set(map(int, indices))
    first, last = (min(wanted), max(wanted))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    decoded = {}
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            decoded[idx] = halfscale(frame)
    cap.release()
    if wanted - set(decoded):
        raise RuntimeError(f'Missing sampled frames in {path}')
    return [decoded[int(i)] for i in indices]

class Peak:

    def __init__(self):
        self.process = psutil.Process()
        self.value = 0
        self.done = threading.Event()

    def __enter__(self):

        def poll():
            while not self.done.is_set():
                self.value = max(self.value, self.process.memory_info().rss)
                self.done.wait(0.02)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.value = max(self.value, self.process.memory_info().rss)

def preservation_inventory():
    records = []
    for p in sorted(_model_artifact_glob(PREVIOUS, '*', recursive=True)):
        if not _resolve_model_artifact(p).is_file():
            continue
        st = p.stat()
        r = dict(path=str(p.relative_to(PREVIOUS)), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if p.suffix in ('.py', '.json', '.txt') or p.name in ('test_predictions.csv', 'validation_probabilities.csv'):
            r['sha256'] = digest(p)
        records.append(r)
    return records

def audit():
    out = HERE / 'outputs'
    out.mkdir(exist_ok=True)
    if not _resolve_model_artifact(out / 'previous_experiment_inventory.json').exists():
        write_json(out / 'previous_experiment_inventory.json', preservation_inventory())
    m = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'outputs/examples.csv'))
    initial = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'outputs/initial_split.csv'))
    old_audit = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'outputs/source_audit.csv')).set_index(['domain', 'id'])
    expected_masks = _load_model_result_array(PREVIOUS / 'outputs/cache/R224/masks.npy', mmap_mode='r')
    records = []
    probes = {}
    bad = []
    t = time.perf_counter()
    for j, path in enumerate(m.source_video.unique()):
        p = Path(path)
        info = probe(p)
        issues = []
        if not info['opened'] or info['fps'] <= 0 or info['frames'] <= 0:
            issues.append('unreadable_or_invalid_time')
        cap = cv2.VideoCapture(str(p))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            issues.append('first_frame_decode_failed')
            half_shape = None
        else:
            actual = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            half_shape = list(actual.shape[:2])
            if min(actual.shape[:2]) < 320:
                issues.append('halfscale_crop320_does_not_fit')
            if list(frame.shape[:2]) != [info['height'], info['width']]:
                issues.append('header_frame_dimensions_differ')
        stat = p.stat() if _resolve_model_artifact(p).exists() else None
        part = m[m.source_video == path]
        if stat and ((part.source_bytes != stat.st_size).any() or (part.source_mtime_ns != stat.st_mtime_ns).any()):
            issues.append('source_changed_since_previous_manifest')
        record = dict(source=path, domain=str(part.iloc[0].domain), examples=len(part), half_shape=half_shape, status=';'.join(issues) or 'ok', source_bytes=stat.st_size if stat else None, source_mtime_ns=stat.st_mtime_ns if stat else None, **info)
        probes[path] = record
        records.append(record)
        if issues:
            bad.append(record)
        if j % 250 == 0:
            print(f'AUDIT_SOURCES {j}/{m.source_video.nunique()} failures={len(bad)}', flush=True)
    sample_records = []
    total = 0
    temporal_failures = []
    for r in m.itertuples():
        info = probes[r.source_video]
        if info['fps'] <= 0 or info['frames'] <= 0:
            continue
        timing = type('Times', (), dict(start_time=r.start_s, end_time=r.end_s))()
        idx = sample_indices(info, r.domain, timing)
        key = (r.domain, r.video_name if r.domain == 'CH' else r.block_id)
        old = old_audit.loc[key]
        issues = []
        if len(idx) != int(old.samples) or not np.array_equal(expected_masks[int(r.cache_index)], np.arange(200) < len(idx)):
            issues.append('temporal_mask_mismatch')
        if any((abs(float(info[k]) - float(old[k])) > 1e-06 for k in ('fps', 'frames'))):
            issues.append('previous_source_time_mismatch')
        if issues:
            temporal_failures.append(dict(block_id=r.block_id, issues=issues))
        sample_records.append(dict(block_id=r.block_id, domain=r.domain, samples=len(idx), first_index=int(idx[0]), last_index=int(idx[-1]), indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(), status=';'.join(issues) or 'ok'))
        total += len(idx)
    m['source_bytes'] = [probes[p]['source_bytes'] for p in m.source_video]
    m['source_mtime_ns'] = [probes[p]['source_mtime_ns'] for p in m.source_video]
    m.to_csv(out / 'examples.csv', index=False)
    initial.to_csv(out / 'initial_split.csv', index=False)
    pd.DataFrame(records).to_csv(out / 'source_audit.csv', index=False)
    pd.DataFrame(sample_records).to_csv(out / 'temporal_audit.csv', index=False)
    split_boundary = int(len(initial) * 0.8)
    summary = dict(status='ready' if not bad and (not temporal_failures) else 'blocked', sources=len(records), examples=len(m), historical=int((m.domain == 'CH').sum()), final=int((m.domain == 'CV').sum()), source_samples=total, source_audit_seconds=time.perf_counter() - t, source_failures=bad, temporal_failures=temporal_failures, dimensions=[dict(width=w, height=h, sources=n) for (w, h), n in collections.Counter(((r['width'], r['height']) for r in records)).items()], initial_rows=len(initial), initial_unique_names=int(initial.video_name.nunique()), initial_train_validation_shared_names=len(set(initial.iloc[:split_boundary].video_name) & set(initial.iloc[split_boundary:].video_name)), no_exclusions=True, no_padding=True, no_enlargement=True, scale=0.5, resolutions=CONFIG['resolutions'], reference_reuse='No previous feature cache, initial checkpoint, trained head or test score is reused. Both conditions regenerated with identical full-frame halfscale then crop.', known_limitations=['65 names shared between initial training and validation after balancing.', 'D2 CH/CV source independence is not established; possible F17,F3,F10 shared sources.', 'Six reused matches only; seeds are paired repetitions, not independent matches.'])
    write_json(out / 'source_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if bad or temporal_failures:
        raise RuntimeError('Source or temporal audit blocked; inspect outputs/source_audit.json before defining any fallback protocol')

def visual():
    out = HERE / 'outputs/geometry'
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(_resolve_model_artifact(HERE / 'outputs/examples.csv'))
    rows = []
    chosen = []
    for group in list(m[m.domain == 'CH'].source_group.unique())[:3]:
        chosen.append(m[(m.domain == 'CH') & (m.source_group == group)].iloc[0])
    for group in sorted(m[m.domain == 'CV'].group_id.unique()):
        part = m[(m.domain == 'CV') & (m.group_id == group)]
        chosen.append(part.iloc[len(part) // 2])
    sheets = []
    for j, row in enumerate(chosen):
        cap = cv2.VideoCapture(row.source_video)
        info = probe(row.source_video)
        timing = type('Times', (), dict(start_time=row.start_s, end_time=row.end_s))()
        indices = sample_indices(info, row.domain, timing)
        index = int(indices[len(indices) // 2])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError('Geometry example failed to decode')
        half = halfscale(frame)
        a, b = (center_crop(half, 224), center_crop(half, 320))
        assert np.array_equal(a, center_crop(b, 224))
        h, w = frame.shape[:2]
        sw, sh = (round(w * 0.5), round(h * 0.5))
        x, y = ((sw - 224) // 2, (sh - 224) // 2)
        x0, y0 = (round(x * w / sw), round(y * h / sh))
        x1, y1 = (round((x + 224) * w / sw), round((y + 224) * h / sh))
        old224 = cv2.resize(frame[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_LINEAR)
        old320 = cv2.resize(frame[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_LINEAR)
        canvas = half.copy()
        for size, color in [(320, (30, 210, 255)), (224, (100, 240, 60))]:
            x, y = ((canvas.shape[1] - size) // 2, (canvas.shape[0] - size) // 2)
            cv2.rectangle(canvas, (x, y), (x + size - 1, y + size - 1), color, 2)
        cv2.putText(canvas, 'x0.5: green224 yellow320', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.zeros((400, 1280, 3), np.uint8)
        panel[35:395, :640] = canvas
        panel[35:355, 660:980] = b
        panel[35:259, 1000:1224] = a
        title = f'{row.domain} {row.source_group} | source {w}x{h} | frame {index}'
        cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, 'crop320 (1:1 pixels)', (660, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, 'crop224', (1000, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        stem = f'{j:02d}_{row.domain}_{row.source_group}'
        cv2.imwrite(str(out / (stem + '.png')), panel)
        cv2.imwrite(str(out / (stem + '_crop224.png')), a)
        cv2.imwrite(str(out / (stem + '_crop320.png')), b)
        rows.append(dict(source=row.source_video, block_id=row.block_id, frame=index, source_shape=list(frame.shape), half_shape=list(half.shape), crop224=list(a.shape), crop320=list(b.shape), central224_bit_identical=True, old224_pixels_equal=bool(np.array_equal(a, old224)), new320_equals_old_fixed_FOV320=bool(np.array_equal(b, old320)), figure=stem + '.png'))
        sheets.append(cv2.resize(panel, (960, 300)))
    cv2.imwrite(str(out / 'field_of_view_comparison.png'), np.concatenate(sheets, axis=0))
    write_json(out / 'geometry_validation.json', dict(status='passed', examples=rows, interpretation='Full resize x0.5, then center crop.224 native coverage448x448;320 coverage640x640.320 covers 2.0408 times the area with identical object pixel scale.'))
    print('GEOMETRY_PASSED', len(rows), 'examples', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'visual'])
    args = parser.parse_args()
    audit() if args.command == 'audit' else visual()
