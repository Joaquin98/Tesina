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
import os, sys, json, time, hashlib
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1').items():
    os.environ[k] = v
CONFIGS = [('original', 1, 1), ('x2', 2, 2), ('old_line4', 4, 2), ('old_line8', 8, 2), ('line2', 2, 1), ('line4', 4, 1), ('line8', 8, 1)]
START = time.perf_counter()

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8-sig'))

def save(p, d):
    p = Path(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def stat(p):
    s = Path(p).stat()
    return [s.st_size, s.st_mtime_ns]

def progress(stage, **kw):
    d = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), elapsed_s=time.perf_counter() - START, **kw)
    save(HERE / 'progress.json', d)
    print(json.dumps(d), flush=True)

def initialize():
    import tensorflow as tf, psutil
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    return tf

def model_path(key):
    return ROOT / 'data/m0_cost2_professional_human_20261027/models/m0_cost2.keras' if key == 'x2' else HERE / f'models/{key}/m0.keras'

def m1_paths(key):
    if key == 'original':
        folder = ROOT / 'data/m1_ensemble_vs_all_development_20261012/models'
    elif key == 'x2':
        folder = ROOT / 'data/m1_recall_cost_20261027/models/cost_2'
    elif key.startswith('old_line'):
        folder = ROOT / f'data/m1_line_costs_20261028/models/line{key[8:]}_scrum2'
    else:
        folder = ROOT / f'data/m1_line_only_costs_20261028/models/{key}_scrum1'
    return (folder / 'dual.keras', folder / 'geometry_rf.joblib')
JOBS = [dict(date='20250816', duration=6166.44, n=154161, kickoff=188.2, rgb=ROOT / 'data/professional_rugby_m1_game1_20261024/results/rgb_full.npy', baseline='20261027_argentina_allblacks_20250816_m1_cost2_stride1', reference=ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json', kind='human'), dict(date='20250823', duration=5680.44, n=142011, kickoff=184.2, rgb=ROOT / 'data/professional_rugby_m1_single_20261022/results/rgb_full.npy', baseline='20261027_argentina_allblacks_20250823_m1_cost2_stride1', reference=ROOT / 'data/sampling_audit_20261026/reconciliation/2025-08-23_reference_v3.json', kind='partial automated diagnostic reference'), dict(date='20231020', duration=5776.88, n=144422, kickoff=6.0, rgb=ROOT / 'data/m1_cost2_rwc2023_full_20261028/rgb_full.npy', baseline='20261028_argentina_allblacks_20231020_m1_cost2_stride1', reference=ROOT / 'data/sampling_audit_20261026/reconciliation/2023-10-20_reference_v3.json', kind='partial automated diagnostic reference')]
