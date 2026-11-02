import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, os, json, hashlib, time
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1', KERAS_HOME=str(HERE / 'runtime/keras'), YOLO_CONFIG_DIR=str(HERE / 'runtime/ultralytics')).items():
    os.environ[k] = v
CSV = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
ARRAYS = ROOT / 'data/m1_recall_cost_20261027/prepared_inputs'
BASE = ROOT / 'models/M0/checkpoints/best_model/best_model.keras'
PRO = ROOT / 'data/m0_m1_cost_sampling_20261029'
CONFIGS = [('m0_control', 'm0', 1, 1), ('m0_line4_scrum2', 'm0', 4, 2), ('m1_control', 'm1', 1, 1), ('m1_x2', 'm1', 2, 2)]
FOLDS = [f'PV{i:02d}' for i in range(1, 7)]

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8-sig'))

def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def stat(p):
    s = Path(p).stat()
    return [s.st_size, s.st_mtime_ns]

def progress(family, **values):
    d = dict(pid=os.getpid(), updated_unix=time.time(), family=family, **values)
    save(HERE / f'progress_{family}.json', d)
    print(json.dumps(d), flush=True)
