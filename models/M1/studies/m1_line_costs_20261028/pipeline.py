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
import sys, subprocess, time, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(source_directory(HERE)))
from models.M1.studies.m1_line_costs_20261028.run import save

def main():
    start = time.perf_counter()
    for name in ('run.py', 'evaluate.py', 'publish.py'):
        save(HERE / 'pipeline_progress.json', dict(stage=name, elapsed_s=time.perf_counter() - start, updated_unix=time.time()))
        with _resolve_model_artifact(HERE / (name + '.log')).open('a', encoding='utf-8') as f:
            subprocess.run([sys.executable, '-B', str(HERE / name)], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    save(HERE / 'pipeline_complete.json', dict(status='complete', elapsed_s=time.perf_counter() - start))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'pipeline_error.json', dict(error=traceback.format_exc()))
        raise
