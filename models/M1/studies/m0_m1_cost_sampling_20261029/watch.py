"""Lightweight live status; never rewrites metrics or training artifacts."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
WEB = ROOT / 'src/web/static/reports/m0_m1_cost_sampling_20261029'

def main():
    WEB.mkdir(parents=True, exist_ok=True)
    while True:
        complete = _resolve_model_artifact(HERE / 'pipeline_complete.json').exists()
        d = dict(updated_unix=time.time(), complete=complete, training=read(HERE / 'progress.json') if _resolve_model_artifact(HERE / 'progress.json').exists() else {}, evaluation=read(HERE / 'evaluation_progress.json') if _resolve_model_artifact(HERE / 'evaluation_progress.json').exists() else {}, completed_models=len(list(_model_artifact_glob(HERE / 'models', '*/complete.json', recursive=False))), completed_configurations=len(list(_model_artifact_glob(HERE / 'evaluation', '*/*/metrics.json', recursive=False))), errors={p.name: read(p) for p in _model_artifact_glob(HERE, '*_error.json', recursive=False)})
        save(WEB / 'live_progress.json', d)
        if complete or d['errors']:
            return
        time.sleep(20)
if __name__ == '__main__':
    main()
