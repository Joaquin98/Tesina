import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, time, subprocess, sys, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/m2_professional_human_20261027'
WEB.mkdir(parents=True, exist_ok=True)
try:
    while True:
        states = []
        for w in (0, 1):
            p = HERE / f'extract_{w}_progress.json'
            states.append(json.loads(_resolve_model_artifact(p).read_text()) if _resolve_model_artifact(p).exists() else {})
            err = HERE / f'extract_{w}.err.log'
            if _resolve_model_artifact(err).exists() and 'Traceback (most recent call last)' in _resolve_model_artifact(err).read_text():
                raise RuntimeError(_resolve_model_artifact(err).read_text()[-3000:])
        if all((s.get('stage') == 'complete' for s in states)):
            break
        done = sum((s.get('observations_done', 0) for s in states))
        (WEB / 'index.html').write_text(f'<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>M2 en proceso</title><body style="font:18px system-ui;max-width:900px;margin:40px auto"><h1>M2 · extracción en curso</h1><p>YOLO: {done:,} / 88.092 observaciones guardadas. Dos procesos CPU, caché reanudable.</p><p>Se reutiliza ResNet. Al terminar se publicará aquí la comparación con M1 original y M1 ×2 usando las anotaciones manuales corregidas. Esta página se actualiza cada30segundos.</p></body></html>', encoding='utf-8')
        time.sleep(30)
    for name in ('evaluate.py', 'publish.py'):
        subprocess.run([sys.executable, '-B', '-u', str(HERE / name)], cwd=ROOT, check=True)
    (HERE / 'pipeline_complete.json').write_text(json.dumps({'status': 'complete', 'finished': time.time()}), encoding='utf-8')
except Exception:
    (HERE / 'pipeline_error.json').write_text(json.dumps({'status': 'error', 'traceback': traceback.format_exc()}), encoding='utf-8')
    raise
