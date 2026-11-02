import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, time, sys, subprocess, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/m0_cost2_professional_human_20261027'
WEB.mkdir(parents=True, exist_ok=True)
try:
    while not _resolve_model_artifact(HERE / 'completion.json').exists():
        f = HERE / 'progress.json'
        s = json.loads(_resolve_model_artifact(f).read_text()) if _resolve_model_artifact(f).exists() else {}
        if s.get('stage') == 'error':
            raise RuntimeError(s.get('traceback'))
        (WEB / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><body style="font:18px system-ui;max-width:900px;margin:40px auto"><h1>M0 Ã—2 Â· prueba en curso</h1><p>Entrenamiento RGB desde cachÃ©, sin extracciÃ³n nueva.</p><pre>' + json.dumps(s, ensure_ascii=False, indent=2) + '</pre><p>Esta pÃ¡gina mostrarÃ¡ la comparaciÃ³n con M1 Ã—2 al terminar y verificarse.</p></body></html>', encoding='utf-8')
        time.sleep(30)
    subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'publish.py'))], cwd=ROOT, check=True)
    (HERE / 'pipeline_complete.json').write_text(json.dumps({'status': 'complete', 'finished': time.time()}), encoding='utf-8')
except Exception:
    (HERE / 'pipeline_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}), encoding='utf-8')
    raise
