import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import subprocess, sys, time, json, traceback, html
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/scrum_hard_negatives_20261027'
WEB.mkdir(parents=True, exist_ok=True)

def read(p):
    try:
        return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def show(stage):
    d = {f: read(HERE / f / 'progress.json') for f in ('m0', 'm1')}
    (WEB / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><body style="font:17px system-ui;max-width:1000px;margin:40px auto"><h1>Reducir falsas alertas scrum Â· M0 Ã—2 y M1 Ã—2</h1><p>Prueba exploratoria con cachÃ©. Una Ã©poca de ajuste, control y pesos Ã—1,5 / Ã—2 sobre ejemplos de entrenamiento Â«otro â†’ scrumÂ». Sin extracciÃ³n nueva.</p><p>Etapa: ' + html.escape(stage) + '</p><pre style="white-space:pre-wrap">' + html.escape(json.dumps(d, ensure_ascii=False, indent=2)) + '</pre></body></html>', encoding='utf-8')
try:
    for mode in ('train', 'evaluate'):
        jobs = []
        handles = []
        for family in ('m0', 'm1'):
            folder = HERE / family
            folder.mkdir(exist_ok=True)
            out = _resolve_model_artifact(folder / (mode + '.log')).open('a')
            err = _resolve_model_artifact(folder / (mode + '.err.log')).open('a')
            handles.extend([out, err])
            job = subprocess.Popen([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'run.py')), mode, family], cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            jobs.append((family, job))
        while any((job.poll() is None for _, job in jobs)):
            show(mode)
            time.sleep(20)
        for h in handles:
            h.close()
        failures = [(family, job.returncode) for family, job in jobs if job.returncode]
        if failures:
            raise RuntimeError(str(failures))
    show('publicando y verificando')
    subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'publish.py'))], cwd=ROOT, check=True)
    (HERE / 'pipeline_complete.json').write_text(json.dumps({'status': 'complete', 'finished_unix': time.time()}), encoding='utf-8')
except Exception:
    (HERE / 'pipeline_error.json').write_text(json.dumps({'traceback': traceback.format_exc()}), encoding='utf-8')
    raise
