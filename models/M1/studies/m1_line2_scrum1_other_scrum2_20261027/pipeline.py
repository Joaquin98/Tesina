import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, subprocess, time, json, html, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/m1_line2_scrum1_other_scrum2_20261027'
WEB.mkdir(parents=True, exist_ok=True)
try:
    for mode in ('train', 'evaluate'):
        with _resolve_model_artifact(HERE / (mode + '.log')).open('a') as out, _resolve_model_artifact(HERE / (mode + '.err.log')).open('a') as err:
            job = subprocess.Popen([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'run.py')), mode], cwd=ROOT, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            while job.poll() is None:
                try:
                    status = json.loads(_resolve_model_artifact(HERE / 'progress.json').read_text(encoding='utf-8'))
                except (FileNotFoundError, json.JSONDecodeError):
                    status = {}
                (WEB / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><body style="font:18px system-ui;max-width:1000px;margin:40px auto"><h1>M1 Â· line Ã—2, scrum Ã—1, otroâ†’scrum Ã—2</h1><p>Prueba con cachÃ©: cinco Ã©pocas desde el mismo inicio del M1 original. Negativos de entrenamiento congelados, sin usar etiquetas del partido para entrenar.</p><pre style="white-space:pre-wrap">' + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + '</pre></body></html>', encoding='utf-8')
                time.sleep(20)
            if job.returncode:
                raise RuntimeError(mode + ' failed ' + str(job.returncode))
    subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'publish.py'))], cwd=ROOT, check=True)
    (HERE / 'pipeline_complete.json').write_text(json.dumps(dict(status='complete', finished_unix=time.time())), encoding='utf-8')
except Exception:
    (HERE / 'pipeline_error.json').write_text(json.dumps(dict(traceback=traceback.format_exc())), encoding='utf-8')
    raise
