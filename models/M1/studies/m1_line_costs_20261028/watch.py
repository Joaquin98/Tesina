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
import sys, time, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(source_directory(HERE)))
from models.M1.studies.m1_line_costs_20261028.run import read, save
WEB = ROOT / 'src/web/static/reports/m1_line_costs_20261028'

def main():
    WEB.mkdir(parents=True, exist_ok=True)
    if not _resolve_model_artifact(HERE / 'results.json').exists():
        (WEB / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 Â· entrenamiento con mayor peso de line</title><style>body{font:17px/1.5 system-ui;max-width:950px;margin:35px auto;padding:0 18px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126b84}</style><h1>M1: mayor peso de line</h1><p>En ejecuciÃ³n: line4/otro1/scrum2 y line8/otro1/scrum2. Se compararÃ¡n con original y M1Ã—2 sobre los tres partidos completos. Entrenamiento amateur, misma inicializaciÃ³n y5Ã©pocas; sin extracciÃ³n nueva ni incorporaciÃ³n de los partidos al entrenamiento.</p><p>16/08 se evaluarÃ¡ con las anotaciones manuales;23/08 y Mundial con inventarios diagnósticos automáticos parciales. Se mostrarÃ¡n sensibilidad y precisiÃ³n por clase, jugadas omitidas y minutos de revisiÃ³n. El informe final reemplazarÃ¡ esta pÃ¡gina automÃ¡ticamente.</p><pre id="status">Cargandoâ€¦</pre><script>async function update(){try{const s=await(await fetch(\'progress.json?t=\'+Date.now())).json();if(s.status===\'complete\'){location.reload();return}document.getElementById(\'status\').textContent=JSON.stringify(s,null,2)}catch(e){}}update();setInterval(update,15000)</script></html>', encoding='utf-8')
    while True:
        errors = [dict(file=p.name, data=read(p)) for p in _model_artifact_glob(HERE, '*error*.json', recursive=False)]
        done = _resolve_model_artifact(HERE / 'pipeline_complete.json').exists()
        progress = read(HERE / 'progress.json') if _resolve_model_artifact(HERE / 'progress.json').exists() else {}
        phase = read(HERE / 'pipeline_progress.json') if _resolve_model_artifact(HERE / 'pipeline_progress.json').exists() else {}
        save(WEB / 'progress.json', dict(status='complete' if done else 'error' if errors else 'running', phase=phase.get('stage'), updated_unix=time.time(), progress=progress, errors=errors))
        if done or errors:
            break
        time.sleep(20)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        (HERE / 'watch.err.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise
