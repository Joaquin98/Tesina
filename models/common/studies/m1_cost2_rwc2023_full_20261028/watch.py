"""Expose measured progress in the local tool until the final report replaces it."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, time, json
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(source_directory(HERE)))
from extract import read, save, N
WEB = ROOT / 'src/web/static/reports/m1_cost2_rwc2023_full_20261028'

def main():
    WEB.mkdir(parents=True, exist_ok=True)
    if not _resolve_model_artifact(HERE / 'results.json').exists():
        (WEB / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 ×2 completo · ejecución</title><style>body{font:17px/1.5 system-ui;max-width:900px;margin:40px auto;padding:0 18px}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#126b84}</style><h1>M1 ×2 completo · Mundial 2023</h1><p>Extracción de 138.644 fotogramas faltantes; se reutilizan 5.778 del caché existente. Después se ejecutarán M1 original y ×2 sobre el mismo caché completo y se publicará la comparación aquí. Sin entrenamiento nuevo.</p><p>El tiempo restante es una estimación provisional basada en la velocidad medida. Puede cambiar con la carga de la computadora.</p><pre id="status">Cargando progreso…</pre><p><a href="/compare?comparison=20261027_argentina_allblacks_20250823_original_vs_cost2">23/08/2025: prueba completa ya disponible</a></p><p>Las referencias de ambos partidos son inventarios diagnósticos automáticos parciales; no equivalen a las anotaciones manuales humanas del 16/08.</p><script>async function poll(){try{const r=await fetch(\'progress.json?t=\'+Date.now()),s=await r.json();if(s.status===\'complete\'){location.reload();return}document.getElementById(\'status\').textContent=JSON.stringify(s,null,2)}catch(e){document.getElementById(\'status\').textContent=\'Esperando actualización…\'}}poll();setInterval(poll,15000)</script></html>', encoding='utf-8')
    while True:
        states = [read(p) for p in _model_artifact_glob(HERE, 'progress_worker*.json', recursive=False)]
        elapsed = max([s['elapsed_s'] for s in states] + [0.0])
        new = sum((s.get('new_observations', 0) for s in states))
        rate = new / elapsed if elapsed else 0.0
        pipeline = read(HERE / 'pipeline_progress.json') if _resolve_model_artifact(HERE / 'pipeline_progress.json').exists() else {}
        errors = [read(p) for p in _model_artifact_glob(HERE, '*error*.json', recursive=False)]
        status = 'complete' if _resolve_model_artifact(HERE / 'pipeline_complete.json').exists() else 'error' if errors else 'running'
        v = dict(status=status, updated_unix=time.time(), stage=pipeline.get('stage'), completed_chunks=pipeline.get('completed_chunks'), total_chunks=71, new_observations=new, total_new_required=138644, reused_anchors=5778, native_frames=N, measured_new_observations_per_second=round(rate, 2), estimated_extraction_remaining_minutes=round((138644 - new) / rate / 60, 1) if rate else None, errors=errors)
        save(WEB / 'progress.json', v)
        if status in ('complete', 'error'):
            break
        time.sleep(20)
if __name__ == '__main__':
    main()
