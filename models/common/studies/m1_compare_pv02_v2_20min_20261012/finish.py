"""Finish measurements, generate figure/report and replace the wrong slide."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import subprocess, time, json, os, sys
from datetime import datetime, timezone
B = experiment_directory(__file__)
ROOT = B.parents[1]
P = ROOT / 'docs/presentacion_guion_20260908/.build/v9_yolo_ablation_v2_20min'
NODE = Path('node')
PYTHON = Path(sys.executable)
PRESENTATION_TOOLS = Path(os.environ.get('PRESENTATION_TOOLS', ROOT / 'tools/presentations'))
# Optional presentation dependencies are supplied by the local Node installation.

def status(stage, **extra):
    (B / 'delivery_state.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **extra}, indent=2), encoding='utf-8')

def run(args, cwd=B):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)
status('waiting_for_timed_inferences')
while True:
    try:
        state = json.loads(_resolve_model_artifact(B / 'queue_state.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    if state.get('status') == 'failed':
        status('inference_failed', details=state)
        raise SystemExit(1)
    if state.get('status') == 'complete':
        break
    time.sleep(20)
try:
    status('building_comparison_and_report')
    run([sys.executable, '-B', _resolve_model_artifact(B / 'make_comparison.py')])
    run([sys.executable, '-B', _resolve_model_artifact(B / 'report.py')])
    run([PYTHON, _resolve_model_artifact(B / 'prepare_slides.py')])
    status('authoring_presentation')
    run([NODE, PRESENTATION_TOOLS / 'container_tools/mark_artifact_operation_started.mjs', '--operation-kind', 'edit', '--expected-output-count', '1', '--output-format', 'pptx'], PRESENTATION_TOOLS)
    run([NODE, P / 'author.mjs'], P)
    run([PYTHON, _resolve_model_artifact(P / 'merge.py')])
    run([NODE, P / 'finalize.mjs'])
    run([PYTHON, _resolve_model_artifact(P / 'verify.py')])
    status('complete', presentation=str(ROOT / 'docs/presentacion_guion_20260908/output/presentacion_defensa_desde_guion_v9_en_desarrollo_r39.pptx'), image=str(B / 'figures/comparacion_real_inferencias_v2_pv02_20min.png'), report=str(B / 'README.md'))
except Exception as error:
    status('delivery_failed', error=str(error))
    raise
