"""Sequential, resumable executor for the complete 300-run Panel-V matrix."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .data import OUTPUTS
from .protocol import DATA_REGIMES, LEVELS, MODELS, PANEL_V_TRAINABLE_EXCLUSIONS
from .runner import REGISTRY, audit_panel_v, summarize
LOCK = OUTPUTS / 'panel_v_queue_state.json'
PROTOCOL_TEST = 'tests/test_final_model_comparison_protocol.py'

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_state(status: str, **values) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    payload = {'pid': os.getpid(), 'status': status, 'updated_at': utc_now(), **values}
    temporary = LOCK.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(LOCK)

def active_runs() -> list[str]:
    if not _resolve_model_artifact(REGISTRY).is_file():
        return []
    registry = pd.read_csv(_resolve_model_artifact(REGISTRY))
    if 'status' not in registry or 'run_id' not in registry:
        return []
    return registry.loc[registry.status == 'ejecutando', 'run_id'].astype(str).tolist()

def run(command: list[str]) -> None:
    print(f"[{utc_now()}] RUN {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=experiment_directory(__file__).parents[1], check=False)
    if completed.returncode:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')

def recipe_already_verified(recipe: str, audit: dict) -> bool:
    """Return true only when the audit proves all six folds of a recipe valid."""
    detail_path = Path(str(audit.get('detail', '')))
    if not _resolve_model_artifact(detail_path).is_file():
        return False
    parts = recipe.split('-')
    if len(parts) not in {2, 3}:
        return False
    model, level = parts[:2]
    detail = pd.read_csv(_resolve_model_artifact(detail_path))
    required = {'model', 'level', 'data_regime', 'fold', 'state'}
    if not required.issubset(detail.columns):
        return False
    selected = detail.loc[detail['model'].astype(str).eq(model) & detail['level'].astype(str).eq(level)].copy()
    if len(parts) == 3:
        selected = selected.loc[selected['data_regime'].fillna('').astype(str).eq(parts[2])]
    expected_folds = {f'PV{index:02d}' for index in range(1, 7)}
    verified_folds = set(selected.loc[selected['state'].astype(str).eq('verificada'), 'fold'].astype(str))
    return verified_folds == expected_folds

def recipe_commands() -> list[tuple[str, list[str]]]:
    python = sys.executable
    module = 'models.common.comparison.runner'
    recipes = []
    for model in MODELS:
        recipes.append((f'{model}-B', [python, '-m', module, 'run-matrix', '--panel', 'PV', '--model', model, '--level', 'B', '--resume']))
    for model in MODELS:
        for level in LEVELS:
            for regime in DATA_REGIMES:
                if model in PANEL_V_TRAINABLE_EXCLUSIONS:
                    continue
                recipes.append((f'{model}-{level}-{regime}', [python, '-m', module, 'run-matrix', '--panel', 'PV', '--model', model, '--level', level, '--regime', regime, '--resume', '--epochs', '40', '--patience', '7', '--batch-size', '16']))
    return recipes

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--cache-retry-seconds', type=int, default=300)
    args = parser.parse_args(argv)
    recipes = recipe_commands()
    cycle = 0
    try:
        while True:
            while active_runs():
                write_state('esperando_lote_activo', active_runs=active_runs(), cycle=cycle)
                time.sleep(max(10, args.poll_seconds))
            before = audit_panel_v()
            if before['all_required_verified']:
                summarize()
                write_state('completa', audit=before, cycle=cycle)
                print(json.dumps(before, ensure_ascii=False, indent=2), flush=True)
                return 0
            verified_before = int(before['states'].get('verificada', 0))
            current_audit = before
            for recipe, command in recipes:
                if recipe_already_verified(recipe, current_audit):
                    continue
                while active_runs():
                    write_state('esperando_lote_activo', active_runs=active_runs(), recipe=recipe, cycle=cycle)
                    time.sleep(max(10, args.poll_seconds))
                write_state('ejecutando_receta', recipe=recipe, cycle=cycle, audit=current_audit)
                run(command)
                run([sys.executable, '-m', 'pytest', PROTOCOL_TEST, '-q'])
                summary = summarize()
                audit = audit_panel_v()
                write_state('receta_auditada', recipe=recipe, cycle=cycle, audit=audit, summary=summary)
                current_audit = audit
                if audit['all_required_verified']:
                    write_state('completa', recipe=recipe, cycle=cycle, audit=audit, summary=summary)
                    return 0
            after = audit_panel_v()
            verified_after = int(after['states'].get('verificada', 0))
            cycle += 1
            if verified_after <= verified_before:
                write_state('esperando_datos_disponibles', cycle=cycle, audit=after)
                time.sleep(max(30, args.cache_retry_seconds))
    except Exception as error:
        write_state('fallida', cycle=cycle, error=repr(error))
        raise
if __name__ == '__main__':
    raise SystemExit(main())
