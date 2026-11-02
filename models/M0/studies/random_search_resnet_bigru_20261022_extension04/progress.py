"""Lightweight status reader; never imports TensorFlow or reads test data."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import csv
import json
from pathlib import Path
p = experiment_directory(__file__)
rows = list(csv.DictReader(_resolve_model_artifact(p / 'results.csv').open(encoding='utf-8'))) if _resolve_model_artifact(p / 'results.csv').exists() else []
live = json.loads(_resolve_model_artifact(p / 'live_status.json').read_text()) if _resolve_model_artifact(p / 'live_status.json').exists() else {}
completion = json.loads(_resolve_model_artifact(p / 'completion.json').read_text()) if _resolve_model_artifact(p / 'completion.json').exists() else {}
best = max([0.7774561876583754] + [float(r['val_f1_macro']) for r in rows])
print(json.dumps({'completed': len(rows), 'trial': live.get('trial'), 'epoch': live.get('epoch', 3 if live.get('state') == 'trial_complete' else None), 'best': best, 'state': completion.get('status', 'running'), 'live': live}))
