"""
grid_search_focused.py
======================
Focused (stage-2) hyperparameter search around top-performing grid-search regions.

Reuses the training pipeline from `grid_search.py` with a focused search
space and corresponding defaults.

Usage (from models/M0/):
  python grid_search_focused.py
  python grid_search_focused.py EPOCHS=20 RESUME=1
  python grid_search_focused.py EXTRACTORS=resnet50 RNN_TYPES=gru,lstm

Notes:
- Defaults are tuned from observed top-10 trends:
  - extractor: resnet50
  - bidirectional recurrent models
  - rnn_units near 64/128
  - dropout around 0.3 (with local neighborhood)
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import sys
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        os.environ[k.upper()] = v
from models.M0 import grid_search as gs
def _env_list(key, default):
    raw = os.environ.get(key, '')
    return [x.strip() for x in raw.split(',') if x.strip()] or default
gs.EXTRACTORS = _env_list('EXTRACTORS', ['resnet50'])
gs.RNN_TYPES = _env_list('RNN_TYPES', ['gru', 'lstm'])
gs.PARAM_GRID = {'rnn_units': [64, 128], 'bidirectional': [True], 'dropout': [0.2, 0.3, 0.4, 0.5], 'dense_units': [0, 64], 'learning_rate': [0.0001, 0.0003, 0.001], 'batch_size': [16, 24, 32]}
if __name__ == '__main__':
    print('=' * 64)
    print('  Focused Grid Search (stage-2)')
    print(f'  Extractors : {gs.EXTRACTORS}')
    print(f'  RNN types  : {gs.RNN_TYPES}')
    combos = 1
    for vals in gs.PARAM_GRID.values():
        combos *= len(vals)
    print(f'  Param combos per extractor×RNN : {combos}')
    print(f'  Total focused trials : {len(gs.EXTRACTORS) * len(gs.RNN_TYPES) * combos}')
    print('=' * 64)
    gs.main()
