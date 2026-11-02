import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import ctypes, sys, os
x = (ctypes.c_ulonglong * 8)()
ctypes.cast(x, ctypes.POINTER(ctypes.c_ulong))[0] = 64
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(x))
print('Python:', sys.executable)
print('RAM GB total/free:', round(x[1] / 2 ** 30, 2), round(x[2] / 2 ** 30, 2))
print('Keras backend env:', os.environ.get('KERAS_BACKEND', 'default'))
