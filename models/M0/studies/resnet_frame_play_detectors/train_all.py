"""Train the two one-vs-rest ResNet frame detectors."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import subprocess
import sys
from pathlib import Path
TARGETS = ('scrum', 'line')

def main() -> None:
    script = Path(__file__).with_name('train_binary_frame_resnet.py')
    passthrough_args = sys.argv[1:]
    for target in TARGETS:
        command = [sys.executable, str(script), f'TARGET={target}', *passthrough_args]
        print('=' * 80)
        print(f'Training {target} detector')
        print(' '.join(command))
        print('=' * 80)
        subprocess.run(command, check=True, env=os.environ.copy())
if __name__ == '__main__':
    main()
