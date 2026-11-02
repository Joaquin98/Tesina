import ast
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_web_accepts_a_known_prepared_clip():
    # Exercise the validators without starting Flask or creating runtime folders.
    path = ROOT / 'src/web/app.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'_supports_training_preprocessing', '_validate_inference_source'}]
    namespace = {'Path': Path, '_resolve_model_artifact': lambda p: p, '_video_dimensions': lambda p: (224, 224), '_is_known_preprocessed_input': lambda *args: True}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    clip = ROOT / 'data/videos/clips_224/C0001.mp4'
    assert namespace['_supports_training_preprocessing'](clip)
    namespace['_validate_inference_source'](clip)

@pytest.mark.skipif(shutil.which('node') is None, reason='Node is needed to exercise the browser geometry')
def test_browser_overlay_uses_the_entire_prepared_frame():
    source = ROOT / 'src/web/static/js/model-roi-overlay.js'
    script = "const fs=require('fs'),vm=require('vm'),s={window:{}};vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),s);console.log(JSON.stringify(s.window.ModelRoiOverlay.normalizationGeometry(224,224)));"
    geometry = json.loads(subprocess.check_output(['node', '-e', script, str(source)]))
    assert geometry == {'sourceX': 0, 'sourceY': 0, 'sourceWidth': 224, 'sourceHeight': 224}

@pytest.mark.parametrize('module,class_name', [('models.M2.predictor', 'RGBYoloHighresQuantileFusionDetector'), ('models.M3.predictor', 'M3CrossFoldDetector')])
def test_live_prepared_window_does_not_require_a_second_source_video(module, class_name):
    cls = getattr(importlib.import_module(module), class_name)
    model = object.__new__(cls)
    model.pose_input_mode = 'source_equivalent_roi'
    model._predict_feature_batch = lambda frames, batch: np.zeros((len(frames), 2048), dtype=np.float32)
    model._begin_auxiliary_sequence = lambda count: None
    observed = []
    model._on_raw_sampled_frame = lambda frame: observed.append(frame.copy())
    model._finish_auxiliary_sequence = lambda: None
    model._predict_sequence_window = lambda *args: np.asarray([[.2, .5, .3]], dtype=np.float32)
    rgb = np.random.default_rng(5).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    result = model.predict_live_window([rgb], sequence_len=200)
    np.testing.assert_array_equal(observed[0], rgb[:, :, ::-1])
    np.testing.assert_allclose(result, [[.2, .5, .3]])
