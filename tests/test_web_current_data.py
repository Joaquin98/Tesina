import ast
import csv
import logging
import sys
import threading
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.model_catalog import resolve_artifact


def web_namespace():
    path = ROOT / 'src/web/app.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = {'_read_csv_rows', '_initial_clip_label', '_initial_training_catalog',
                 '_build_initial_training_catalog', 'list_initial_training_clips', '_to_int'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name.startswith('INITIAL_') and name not in {'INITIAL_TRAINING_PREVIEW_LOCK'} for name in names):
                nodes.append(node)
    app = Flask(__name__)
    ns = dict(Path=Path, PROJECT_ROOT=ROOT, WEB_RUNTIME=ROOT/'.runtime/web', csv=csv, Counter=Counter, threading=threading,
              _resolve_model_artifact=resolve_artifact, logger=logging.getLogger(__name__),
              app=app, jsonify=jsonify, request=request)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def test_catalog_uses_current_labels_and_three_disjoint_splits():
    ns = web_namespace()
    clips, summary = ns['_build_initial_training_catalog']()
    assert len(clips) == 4082
    assert Counter(clip['split'] for clip in clips) == {'train': 2285, 'validation': 980, 'test': 817}
    with (ROOT/'data/clips_dataset/manifest.csv').open(newline='', encoding='utf-8') as handle:
        truth = {row['video_name']: row for row in csv.DictReader(handle)}
    assert all(clip['label'] == truth[clip['filename']]['tag'] for clip in clips)
    assert summary['unassigned_clips'] == 0
    assert summary['validation_clips'] == 980
    assert summary['unique_training_clips'] == 2285
    assert summary['training_examples'] >= 2285


def test_validation_filter_returns_only_validation_clips():
    ns = web_namespace()
    with ns['app'].test_request_context('/api/training-clips?split=validation'):
        response = ns['list_initial_training_clips']()
        assert not isinstance(response, tuple)
        payload = response.get_json()
        assert payload['success']
        assert payload['pagination']['total'] == 980
        assert {clip['split'] for clip in payload['clips']} == {'validation'}


def test_main_and_demo_use_current_match_library():
    for filename in ['app.py', 'demo_app.py']:
        path = ROOT/'src/web'/filename
        tree = ast.parse(path.read_text(encoding='utf-8'))
        app = Flask(__name__)
        ns = dict(PROJECT_ROOT=ROOT, app=app)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                (isinstance(t, ast.Name) and t.id == 'VIDEO_LIBRARY') or
                (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == 'VIDEO_LIBRARY')
                for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        assert ns.get('VIDEO_LIBRARY', app.config.get('VIDEO_LIBRARY')) == ROOT/'data/videos/partidos_evaluacion'


def test_training_history_uses_local_manifests_without_old_reports(tmp_path, monkeypatch):
    import json
    from src.web import training_history as history
    run = tmp_path/'models/M1/runs/example'
    run.mkdir(parents=True)
    (run/'run.json').write_text(json.dumps({'schema': 1, 'status': 'complete', 'family': 'M1', 'validation': {'accuracy': .75, 'f1_macro': .7}, 'training': {'epochs': 2}, 'dataset': {'train': {'selected_examples': 12}, 'validation': {'selected_examples': 6}}}), encoding='utf8')
    pending = tmp_path/'models/M1/runs/.partial_run_example'
    pending.mkdir()
    (pending/'run.json').write_text('{}', encoding='utf8')
    monkeypatch.setattr(history, 'index', lambda: {'runs': []})
    payload = history.read_training_history(tmp_path)
    assert len(payload['local_runs']) == 1
    assert payload['local_runs'][0]['family'] == 'M1'
    assert payload['local_runs'][0]['validation_accuracy'] == .75
    assert payload['local_runs'][0]['training_examples'] == 12
    assert payload['historical_runs'] == []


def test_templates_do_not_link_to_removed_static_pages():
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {'href', 'src'} and value and value.startswith('/static/'):
                    relative = value.split('?', 1)[0].removeprefix('/static/')
                    assert (ROOT/'src/web/static'/relative).is_file(), relative
    for path in (ROOT/'src/web/templates').glob('*.html'):
        Links().feed(path.read_text(encoding='utf8'))


def test_analysis_keeps_model_identifier_and_rejects_inference_errors(tmp_path):
    import json
    import time
    from datetime import datetime
    from types import SimpleNamespace
    import pytest
    path = ROOT/'src/web/app.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef) and n.name == '_execute_analysis')
    selected = []
    engine = SimpleNamespace(load_model=selected.append, analyze_video=lambda *a, **kw: {'status': 'success', 'metadata': {'duration': 7}, 'detections': []})
    ns = dict(time=time, datetime=datetime, json=json, logger=logging.getLogger(__name__),
              app=SimpleNamespace(config={'RESULTS_FOLDER': tmp_path}), inference_engine=engine,
              _resolve_video_path=lambda p: ROOT/'data/videos/clips_224/C0001.mp4',
              _resolve_model_artifact=resolve_artifact, _validate_inference_source=lambda p: None,
              _to_float=lambda v, d: float(v) if v is not None else d)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    request = {'model_name': 'm3_h2_d2_cross_fold_ensemble', 'video_path': 'C0001.mp4'}
    assert ns['_execute_analysis'](request)['success']
    assert selected == [request['model_name']]
    for output in tmp_path.iterdir():
        output.unlink()
    engine.analyze_video = lambda *a, **kw: {'status': 'error', 'error': 'Cannot decode video'}
    with pytest.raises(RuntimeError, match='Cannot decode video'):
        ns['_execute_analysis'](request)
    assert not list(tmp_path.iterdir())


def test_m2_is_selectable_with_weights_even_without_optional_summary():
    import json
    from types import SimpleNamespace
    path = ROOT/'src/web/keras_inference_engine.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'KerasInferenceEngine')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'list_available_models')
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(method.body) if assigns(n, 'highres_deployment'))
    end = next(i for i, n in enumerate(method.body) if assigns(n, 'event_verifier_path'))
    ns = dict(self=SimpleNamespace(models_dir=ROOT/'models'), json=json, models=[],
              _resolve_model_artifact=resolve_artifact,
              RGB_YOLO_HIGHRES_DEPLOYMENT_RELATIVE='data/player_geometry_highres_pose/outputs/deployment_quantile_100frames',
              RGB_YOLO_HIGHRES_MODEL_NAME='rgb_yolo_quantile_highres_roi',
              RGB_YOLO_QUANTILE_FUSION_MODEL_NAME='rgb_yolo_quantile_fusion_all',
              PLAYER_GEOMETRY_FUSION_POSE_MODEL='yolo11n-pose.pt',
              quantile_feature_root=ROOT/'data/joint_multimodal_20260813/rgb_dynamics_seed42')
    exec(compile(ast.Module(body=method.body[start:end], type_ignores=[]), str(path), 'exec'), ns)
    assert len(ns['models']) == 1
    assert ns['models'][0]['name'] == 'rgb_yolo_quantile_highres_roi'
    assert ns['models'][0]['metrics']['accuracy'] is None
    assert not ns['models'][0]['recommended']
    ns.update(RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME='rgb_yolo_historical_primera_calibrated', PLAYER_GEOMETRY_FUSION_MODEL_NAME='player_geometry_fusion_calibrated')
    profiles = next(i for i, n in enumerate(method.body) if assigns(n, 'recommendation_profiles'))
    exec(compile(ast.Module(body=method.body[profiles:-1], type_ignores=[]), str(path), 'exec'), ns)
    assert not ns['models'][0]['recommended']
