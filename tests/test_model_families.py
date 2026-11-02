"""The four public model packages keep one consistent loading interface."""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_model_verification_does_not_require_discarded_reports(tmp_path, monkeypatch, capsys):
    from tools import modelos
    asset = tmp_path / 'weights.keras'
    asset.write_bytes(b'model')
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    catalog = {'objects': {digest: {'path': 'weights.keras'}}, 'aliases': {'models/checkpoint.keras': 'weights.keras', 'data/history/report.json': 'old/report.json'}, 'archived_reports': {'old/report.json': 'reports/report.json'}}
    monkeypatch.setattr(modelos, 'ROOT', tmp_path)
    monkeypatch.setattr(modelos, 'index', lambda: catalog)
    def resolve(path):
        relative = Path(path).relative_to(tmp_path).as_posix()
        target = catalog['aliases'][relative]
        if target in catalog['archived_reports']:
            raise FileNotFoundError('Discarded historical report')
        return tmp_path / target
    monkeypatch.setattr(modelos, 'resolve_artifact', resolve)
    monkeypatch.setattr(sys, 'argv', ['modelos.py', 'verify'])
    modelos.main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'ok'
    assert result['unique_weight_files'] == 1
    assert result['aliases'] == 1
    assert result['archived_result_aliases'] == 1

def test_model_verification_still_rejects_missing_local_assets(tmp_path, monkeypatch):
    from tools import modelos
    catalog = {'objects': {}, 'aliases': {'models/checkpoint.keras': 'missing.keras'}, 'archived_reports': {}}
    monkeypatch.setattr(modelos, 'ROOT', tmp_path)
    monkeypatch.setattr(modelos, 'index', lambda: catalog)
    monkeypatch.setattr(modelos, 'resolve_artifact', lambda path: tmp_path / 'missing.keras')
    monkeypatch.setattr(sys, 'argv', ['modelos.py', 'verify'])
    with pytest.raises(AssertionError):
        modelos.main()

@pytest.mark.parametrize('family,module,class_name', [
    ('M0','models.M0.web_adapter','RugbyPlayDetector'),
    ('M1','models.M1.inference','M1H2D2CrossFoldDetector'),
    ('M2','models.M2.predictor','RGBYoloHighresQuantileFusionDetector'),
    ('M3','models.M3.predictor','M3H2D2CrossFoldDetector'),
])
def test_family_loader_preserves_selected_constructor(family,module,class_name):
    package=importlib.import_module('models.'+family)
    constructor=Mock(return_value=SimpleNamespace())
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory).resolve()
        with patch.dict(sys.modules,{module:SimpleNamespace(**{class_name:constructor})}):
            loaded=package.load_model(root)
        assert loaded is constructor.return_value
        if family=='M0':
            constructor.assert_called_once_with(str(root/'models/M0'),model_name='best_model/best_model.keras',extractor_name='resnet50')
        else:constructor.assert_called_once_with(root)


def test_web_catalog_reads_size_from_resolved_artifact(tmp_path,monkeypatch):
    from src.web import keras_inference_engine as engine
    root=tmp_path/'delivery';models=root/'models';models.mkdir(parents=True)
    stored=tmp_path/'trained.joblib';stored.write_bytes(b'weights')
    aliases={str(root/engine.META_CORRECTION_ARTIFACT_RELATIVE):stored,
             str(root/engine.META_CORRECTION_PRIMARY_RELATIVE):stored}
    monkeypatch.setattr(engine,'_resolve_model_artifact',lambda p:aliases.get(str(p),Path(p)))
    monkeypatch.setattr(engine,'_model_artifact_glob',lambda *a,**kw:iter(()))
    listed=engine.KerasInferenceEngine(models).list_available_models()
    item=next(x for x in listed if x['name']==engine.META_CORRECTION_MODEL_NAME)
    assert item['size_mb']==round(stored.stat().st_size/(1024*1024),2)
