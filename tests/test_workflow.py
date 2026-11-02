"""Dataset separation and saved-run contracts for the unified workflow."""
from pathlib import Path
import csv
import importlib
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def manifests(tmp_path, overlap=False):
    for split in ('train', 'validation', 'test'):
        path = tmp_path / (split + '.csv')
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['video_name', 'tag', 'path', 'sha256'])
            writer.writeheader()
            for label in ('line', 'other', 'scrum'):
                name = f'{split}_{label}.mp4'
                if overlap and split == 'validation' and label == 'line':
                    name = 'train_line.mp4'
                video = tmp_path / name
                video.write_bytes(name.encode())
                writer.writerow(dict(video_name=name, tag=label, path=str(video), sha256=''))
    return tmp_path

def test_dataset_requires_disjoint_train_validation_and_test(tmp_path):
    from models.common.workflow import read_splits
    with pytest.raises(ValueError, match='overlap|compart|solap'):
        read_splits(manifests(tmp_path, overlap=True))

def test_dataset_detects_identical_content_under_different_names(tmp_path):
    from models.common.workflow import read_splits
    manifests(tmp_path)
    (tmp_path / 'validation_line.mp4').write_bytes((tmp_path / 'train_line.mp4').read_bytes())
    with pytest.raises(ValueError, match='overlap|compart|solap'):
        read_splits(tmp_path)

def test_dataset_preserves_labels_and_split_membership(tmp_path):
    from models.common.workflow import read_splits
    splits = read_splits(manifests(tmp_path))
    assert {name: len(rows) for name, rows in splits.items()} == {'train': 3, 'validation': 3, 'test': 3}
    assert {row['tag'] for row in splits['train']} == {'line', 'other', 'scrum'}

@pytest.mark.parametrize('family', ['M0', 'M1', 'M2', 'M3'])
def test_each_family_exposes_training_and_saved_run_loading(family):
    module = importlib.import_module(f'models.{family}.workflow')
    assert callable(module.train)
    assert callable(module.load)

def test_no_root_experiment_dependency_in_model_imports():
    import ast
    for path in (ROOT / 'models').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] != 'experiments', path
            if isinstance(node, ast.Import):
                assert all(item.name.split('.')[0] != 'experiments' for item in node.names), path

def test_cli_exposes_all_workflow_stages():
    import subprocess
    help_text = subprocess.check_output([sys.executable, '-B', str(ROOT / 'tools/modelos.py'), '--help'], cwd=ROOT, text=True)
    assert all(command in help_text for command in ('cache', 'train', 'predict', 'evaluate', '--run', '--no-cache'))

def test_original_repository_paths_are_rejected():
    from models.common.workflow import local_path
    with pytest.raises(ValueError, match='original'):
        local_path(ROOT.parents[1] / 'models/foreign')

def test_cached_numpy_payloads_are_excluded_from_git():
    import subprocess
    for name in ('.runtime/cache/features/sample.npz', 'models/M0/.partial_run_example/run.json'):
        assert subprocess.run(['git', 'check-ignore', '--no-index', name], cwd=ROOT, capture_output=True).returncode == 0

def test_saved_run_checks_family_and_file_integrity(tmp_path):
    import json
    from models.common.training import read_run
    from models.inference_cache import file_digest
    artifact = tmp_path / 'head.keras'
    artifact.write_bytes(b'weights')
    manifest = dict(schema=1, status='complete', family='M0', labels=['line', 'other', 'scrum'], artifacts={'model': {'path': artifact.name, 'sha256': file_digest(artifact)}}, references={})
    (tmp_path / 'run.json').write_text(json.dumps(manifest))
    assert read_run(tmp_path, 'M0')['family'] == 'M0'
    with pytest.raises(ValueError, match='modelo'):
        read_run(tmp_path, 'M1')
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='modificado'):
        read_run(tmp_path, 'M0')

def test_training_does_not_overwrite_an_existing_run(tmp_path, monkeypatch):
    from models.common import workflow
    monkeypatch.setattr(workflow, 'configure', lambda threads: None)
    with pytest.raises(FileExistsError):
        workflow.train_run('M0', dataset=tmp_path / 'unused', output=tmp_path)
