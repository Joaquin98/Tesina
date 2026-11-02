"""Training implementations and their data paths survive source relocation."""
import ast
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.mark.parametrize('family,function', [('M0', 'fit_m0'), ('M1', 'fit_predict_m1'), ('M2', 'fit_predict_m2'), ('M3', 'fit_predict_m3_or_m4')])
def test_comparison_trainers_live_in_their_own_family(family, function):
    module = importlib.import_module(f'models.{family}.comparison_training')
    assert getattr(module, function).__module__ == module.__name__
    assert Path(module.__file__).resolve().is_relative_to(ROOT / 'models' / family)

@pytest.mark.parametrize('module,legacy', [
    ('models.M1.config', 'player_geometry_fusion'),
    ('models.M2.multimodal.run_rgb_dynamics', 'joint_multimodal_20260813'),
    ('models.M3.run_temporal_architecture_suite', 'player_geometry_highres_pose'),
])
def test_training_assets_stay_at_their_existing_paths(module, legacy):
    loaded = importlib.import_module(module)
    assert getattr(loaded, 'EXPERIMENT_DIR', getattr(loaded, 'HERE', None)) == ROOT / 'data' / legacy
    assert getattr(loaded, 'ROOT', getattr(loaded, 'REPO', ROOT)) == ROOT

def test_old_code_paths_resolve_to_one_new_source():
    from models.model_catalog import resolve_artifact
    old = ROOT / 'data/player_geometry_highres_pose/run_temporal_architecture_suite.py'
    new = ROOT / 'models/M3/run_temporal_architecture_suite.py'
    assert not old.exists()
    assert resolve_artifact(old) == new
    assert new.is_file()

def test_training_subprocess_commands_use_existing_source_files():
    from models.M2.run_experiment import commands
    for command in commands(workers=1, batch_size=1).values():
        assert Path(command[1]).is_file(), command
        assert Path(command[1]).is_relative_to(ROOT / 'models/M2')

def test_shared_sequence_builds_dual_inputs_from_the_comparison_data(monkeypatch):
    import numpy as np
    import pandas as pd
    import models.training_common as common
    import models.common.comparison.data as data
    monkeypatch.setattr(common, 'load_resnet', lambda row: (np.zeros((200, 2048), np.float32), np.ones(200, bool), 200))
    monkeypatch.setattr(common, 'load_yolo', lambda row: {'sample': row['sample']})
    monkeypatch.setattr(common, 'labels_for', lambda frame: np.zeros(len(frame), np.int64))
    monkeypatch.setattr(data, 'geometry_sequence', lambda raw: np.full((200, 62), raw['sample'], np.float32))
    sequence = common.ResNetCacheSequence(pd.DataFrame({'sample': [7]}), batch_size=1, shuffle=False, seed=0, dual=True)
    inputs, labels = sequence[0]
    assert inputs['rgb'].shape == (1, 200, 2048)
    assert inputs['pose'].shape == (1, 200, 62)
    assert (inputs['pose'] == 7).all()
    assert labels.tolist() == [0]

@pytest.mark.parametrize('family', ['M0', 'M1', 'M2', 'M3'])
def test_extracted_trainers_keep_all_global_dependencies(family):
    import builtins
    import symtable
    module = importlib.import_module(f'models.{family}.comparison_training')
    table = symtable.symtable(Path(module.__file__).read_text(encoding='utf-8'), module.__file__, 'exec')
    def check(scope):
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                assert symbol.get_name() in vars(module) or hasattr(builtins, symbol.get_name()), (scope.get_name(), symbol.get_name())
        for child in scope.get_children():
            check(child)
    for child in table.get_children():
        check(child)

def test_comparison_dispatch_reuses_family_implementations():
    from models.common.comparison import models as dispatch
    from models.M0.comparison_training import fit_m0
    from models.M1.comparison_training import fit_predict_m1
    from models.M2.comparison_training import fit_predict_m2
    from models.M3.comparison_training import fit_predict_m3_or_m4
    assert dispatch.fit_m0 is fit_m0
    assert dispatch.fit_predict_m1 is fit_predict_m1
    assert dispatch.fit_predict_m2 is fit_predict_m2
    assert dispatch.fit_predict_m3_or_m4 is fit_predict_m3_or_m4

def test_comparison_runner_imports_the_relocated_event_verifier():
    from models.common.comparison import runner
    from models.M3.event_verifier_training import fit_verifier
    assert runner.fit_verifier is fit_verifier

def test_every_relocated_source_has_one_copy_and_preserves_its_asset_directory():
    from models.source_layout import SOURCE_MOVES, source_path, experiment_directory
    for old, new in SOURCE_MOVES.items():
        previous, current = ROOT / old, ROOT / new
        assert not previous.exists(), old
        assert current.is_file(), new
        assert source_path(previous) == current
        expected = ROOT / 'data' / Path(old).relative_to('experiments') if old.startswith('experiments/') else previous
        assert experiment_directory(current) == expected.parent

def test_model_fitting_implementations_are_outside_experiment_orchestration():
    for path in (ROOT / 'models/common/comparison').rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'fit', 'fit_transform', 'backward', 'step'}, (path, node.lineno)

def test_no_shared_trainer_implementations_remain_in_comparison_coordinator():
    tree = ast.parse((ROOT / 'models/common/comparison/models.py').read_text(encoding='utf-8'))
    assert {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))} == {'fit_predict'}
