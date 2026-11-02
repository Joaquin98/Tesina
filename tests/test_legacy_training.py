import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_legacy_config_uses_the_current_three_partitions():
    from models.M0 import config
    from models.model_catalog import resolve_artifact
    for name, filename in [('TRAIN_CSV', 'train.csv'), ('VALIDATION_CSV', 'validation.csv'), ('TEST_CSV', 'test.csv')]:
        path = Path(getattr(config, name))
        assert path == ROOT/'data/clips_dataset'/filename
        assert resolve_artifact(path) == path and path.is_file()


@pytest.mark.parametrize('module,family', [('models.M0.train','M0'), ('models.M0.train_best','M0'), ('models.M0.complete_model','M0'), ('models.M1.train','M1'), ('models.M2.train','M2'), ('models.M2.train_deployment','M2'), ('models.M3.train','M3')])
def test_training_entry_points_forward_the_current_workflow(module, family, monkeypatch):
    import tools.modelos as cli
    calls = []
    monkeypatch.setattr(cli, 'main', lambda argv=None: calls.append(argv))
    target = importlib.import_module(module)
    target.main(['--dataset', 'current_dataset', '--epochs', '1', '--output', 'new_run'])
    assert calls == [['train', '--family', family, '--dataset', 'current_dataset', '--epochs', '1', '--output', 'new_run']]


def test_specialized_trainers_use_explicit_validation():
    files = ['train_light_live.py','train_all_extractors.py','train_frame_only_baseline.py','grid_search.py','optuna_search.py','hyperparameter_tuning.py','train_best_focused.py','train_best_human_review.py']
    for name in files:
        path = ROOT/'models/M0'/name
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not any(key.arg == 'validation_split' for key in node.keywords), (name, node.lineno)


def test_current_partition_reader_rejects_overlap(tmp_path):
    import csv
    from models.common.legacy_training import current_frames
    for split in ['train','validation','test']:
        with (tmp_path/f'{split}.csv').open('w',encoding='utf8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['clip_id','video_name','tag','path','sha256'])
            writer.writeheader()
            writer.writerow(dict(clip_id='shared' if split != 'test' else 'test',video_name=split+'.mp4',tag='line',path=split+'.mp4',sha256=split))
    with pytest.raises(ValueError, match='compartid|asignad|overlap'):
        current_frames(tmp_path)


def test_current_partition_counts_and_validation_membership():
    from models.common.legacy_training import current_frames
    train, validation, test = current_frames()
    assert [len(part) for part in (train, validation, test)] == [2285, 980, 817]
    assert set(validation['split']) == {'validation'}
    assert not (set(train['video_name']) & set(validation['video_name']))
    assert not (set(train['video_name']) & set(test['video_name']))


def test_auxiliary_validation_reads_only_the_saved_validation_partition(monkeypatch):
    import numpy as np
    import pandas as pd
    from models.common import legacy_training as legacy
    partitions = tuple(pd.DataFrame([{'video_name': split+'.mp4', 'tag': 'line'}]) for split in ['train','validation','test'])
    monkeypatch.setattr(legacy, 'current_frames', lambda: partitions)
    requested=[]
    def cached(name, extractor):
        requested.append(name)
        return np.zeros((200,2048),np.float32),np.arange(200)<7
    monkeypatch.setattr(legacy,'cached_rgb',cached)
    inputs,labels=legacy.validation_data(['line','other','scrum'])
    assert requested == ['validation.mp4']
    assert inputs[0].shape == (1,200,2048)
    assert inputs[1].sum() == 7
    assert labels.tolist() == [0]


def test_auxiliary_outputs_cannot_use_the_original(monkeypatch):
    from models.common.legacy_training import auxiliary_directory
    monkeypatch.setenv('TESINA_AUXILIARY_OUTPUT',str(ROOT.parents[1]/'unsafe_output'))
    with pytest.raises(ValueError,match='original'):
        auxiliary_directory('test')


def test_missing_features_are_prepared_instead_of_skipping_clips(monkeypatch,tmp_path):
    import numpy as np
    from models.common import legacy_training as legacy
    from models.common import features
    source=object();observed=[]
    monkeypatch.setattr(legacy,'_rgb_source',lambda extractor:source)
    monkeypatch.setenv('TESINA_CACHE_DIR',str(tmp_path))
    def prepare(model,video,**kwargs):
        observed.append((model,video,kwargs['cache_dir']))
        return np.ones((9,2048),np.float32),{}, {'status':'created'},{}
    monkeypatch.setattr(features,'prepare',prepare)
    values,mask=legacy.cached_rgb('C0001.mp4')
    assert observed == [(source,ROOT/'data/videos/clips_224/C0001.mp4',str(tmp_path))]
    assert values.shape == (200,2048) and mask.sum()==9


def test_descriptor_helper_import_does_not_change_pose_configuration(monkeypatch,tmp_path):
    import os
    module=importlib.import_module('models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite')
    monkeypatch.setenv('PLAYER_GEOMETRY_LENGTH','37')
    monkeypatch.setenv('PLAYER_GEOMETRY_POSE_CACHE_ROOT',str(tmp_path))
    importlib.reload(module)
    assert os.environ['PLAYER_GEOMETRY_LENGTH']=='37'
    assert os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT']==str(tmp_path)
