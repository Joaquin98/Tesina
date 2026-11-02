import ast
import __future__
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load_functions(relative,names,namespace):
    path=ROOT/relative
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    from models.common.legacy_training import DATASET
    from models.common.study_data import study_csv_path
    namespace.update(Path=Path,pd=pd,np=np,CURRENT_DATASET=DATASET,study_csv_path=study_csv_path)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace


def test_study_csv_roles_resolve_to_current_files():
    from models.common.study_data import study_csv_path
    for legacy,current in [('train_natural.csv','train.csv'),('validation_fixed.csv','validation.csv'),('test_natural.csv','test.csv')]:
        path=study_csv_path(ROOT/'data/clips_dataset'/legacy)
        assert path==ROOT/'data/clips_dataset'/current
        assert path.is_file()
    arbitrary=ROOT/'models/M0/runs/new/predictions.csv'
    assert study_csv_path(arbitrary)==arbitrary


def test_natural_study_reads_current_disjoint_partitions():
    from models.common.legacy_training import current_frames
    from models.model_catalog import resolve_artifact
    ns=dict(SOURCE_SPLITS=ROOT/'data/class_imbalance_comparison/full/splits',_resolve_model_artifact=resolve_artifact,LABELS=('line','other','scrum'),SEED=42,current_frames=current_frames)
    ns=load_functions('models/M0/studies/natural_with_human_review/run_experiment.py',{'load_base_splits'},ns)
    parts=ns['load_base_splits'](False,'natural')
    assert [len(p) for p in parts]==[2285,980,817]
    expected=current_frames()
    assert all(set(a.video_name)==set(b.video_name) for a,b in zip(parts,expected))


@pytest.mark.parametrize('relative',['models/M0/studies/random_search_resnet_bigru_20261022/run_search.py','models/M0/studies/random_search_resnet_bigru_20261022_extension04/frozen_core.py'])
def test_search_balances_only_after_loading_current_partitions(relative,tmp_path):
    from models.common.legacy_training import current_frames
    from models.model_catalog import resolve_artifact
    ns=dict(HERE=tmp_path,ROOT=ROOT,SOURCE=ROOT/'models/M0',LABELS=['line','other','scrum'],SPLIT_SEED=42,current_frames=current_frames,_resolve_model_artifact=resolve_artifact,save=lambda *a:None,digest=lambda *a:'digest')
    ns=load_functions(relative,{'prepare','manifest','counts'},ns)
    train,validation,test=ns['prepare']()
    original=current_frames()
    assert len(train)==1260
    assert set(train.video_name)<=set(original[0].video_name)
    assert set(validation.video_name)==set(original[1].video_name)
    assert set(test.video_name)==set(original[2].video_name)
    assert train.tag.value_counts().to_dict()=={'line':420,'other':420,'scrum':420}


def test_quality_study_does_not_split_training_rows_again():
    path=ROOT/'models/M0/studies/keras_applications_comparison/run_quality_comparison.py'
    tree=ast.parse(path.read_text(encoding='utf8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not any(k.arg=='validation_split' for k in node.keywords)


def test_rgb_bank_reads_shared_cache_on_demand(monkeypatch):
    from models.common import study_data as data
    rows=pd.DataFrame([dict(video_name='one.mp4',tag='line'),dict(video_name='two.mp4',tag='other')])
    monkeypatch.setattr(data,'current_frames',lambda: (rows,rows.iloc[:0],rows.iloc[:0]))
    calls=[]
    def rgb(name):
        calls.append(name)
        return np.full((200,2048),1 if name=='one.mp4' else 2,np.float32),np.arange(200)<7
    monkeypatch.setattr(data,'cached_rgb',rgb)
    bank=data.RGBFeatureStore()
    assert calls==[]
    assert bank.features.shape==(2,200,2048)
    np.testing.assert_array_equal(bank.features[[1,0]][:,0,0],[2,1])
    assert calls==['two.mp4','one.mp4']


def test_controlled_study_preserves_validation_membership():
    from models.common.legacy_training import current_frames
    ns=load_functions('models/M0/studies/independent_evaluation/run_simple_unique_validation.py', {'create_controlled_split'}, dict(current_frames=current_frames,LABELS=['line','other','scrum'],SEED=42,TRAIN_TARGET_PER_CLASS=480))
    train,validation,audit=ns['create_controlled_split']()
    expected=current_frames()
    assert set(validation.video_name)==set(expected[1].video_name)
    assert set(train.video_name)<=set(expected[0].video_name)
    assert audit['new_train_validation_overlap']==0
    assert len(train)==1440


def test_repeat_study_uses_current_partitions():
    from models.common.legacy_training import current_frames
    ns=load_functions('models/M1/studies/alternative_split_robustness/run_experiment.py', {'load_split'}, dict(current_frames=current_frames))
    for actual,expected in zip(ns['load_split'](2),current_frames()):
        assert set(actual.video_name)==set(expected.video_name)
