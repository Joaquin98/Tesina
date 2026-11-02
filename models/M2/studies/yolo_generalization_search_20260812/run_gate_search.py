"""Busca compuertas basadas en confianza histÃ³rica y calidad YOLO."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json, os, sys
from pathlib import Path
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data/player_geometry_fusion/pose_cache/frames_200')
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[1])))
import joblib, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite
OUT = experiment_directory(__file__) / 'gate_search_seed42'

def quality(frame):
    out = []
    for n in frame.video_name.astype(str):
        d = run_suite.load_clip(n)
        valid = d['valid'].astype(bool)
        pc = d['person_conf'].astype(float)
        kc = d['keypoint_conf'].astype(float)
        count = valid.sum(1)
        pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(count, 1))
        km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(count * kc.shape[-1], 1))
        out.append([pm.mean(), km.mean(), np.clip(count / 30, 0, 1).mean(), (count > 0).mean(), (pm > 0.35).mean(), (km > 0.3).mean()])
    return np.asarray(out, float)

def main():
    train, val, test = R.common.load_base_splits(False, 'natural')
    yv = R.labels(val)
    yt = R.labels(test)
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', val)
    bt = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    rv = quality(val)
    rt = quality(test)
    gv = []
    gt = []
    for seed in (17, 42, 2026):
        root = Path('data/historical_locked_yolo_residual_20260811') / f'results_seed{seed}'
        res = json.loads(_resolve_model_artifact(root / 'selected_result.json').read_text())
        art = joblib.load(_resolve_model_artifact(root / 'selected_model.joblib'))
        cfg = res['selected_configuration']
        rep = cfg['representation']
        xv = R.cached_features(val, 'validation', rep, OUT / 'cache')
        xt = R.cached_features(test, 'test', rep, OUT / 'cache')
        gv.append(R.normalize(art['model'].predict_proba(art['pca'].transform(art['scaler'].transform(xv)))))
        gt.append(R.normalize(art['model'].predict_proba(art['pca'].transform(art['scaler'].transform(xt)))))
    geomv = R.normalize(np.average(gv, axis=0, weights=[0.2, 0.2, 0.6]))
    geomt = R.normalize(np.average(gt, axis=0, weights=[0.2, 0.2, 0.6]))
    rows = []
    for conf in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for direction in ('high', 'low'):
                    av = (bv.max(1) < conf) & (rv[:, 0] >= threshold if direction == 'high' else rv[:, 0] < threshold)
                    at = (bt.max(1) < conf) & (rt[:, 0] >= threshold if direction == 'high' else rt[:, 0] < threshold)
                    v = bv.copy()
                    t = bt.copy()
                    v[av] = (1 - alpha) * bv[av] + alpha * geomv[av]
                    t[at] = (1 - alpha) * bt[at] + alpha * geomt[at]
                    v = R.normalize(v)
                    t = R.normalize(t)
                    rows.append({'val_f1': float(f1_score(yv, v.argmax(1), average='macro')), 'val_acc': float(accuracy_score(yv, v.argmax(1))), 'test_f1': float(f1_score(yt, t.argmax(1), average='macro')), 'test_acc': float(accuracy_score(yt, t.argmax(1))), 'conf': conf, 'quality': threshold, 'alpha': alpha, 'direction': direction, 'changed': int(av.sum())})
    rows = sorted(rows, key=lambda x: (x['val_f1'], x['val_acc']), reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ranking.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows[:30], indent=2), flush=True)
if __name__ == '__main__':
    main()
