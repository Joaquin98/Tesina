"""Exporta los Random Forest geométricos necesarios para la inferencia web."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.M1.config import RESULTS_DIR
from models.M1.late_fusion import geometry_statistics
from models.M1.protocol import build_partition

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'web_export'))
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ensemble_summary = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'ensemble_same_rnn' / 'summary.csv')).set_index('seed')
    manifest = {'model_name': 'player_geometry_fusion_calibrated', 'pose_model': args.pose_model, 'class_weights': {'line': 0.8, 'other': 1.0, 'scrum': 0.9}, 'seeds': seeds, 'random_forest_trees': int(args.trees), 'runs': []}
    for seed in seeds:
        train_df, _, _, protocol = build_partition(seed, require_geometry=True, pose_model=args.pose_model)
        train_x, train_y, _ = geometry_statistics(train_df, args.pose_model, 'both')
        forest = RandomForestClassifier(n_estimators=args.trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
        forest.fit(train_x, train_y)
        artifact_path = output_root / f'geometry_rf_seed_{seed}.joblib'
        joblib.dump(forest, artifact_path, compress=3)
        row = ensemble_summary.loc[seed]
        manifest['runs'].append({'seed': seed, 'geometry_alpha': float(row['alpha_geometry']), 'same_rnn_alpha': float(row['alpha_same_rnn_ensemble']), 'rf_path': artifact_path.name, 'geometry_features': int(train_x.shape[1]), 'protocol': protocol})
        print(json.dumps(manifest['runs'][-1], ensure_ascii=False))
    (output_root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Artefactos web exportados en: {output_root}')
if __name__ == '__main__':
    main()
