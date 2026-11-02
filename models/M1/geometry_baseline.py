"""Controles clÃ¡sicos sobre las estadÃ\xadsticas temporales de la cachÃ©."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import RESULTS_DIR
from models.M1.late_fusion import geometry_statistics
from models.M1.protocol import build_partition

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    rows = []
    for seed in seeds:
        train_df, _, test_df, _ = build_partition(seed, require_geometry=True, pose_model=args.pose_model)
        for kind in ('positions', 'poses', 'both'):
            train_x, train_y, _ = geometry_statistics(train_df, args.pose_model, kind)
            test_x, test_y, _ = geometry_statistics(test_df, args.pose_model, kind)
            classifier = RandomForestClassifier(n_estimators=args.trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
            classifier.fit(train_x, train_y)
            predictions = classifier.predict(test_x)
            rows.append({'seed': seed, 'kind': kind, 'features': train_x.shape[1], 'test_accuracy': accuracy_score(test_y, predictions), 'f1_macro': f1_score(test_y, predictions, average='macro', zero_division=0)})
    output_dir = RESULTS_DIR / 'geometry_baselines'
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'per_seed.csv', index=False)
    summary = frame.groupby('kind', as_index=False).agg(runs=('seed', 'count'), test_accuracy_mean=('test_accuracy', 'mean'), test_accuracy_std=('test_accuracy', 'std'), f1_macro_mean=('f1_macro', 'mean'), f1_macro_std=('f1_macro', 'std'))
    summary.to_csv(output_dir / 'summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Resultados: {output_dir}')
if __name__ == '__main__':
    main()
