"""Create immutable grouped outer/inner partitions before any new evaluation."""
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
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import LABELS, OUTPUT_DIR, SEED, SPLIT_DIR, assert_disjoint, class_counts, source_manifest, write_json

def split_frame(df: pd.DataFrame, splitter: StratifiedGroupKFold):
    return splitter.split(df, y=df.tag, groups=df.source_group)

def temporal_overlap(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return sorted(set(left.temporal_prefix) & set(right.temporal_prefix))

def ensure_all_classes(name: str, df: pd.DataFrame) -> None:
    missing = set(LABELS) - set(df.tag)
    if missing:
        raise AssertionError(f'{name} is missing classes: {sorted(missing)}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--inner-folds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    all_df = source_manifest()
    if not all(all_df.clip_exists):
        raise FileNotFoundError(f'Missing {int((~all_df.clip_exists).sum())} labelled clip files')
    all_df = all_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUTPUT_DIR / 'nested_source_manifest.csv', index=False)
    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    fold_summaries = []
    for outer_index, (development_idx, evaluation_idx) in enumerate(split_frame(all_df, outer), 1):
        development = all_df.iloc[development_idx].copy().reset_index(drop=True)
        evaluation = all_df.iloc[evaluation_idx].copy().reset_index(drop=True)
        ensure_all_classes('outer development', development)
        ensure_all_classes('outer evaluation', evaluation)
        inner_splitter = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + outer_index)
        inner_train_idx, validation_idx = next(split_frame(development, inner_splitter))
        train = development.iloc[inner_train_idx].copy().reset_index(drop=True)
        validation = development.iloc[validation_idx].copy().reset_index(drop=True)
        ensure_all_classes('inner train', train)
        ensure_all_classes('inner validation', validation)
        findings = assert_disjoint({'train': train, 'validation': validation, 'evaluation': evaluation})
        temporal_findings = []
        for left_name, left, right_name, right in (('train', train, 'validation', validation), ('train', train, 'evaluation', evaluation), ('validation', validation, 'evaluation', evaluation)):
            overlap = temporal_overlap(left, right)
            temporal_findings.append({'left': left_name, 'right': right_name, 'temporal_prefix_overlap': len(overlap), 'examples': overlap[:10]})
            if overlap:
                raise AssertionError(f'Temporal-source leakage {left_name}/{right_name}: {overlap[:5]}')
        classes = np.array(LABELS)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train.tag)
        class_weights = {label: float(weight) for label, weight in zip(classes, weights)}
        fold_dir = SPLIT_DIR / f'outer_{outer_index:02d}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(fold_dir / 'train.csv', index=False)
        validation.to_csv(fold_dir / 'validation.csv', index=False)
        evaluation.to_csv(fold_dir / 'evaluation.csv', index=False)
        summary = {'outer_fold': outer_index, 'seed': args.seed, 'outer_folds': args.outer_folds, 'inner_folds': args.inner_folds, 'balance_strategy': 'no resampling; class weights from inner training only', 'class_weights_by_label': class_weights, 'train': {'rows': len(train), 'unique_clips': int(train.video_name.nunique()), 'groups': int(train.source_group.nunique()), 'classes': class_counts(train)}, 'validation': {'rows': len(validation), 'unique_clips': int(validation.video_name.nunique()), 'groups': int(validation.source_group.nunique()), 'classes': class_counts(validation)}, 'evaluation': {'rows': len(evaluation), 'unique_clips': int(evaluation.video_name.nunique()), 'groups': int(evaluation.source_group.nunique()), 'classes': class_counts(evaluation)}, 'overlap_checks': findings, 'temporal_overlap_checks': temporal_findings}
        write_json(fold_dir / 'split_audit.json', summary)
        fold_summaries.append(summary)
    write_json(OUTPUT_DIR / 'nested_split_summary.json', fold_summaries)
    print(json.dumps({'folds': len(fold_summaries), 'all_overlap_checks_zero': True, 'output': str(SPLIT_DIR)}, indent=2))
if __name__ == '__main__':
    main()
