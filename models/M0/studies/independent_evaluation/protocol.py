"""Shared, self-contained protocol utilities for the independent evaluation.

This module never writes outside data/independent_evaluation.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
EXPERIMENT_DIR = experiment_directory(__file__)
REPO_ROOT = EXPERIMENT_DIR.parents[1]
MODEL_DIR = REPO_ROOT / 'models' / 'M0'
DATA_DIR = REPO_ROOT / 'data'
SOURCE_TRAIN = CURRENT_DATASET / 'train.csv'
SOURCE_TEST = CURRENT_DATASET / 'test.csv'
BALANCED_TRAIN = CURRENT_DATASET / 'train.csv'
CLIP_ROOT = DATA_DIR / 'videos' / 'clips_224'
FEATURE_ROOT = MODEL_DIR / 'feature_cache'
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs'
SPLIT_DIR = OUTPUT_DIR / 'splits'
LABELS = ('line', 'other', 'scrum')
SEED = 20260712
_STOP_TOKENS = {'todo', 'otros', 'line', 'scrum', 'juego', 'salidas', 'salida', 'ataque', 'defensa', 'recepcion', 'recepciones', 'penales', 'penal', 'palos', 'pie', 'cap', 'cut', 'cls', 'otro', 'other'}

def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve_model_artifact(study_csv_path(path)))
    required = {'video_name', 'tag'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path} does not contain {sorted(missing)}')
    out = df[['video_name', 'tag']].copy()
    out['video_name'] = out['video_name'].astype(str).str.strip()
    out['tag'] = out['tag'].astype(str).str.strip().str.lower()
    return out[(out.video_name != '') & out.tag.isin(LABELS)].reset_index(drop=True)

def sha256_file(path: Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with _resolve_model_artifact(study_csv_path(path)).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def infer_source_group(video_name: str) -> str:
    """Infer a conservative source/match group from historical clip names.

    F/R identifiers are the strongest provenance signal. Older ``Todo_otros``
    clips encode teams/source tokens; these are canonicalized so reversed team
    order cannot cross a split. Uncertain clips are deliberately merged into
    broad groups instead of being treated as independent.
    """
    stem = Path(str(video_name)).stem
    stem = re.sub('_cut$', '', stem, flags=re.IGNORECASE)
    match = re.search('(?:^|_)([FR]\\d+)(?:_|$)', stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    tokens = [t for t in re.split('[^A-Za-z0-9]+', stem) if t]
    teams = []
    for token in tokens:
        low = token.lower()
        if low in _STOP_TOKENS or token.isdigit() or re.fullmatch('\\d+', token):
            continue
        if len(token) >= 3 and token.isalpha():
            teams.append(token.upper())
    teams = sorted(set(teams))
    if len(teams) >= 2:
        return 'TEAMS_' + '_'.join(teams[-3:])
    if teams:
        return 'SOURCE_' + teams[-1]
    if 'scrum' in stem.lower():
        return 'UNCERTAIN_SCRUM'
    if 'line' in stem.lower():
        return 'UNCERTAIN_LINE'
    return 'UNCERTAIN_OTHER'

def temporal_key(video_name: str) -> tuple[str, int | None]:
    stem = re.sub('_cut$', '', Path(str(video_name)).stem, flags=re.IGNORECASE)
    match = re.search('_(\\d+)$', stem)
    index = int(match.group(1)) if match else None
    prefix = re.sub('_\\d+$', '', stem).lower()
    return (prefix, index)

def source_manifest() -> pd.DataFrame:
    train, validation, test = current_frames()
    train = pd.concat([train, validation], ignore_index=True)
    train['historical_partition'] = 'development'
    test['historical_partition'] = 'test_817'
    all_df = pd.concat([train, test], ignore_index=True)
    conflicts = all_df.groupby('video_name').tag.nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise RuntimeError(f'Conflicting labels for {len(conflicts)} clips')
    all_df = all_df.drop_duplicates('video_name', keep='first').reset_index(drop=True)
    all_df['source_group'] = all_df.video_name.map(infer_source_group)
    temporal = all_df.video_name.map(temporal_key)
    all_df['temporal_prefix'] = [item[0] for item in temporal]
    all_df['temporal_index'] = [item[1] for item in temporal]
    all_df['clip_path'] = all_df.video_name.map(lambda n: str((CLIP_ROOT / n).resolve()))
    all_df['clip_exists'] = all_df.video_name.map(lambda n: _resolve_model_artifact(study_csv_path(CLIP_ROOT / n)).exists())
    return all_df

def class_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df.tag.value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}

def group_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df.source_group.value_counts().sort_index().items()}

def assert_disjoint(parts: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    names = list(parts)
    for i, left_name in enumerate(names):
        left = parts[left_name]
        for right_name in names[i + 1:]:
            right = parts[right_name]
            clip_overlap = sorted(set(left.video_name) & set(right.video_name))
            group_overlap = sorted(set(left.source_group) & set(right.source_group))
            findings.append({'left': left_name, 'right': right_name, 'clip_overlap': len(clip_overlap), 'group_overlap': len(group_overlap), 'clip_examples': clip_overlap[:10], 'group_examples': group_overlap[:10]})
            if clip_overlap or group_overlap:
                raise AssertionError(f'Leakage {left_name}/{right_name}: clips={len(clip_overlap)}, groups={len(group_overlap)}')
    return findings

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_candidates(path: Path | None=None) -> list[dict[str, object]]:
    target = path or EXPERIMENT_DIR / 'candidates.json'
    return json.loads(_resolve_model_artifact(study_csv_path(target)).read_text(encoding='utf-8'))['candidates']

def stable_seed(*items: object) -> int:
    raw = '|'.join(map(str, items)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little')

def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
