"""Audita una caché de pose sin volver a ejecutar el estimador."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from models.M1.config import NATURAL_TRAIN_CSV, POSE_MODEL_DEFAULT, POSE_CACHE_DIR, TEST_CSV
from models.M1.protocol import pose_cache_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--manifest', default=None)
    args = parser.parse_args()
    if args.manifest:
        manifest = pd.read_csv(_resolve_model_artifact(args.manifest))
    else:
        manifest = pd.concat([pd.read_csv(_resolve_model_artifact(NATURAL_TRAIN_CSV)), pd.read_csv(_resolve_model_artifact(TEST_CSV))], ignore_index=True)
    manifest = manifest[['video_name', 'tag']].drop_duplicates('video_name').reset_index(drop=True)
    rows = []
    missing = []
    for record in manifest.to_dict('records'):
        name = str(record['video_name'])
        path = pose_cache_path(name, args.model)
        if not _resolve_model_artifact(path).exists():
            missing.append(record)
            continue
        with _load_model_result_array(path, allow_pickle=False) as data:
            valid = np.asarray(data['valid'], dtype=bool)
            frame_mask = np.asarray(data['frame_mask'], dtype=bool)
            person_conf = np.asarray(data['person_conf'], dtype=np.float32)
            keypoint_conf = np.asarray(data['keypoint_conf'], dtype=np.float32)
        valid_conf = person_conf[valid]
        keypoint_values = keypoint_conf[keypoint_conf > 0]
        rows.append({'video_name': name, 'tag': record['tag'], 'frames': int(valid.shape[0]), 'frames_with_persons': int(np.any(valid, axis=1).sum()), 'frames_with_geometry': int(frame_mask.sum()), 'mean_people': float(valid.sum(axis=1).mean()), 'mean_person_confidence': float(valid_conf.mean()) if len(valid_conf) else 0.0, 'mean_keypoint_confidence': float(keypoint_values.mean()) if len(keypoint_values) else 0.0})
    out_dir = POSE_CACHE_DIR / 'audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'clip_audit.csv', index=False)
    summary = {'model': args.model, 'clips_requested': int(len(manifest)), 'clips_audited': int(len(rows)), 'clips_missing': int(len(missing)), 'by_class': pd.DataFrame(rows).groupby('tag').size().to_dict() if rows else {}, 'mean_frames_with_geometry': float(np.mean([row['frames_with_geometry'] for row in rows])) if rows else 0.0, 'mean_people': float(np.mean([row['mean_people'] for row in rows])) if rows else 0.0}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
