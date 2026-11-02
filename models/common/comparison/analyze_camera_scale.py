"""Summarize normalized player scale and pose visibility in the six test matches."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import numpy as np
import pandas as pd
HERE = experiment_directory(__file__)
MANIFEST = HERE / 'outputs' / 'master_examples.csv'
OUTPUT = HERE / 'outputs' / 'camera_scale_by_match.csv'

def summarize_match(rows: pd.DataFrame, *, block_stride: int=5) -> dict[str, float | str | int]:
    frame_heights: list[float] = []
    frame_areas: list[float] = []
    persons_per_frame: list[float] = []
    person_confidences: list[float] = []
    keypoint_visibility: list[float] = []
    valid_frames = 0
    sampled_rows = rows.iloc[::block_stride]
    for path in sampled_rows['yolo_path']:
        with _load_model_result_array(path, allow_pickle=False) as cache:
            frame_mask = cache['frame_mask'].astype(bool)
            boxes = cache['boxes']
            valid = cache['valid'].astype(bool) & frame_mask[:, None]
            person_conf = cache['person_conf']
            keypoint_conf = cache['keypoint_conf']
            for frame_index in np.flatnonzero(frame_mask):
                selected = valid[frame_index]
                valid_frames += 1
                persons_per_frame.append(float(selected.sum()))
                if not selected.any():
                    continue
                current_boxes = boxes[frame_index, selected]
                heights = np.clip(current_boxes[:, 3] - current_boxes[:, 1], 0.0, 1.0)
                widths = np.clip(current_boxes[:, 2] - current_boxes[:, 0], 0.0, 1.0)
                frame_heights.append(float(np.median(heights)))
                frame_areas.append(float(np.median(heights * widths)))
                person_confidences.append(float(np.median(person_conf[frame_index, selected])))
                visible = keypoint_conf[frame_index, selected] >= 0.2
                keypoint_visibility.append(float(visible.mean()))
    return {'match': str(rows['group_id'].iloc[0]), 'blocks': int(len(rows)), 'sampled_blocks': int(len(sampled_rows)), 'sampled_frames': int(valid_frames), 'median_persons_per_frame': float(np.median(persons_per_frame)), 'median_box_height': float(np.median(frame_heights)), 'median_box_area': float(np.median(frame_areas)), 'median_person_confidence': float(np.median(person_confidences)), 'median_visible_keypoint_fraction': float(np.median(keypoint_visibility))}

def main() -> None:
    manifest = pd.read_csv(_resolve_model_artifact(MANIFEST))
    current = manifest[manifest['domain'].eq('CV')]
    rows = [summarize_match(group) for _, group in current.groupby('group_id', sort=True)]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
if __name__ == '__main__':
    main()
