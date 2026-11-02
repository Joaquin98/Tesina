"""Verify that an optimized YOLO backend reproduces the PyTorch pose cache."""
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
KEYS = ('boxes', 'person_conf', 'keypoints', 'keypoint_conf')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate_files = sorted(_model_artifact_glob(args.candidate, '*.npz', recursive=False))
    common = [(args.reference / path.name, path) for path in candidate_files if _resolve_model_artifact(args.reference / path.name).exists()]
    if not common:
        raise RuntimeError('No common pose caches were found')
    maxima = {key: 0.0 for key in KEYS}
    valid_equal = True
    people_equal = True
    for reference_path, candidate_path in common:
        with _load_model_result_array(reference_path) as reference, _load_model_result_array(candidate_path) as candidate:
            reference_valid = reference['valid'].astype(bool)
            candidate_valid = candidate['valid'].astype(bool)
            valid_equal = valid_equal and bool(np.array_equal(reference_valid, candidate_valid))
            people_equal = people_equal and int(reference_valid.sum()) == int(candidate_valid.sum())
            for key in KEYS:
                maxima[key] = max(maxima[key], float(np.max(np.abs(reference[key].astype(np.float32) - candidate[key].astype(np.float32)))))
    passed = bool(valid_equal and people_equal and (maxima['boxes'] <= 0.0001) and (maxima['person_conf'] <= 0.0001) and (maxima['keypoints'] <= 0.0001) and (maxima['keypoint_conf'] <= 0.0001))
    result = {'clips': len(common), 'valid_masks_equal': valid_equal, 'person_counts_equal': people_equal, 'max_abs_difference': maxima, 'tolerance': 0.0001, 'passed': passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)
if __name__ == '__main__':
    main()
