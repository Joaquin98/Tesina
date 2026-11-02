"""Full VF_S01 pose-only inference, including fresh YOLO, timed end to end."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import runtime as rt
from runtime import HERE, ROOT, save, sha
import json, sys, time, csv
from datetime import datetime, timezone
sys.path.insert(0, str(ROOT))

def main():
    started = time.perf_counter()
    import cv2, numpy as np, tensorflow as tf, torch
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from pose_adapter import PoseOnlyDetector, validate_geometry_extraction
    from src.web.keras_inference_engine import KerasInferenceEngine
    from models.M0.web_adapter import preprocess_raw_frame
    imports = time.perf_counter() - started
    evaluation = json.loads(_resolve_model_artifact(HERE / 'results/evaluation.json').read_text(encoding='utf-8'))
    manifest = json.loads(_resolve_model_artifact(HERE / 'inputs/source_manifest.json').read_text(encoding='utf-8'))
    thresholds = evaluation['video_thresholds_median_development_folds']
    t = time.perf_counter()
    detector = PoseOnlyDetector(manifest, thresholds)
    load = time.perf_counter() - t
    save('results/adapter_validation.json', validate_geometry_extraction(detector))
    assert not hasattr(detector, 'feature_extractor') and (not hasattr(detector, 'sequence_model'))
    assert all(('dual' not in member for member in detector.fold_models))
    video = ROOT / 'data/videos/partidos_evaluacion/VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    assert ok
    detector._begin_auxiliary_sequence(1)
    t = time.perf_counter()
    detector._on_sampled_frame(preprocess_raw_frame(frame, target_size=224))
    detector._finish_auxiliary_sequence()
    detector._predict_sequence_window(None, None, 0, 1)
    warm = time.perf_counter() - t
    references = [video, ROOT / manifest['pose_model'], _resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py'), _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), _resolve_model_artifact(ROOT / 'models/M1/geometry_features.py'), _resolve_model_artifact(ROOT / 'models/common/spatial.py')]
    references.extend((ROOT / f['geometry_rf'] for f in manifest['folds']))
    hashes = {str(p): sha(p) for p in references}
    save('results/startup.json', {'imports_seconds': imports, 'model_load_seconds': load, 'warmup_seconds': warm, 'thresholds': thresholds, 'hashes_before': hashes, 'ResNet_loaded': False, 'dual_BiGRU_loaded': False, 'threads': {'tf_intra': 8, 'tf_inter': 1, 'torch_intra': 8, 'torch_inter': 1, 'OpenCV': 1, 'RF_n_jobs': -1}})
    engine = KerasInferenceEngine()
    engine.current_model = detector
    engine.current_model_name = detector.display_name
    engine.current_model_type = 'm1_geometry_only_experimental'
    detector.stats.clear()
    detector.counts.clear()
    last = [0.0]
    start = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()

    def progress(percent, stage, message):
        now = time.perf_counter()
        if now - last[0] >= 20 or percent in [18, 72, 96]:
            row = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': now - start, 'percent': percent, 'stage': stage, 'message': message, 'counts': dict(detector.counts)}
            with _resolve_model_artifact(HERE / 'logs/progress.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')
            (HERE / 'progress.json').write_text(json.dumps(row), encoding='utf-8')
            print('PROGRESS ' + json.dumps(row), flush=True)
            last[0] = now
    result = engine.analyze_video(video, confidence_threshold=thresholds['scrum'], line_confidence_threshold=thresholds['line'], use_segments=True, segment_duration=7, temporal_fps_mode='training_match_25', feature_batch_size=256, feature_predict_batch_size=8, progress_callback=progress)
    elapsed = time.perf_counter() - start
    end_utc = datetime.now(timezone.utc).isoformat()
    assert result['status'] == 'success', result
    policy = result['event_post_processing']
    policy.update(threshold_mode='per_class_pre_merge', per_class_min_target_confidence=thresholds, source='geometry_branch_development_only_medians', variant='geometry_RF_only')
    counts = dict(detector.counts)
    counts.update(ResNet_calls=0, dual_BiGRU_calls=0)
    timing = {'inference_seconds': elapsed, 'inference_start_utc': start_utc, 'inference_end_utc': end_utc, 'video_seconds': result['metadata']['duration'], 'projection_80min_seconds': elapsed * 4800 / result['metadata']['duration'], 'imports_seconds': imports, 'model_load_seconds': load, 'warmup_seconds': warm, 'stage_seconds': dict(detector.stats), 'counts': counts, 'includes': 'decode, same spatial crop and 25FPS sampling, fresh YOLO, geometry, six forests, voting, event policy, result formatting', 'no_persistent_video_feature_reuse': True, 'placeholder_rows_are_ignored_by_classifier': True}
    result['result_id'] = 'M1_geometry_only_VF_S01_20261012'
    result['metadata']['timing'] = timing
    result['metadata']['event_post_processing'] = policy
    assert counts['YOLO_calls'] == counts['placeholder_rows'] == 30030
    assert counts['window_calls'] == len(result['window_predictions']) == 1195 and counts['forest_calls'] == 7170
    assert all((np.all(np.isfinite(list(w['probabilities'].values()))) and abs(sum(w['probabilities'].values()) - 1) < 1e-05 for w in result['window_predictions']))
    after = {str(p): sha(p) for p in references}
    assert after == hashes
    save('results/result.json', result)
    save('results/timing.json', timing)
    for label, rows in [('events', result['detections']), ('windows', result['window_predictions'])]:
        with _resolve_model_artifact(HERE / f'results/{label}.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['class', 'start_seconds', 'end_seconds', 'confidence', 'p_line', 'p_other', 'p_scrum'])
            for row in rows:
                writer.writerow([row['play_type'], row['start_time'], row['end_time'], row['confidence'], *[row['probabilities'][c] for c in ['line', 'other', 'scrum']]])
    save('results/completion.json', {'all_checks_pass': True, 'hashes_unchanged': True, 'hashes_after': after, 'ResNet_calls': 0, 'dual_BiGRU_calls': 0, 'real_YOLO_calls': 30030, 'windows': 1195, 'no_retraining': True, 'no_original_files_modified': True})
    print('COMPLETE ' + json.dumps({'inference_seconds': elapsed, 'events': len(result['detections'])}), flush=True)
if __name__ == '__main__':
    main()
