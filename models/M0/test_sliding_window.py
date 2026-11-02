"""
Quick test script to verify sliding window analysis works correctly.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys
from pathlib import Path
sys.path.insert(0, str(source_directory(Path(__file__).parent)))
from models.M0.web_adapter import RugbyPlayDetector

def test_sliding_window(video_path, confidence_threshold=0.3):
    """Test the sliding window analysis."""
    print('=' * 70)
    print('Testing Sliding Window Analysis')
    print('=' * 70)
    print('\n1. Loading model...')
    detector = RugbyPlayDetector()
    print(f'\n2. Analyzing video: {video_path}')
    print(f'   Confidence threshold: {confidence_threshold}')
    print(f'   Window size: 7 seconds')
    print(f'   Stride: 1 second')
    print('\n' + '=' * 70)
    results = detector.analyze_video_segments(video_path, segment_duration=7, confidence_threshold=confidence_threshold)
    print('\n' + '=' * 70)
    print('FINAL RESULTS')
    print('=' * 70)
    print(f"Status: {results['status']}")
    print(f"Video duration: {results['duration']:.1f}s")
    print(f"Window duration: {results.get('window_duration', 'N/A')}s")
    print(f"Stride duration: {results.get('stride_duration', 'N/A')}s")
    print(f"Total windows processed: {results.get('total_windows', 'N/A')}")
    print(f"\nDetections found: {len(results['detections'])}")
    if results['detections']:
        print('\n' + '-' * 70)
        print('Detailed Detections:')
        print('-' * 70)
        for i, det in enumerate(results['detections'], 1):
            print(f"\n  [{i}] {det['play_type'].upper()}")
            print(f"      Time: {det['start_time']:.1f}s - {det['end_time']:.1f}s (duration: {det['duration']:.1f}s)")
            print(f"      Frames: {det['frame_start']} - {det['frame_end']}")
            print(f"      Confidence: {det['confidence']:.2%}")
            print(f'      Probabilities:')
            for label, prob in sorted(det['probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 30)
                print(f'        {label:8s}: {prob:6.2%} {bar}')
    else:
        print('\n  ⚠️  No detections above confidence threshold.')
        print(f'\n  Suggestions:')
        print(f'    - Try lowering threshold: python {Path(__file__).name} {video_path} 0.2')
        print(f'    - Check if video contains the expected play types')
    print('\n' + '=' * 70)
    return results
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_sliding_window.py <video_path> [confidence_threshold]')
        print('\nExample:')
        print('  python test_sliding_window.py ../../data/uploads/test_video.mp4 0.3')
        sys.exit(1)
    video_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    test_sliding_window(video_path, confidence)
