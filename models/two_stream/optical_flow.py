"""Optical flow computation utilities for Two-Stream Network."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import cv2
import numpy as np
from pathlib import Path
import logging
from config import IMG_SIZE, NUM_FLOW_FRAMES, FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA, FLOW_MAX, FLOW_NORMALIZE
logger = logging.getLogger(__name__)

def crop_center_square(frame):
    """Crop the center square from a frame."""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def compute_optical_flow(video_path, num_flow_frames=NUM_FLOW_FRAMES, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Compute optical flow from a video using Farneback method.
    
    Args:
        video_path: Path to video file
        num_flow_frames: Number of optical flow frames to extract
        target_size: Spatial dimensions (H, W)
    
    Returns:
        flow_stack: Stacked optical flow, shape (num_flow_frames, H, W, 2)
                   Values normalized to [-1, 1]
        rgb_frame: Representative RGB frame, shape (H, W, 3)
                  Values in [0, 1]
    
    Raises:
        FileNotFoundError: If video file not found
        RuntimeError: If video cannot be read or no frames extracted
    """
    video_path = Path(video_path)
    if not _resolve_model_artifact(video_path).exists():
        raise FileNotFoundError(f'Video file not found: {video_path}')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_flow_frames + 1:
        logger.warning(f'Video has {total_frames} frames, less than {num_flow_frames + 1} needed')
    frame_indices = np.linspace(0, total_frames - 1, num_flow_frames + 1, dtype=int)
    flows = []
    rgb_frame = None
    prev_gray = None
    try:
        for idx_in_sequence, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f'Cannot read frame at index {frame_idx}')
                if len(flows) > 0:
                    flows.append(flows[-1])
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, target_size)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=FLOW_PYR_SCALE, levels=FLOW_LEVELS, winsize=FLOW_WINSIZE, iterations=FLOW_ITERATIONS, poly_n=FLOW_POLY_N, poly_sigma=FLOW_POLY_SIGMA, flags=0)
                flows.append(flow)
            if idx_in_sequence == len(frame_indices) // 2:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            prev_gray = gray
    finally:
        cap.release()
    if len(flows) == 0:
        raise RuntimeError(f'No optical flow frames extracted from {video_path}')
    while len(flows) < num_flow_frames:
        flows.append(flows[-1])
    flows = flows[:num_flow_frames]
    flow_stack = np.array(flows, dtype=np.float32)
    if FLOW_NORMALIZE:
        flow_stack = np.clip(flow_stack, -FLOW_MAX, FLOW_MAX) / FLOW_MAX
    return (flow_stack, rgb_frame)

def flow_to_hsv_image(flow, max_magnitude=None):
    """
    Convert optical flow to HSV image for visualization.
    
    Args:
        flow: Optical flow field, shape (H, W, 2)
        max_magnitude: Maximum magnitude for normalization (if None, auto-detect)
    
    Returns:
        rgb_image: RGB image representation of flow, shape (H, W, 3), values in [0, 255]
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if max_magnitude is None:
        max_magnitude = np.max(mag)
    if max_magnitude > 0:
        mag = (mag / max_magnitude * 255).astype(np.uint8)
    else:
        mag = np.zeros_like(mag, dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = mag
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def visualize_flow_stack(flow_stack, save_path=None):
    """
    Visualize a stack of optical flow frames.
    
    Args:
        flow_stack: Stacked flows, shape (num_frames, H, W, 2)
        save_path: Optional path to save visualization
    
    Returns:
        grid_image: Grid of flow visualizations
    """
    num_frames = flow_stack.shape[0]
    h, w = flow_stack.shape[1:3]
    cols = min(5, num_frames)
    rows = (num_frames + cols - 1) // cols
    grid_h = rows * h
    grid_w = cols * w
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for idx, flow in enumerate(flow_stack):
        row = idx // cols
        col = idx % cols
        flow_rgb = flow_to_hsv_image(flow)
        grid_image[row * h:(row + 1) * h, col * w:(col + 1) * w] = flow_rgb
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        logger.info(f'Flow visualization saved to {save_path}')
    return grid_image

def compute_optical_flow_batch(video_paths, num_flow_frames=NUM_FLOW_FRAMES):
    """
    Compute optical flow for multiple videos.
    
    Args:
        video_paths: List of video file paths
        num_flow_frames: Number of flow frames to extract per video
    
    Yields:
        Tuple of (flow_stack, rgb_frame, video_path, error)
    """
    for video_path in video_paths:
        try:
            flow_stack, rgb_frame = compute_optical_flow(video_path, num_flow_frames)
            yield (flow_stack, rgb_frame, video_path, None)
        except Exception as e:
            logger.error(f'Error processing {video_path}: {e}')
            yield (None, None, video_path, str(e))

def verify_flow_computation(video_path, num_flow_frames=NUM_FLOW_FRAMES, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Verify optical flow computation for a single video.
    Useful for debugging.
    
    Args:
        video_path: Path to video file
        num_flow_frames: Number of flow frames
        target_size: Target size
    
    Returns:
        Dictionary with computation details
    """
    try:
        flow_stack, rgb_frame = compute_optical_flow(video_path, num_flow_frames, target_size)
        return {'success': True, 'video_path': str(video_path), 'flow_shape': flow_stack.shape, 'flow_dtype': str(flow_stack.dtype), 'flow_min': float(flow_stack.min()), 'flow_max': float(flow_stack.max()), 'flow_mean': float(flow_stack.mean()), 'rgb_shape': rgb_frame.shape if rgb_frame is not None else None, 'rgb_dtype': str(rgb_frame.dtype) if rgb_frame is not None else None, 'rgb_min': float(rgb_frame.min()) if rgb_frame is not None else None, 'rgb_max': float(rgb_frame.max()) if rgb_frame is not None else None}
    except Exception as e:
        return {'success': False, 'video_path': str(video_path), 'error': str(e)}
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    video_path = 'path/to/test_video.mp4'
    try:
        flow_stack, rgb_frame = compute_optical_flow(video_path, num_flow_frames=10)
        print(f'Flow shape: {flow_stack.shape}')
        print(f'Flow range: [{flow_stack.min():.3f}, {flow_stack.max():.3f}]')
        print(f'RGB shape: {rgb_frame.shape}')
        print(f'RGB range: [{rgb_frame.min():.3f}, {rgb_frame.max():.3f}]')
    except Exception as e:
        print(f'Error: {e}')
