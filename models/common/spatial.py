"""Resolution-independent spatial normalization for web video inference.

The initial training clips came from 1280x720 sources that were scaled by 0.5
to 640x360 and then center-cropped to 224x224.  Web inputs may use a different
encoded resolution while preserving the same camera framing.  A fixed 0.5
factor would therefore change the visible field of view.  This module instead
normalizes every source to the training reference height before applying the
fixed center crop.
"""

from __future__ import annotations

import math

import cv2


MODEL_INPUT_SIZE = 224
TRAINING_REFERENCE_SOURCE_WIDTH = 1280
TRAINING_REFERENCE_SOURCE_HEIGHT = 720
TRAINING_REFERENCE_WIDTH = 640
TRAINING_REFERENCE_HEIGHT = 360
SPATIAL_PREPROCESSING_VERSION = "reference_height_360_center224_v2"
YOLO_SOURCE_ROI_VERSION = "equivalent_source_roi_to_640_v2"


def _round_half_up(value: float) -> int:
    """Match JavaScript's positive ``Math.round`` for display geometry."""
    return int(math.floor(float(value) + 0.5))


def training_normalization_geometry(
    source_width: int,
    source_height: int,
    *,
    target_size: int = MODEL_INPUT_SIZE,
    reference_height: int = TRAINING_REFERENCE_HEIGHT,
) -> dict:
    """Return the exact resize and center-crop geometry for one source.

    Frames already at the requested model size are kept intact.
    Aspect ratio is preserved.  Landscape material is normalized to the
    intermediate height used by training (360 px).  For unusually narrow
    material, the scale is increased just enough to keep the 224 px crop
    available, which avoids padding or geometric distortion.
    """
    source_width = int(source_width)
    source_height = int(source_height)
    target_size = int(target_size)
    reference_height = int(reference_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    if target_size <= 0 or reference_height <= 0:
        raise ValueError("Target and reference dimensions must be positive")

    if (source_width, source_height) == (target_size, target_size):
        scale = 1.0
    else:
        scale = max(reference_height / source_height, target_size / source_width)
    resized_width = max(target_size, _round_half_up(source_width * scale))
    resized_height = max(target_size, _round_half_up(source_height * scale))
    crop_x = (resized_width - target_size) // 2
    crop_y = (resized_height - target_size) // 2

    return {
        "version": SPATIAL_PREPROCESSING_VERSION,
        "source_width": source_width,
        "source_height": source_height,
        "scale": float(scale),
        "resized_width": resized_width,
        "resized_height": resized_height,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_width": target_size,
        "crop_height": target_size,
        "source_roi_x": crop_x * source_width / resized_width,
        "source_roi_y": crop_y * source_height / resized_height,
        "source_roi_width": target_size * source_width / resized_width,
        "source_roi_height": target_size * source_height / resized_height,
    }


def normalize_training_frame(
    frame,
    *,
    target_size: int = MODEL_INPUT_SIZE,
    reference_height: int = TRAINING_REFERENCE_HEIGHT,
):
    """Resize one BGR/RGB frame and return the training-equivalent crop."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    source_height, source_width = frame.shape[:2]
    try:
        geometry = training_normalization_geometry(
            source_width,
            source_height,
            target_size=target_size,
            reference_height=reference_height,
        )
    except ValueError:
        return None

    resized = cv2.resize(
        frame,
        (geometry["resized_width"], geometry["resized_height"]),
        interpolation=cv2.INTER_LINEAR,
    )
    x0 = geometry["crop_x"]
    y0 = geometry["crop_y"]
    return resized[y0:y0 + target_size, x0:x0 + target_size]


def extract_training_equivalent_source_roi(
    frame,
    *,
    target_size: int = MODEL_INPUT_SIZE,
    reference_height: int = TRAINING_REFERENCE_HEIGHT,
):
    """Return the source-resolution ROI corresponding to the training crop.

    ``normalize_training_frame`` first resizes the complete source and then
    extracts a 224x224 crop.  That is appropriate for the RGB backbone, but it
    unnecessarily removes detail before a detector such as YOLO receives the
    same field of view.  This function maps the crop bounds back to source
    coordinates and crops there, before any spatial downsampling.

    The returned ROI therefore covers the same scene region as the RGB crop:
    448x448 for a 1280x720 source and 672x672 for a 1920x1080 source.
    """
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    source_height, source_width = frame.shape[:2]
    try:
        geometry = training_normalization_geometry(
            source_width,
            source_height,
            target_size=target_size,
            reference_height=reference_height,
        )
    except ValueError:
        return None

    # Map the half-open resized-image interval back to a half-open source
    # interval.  Rounding both boundaries preserves the crop centre and gives
    # the exact historical 448x448 / 672x672 mappings for 720p / 1080p.
    x0 = _round_half_up(geometry["source_roi_x"])
    y0 = _round_half_up(geometry["source_roi_y"])
    x1 = _round_half_up(geometry["source_roi_x"] + geometry["source_roi_width"])
    y1 = _round_half_up(geometry["source_roi_y"] + geometry["source_roi_height"])
    x0 = min(max(0, x0), source_width - 1)
    y0 = min(max(0, y0), source_height - 1)
    x1 = min(max(x0 + 1, x1), source_width)
    y1 = min(max(y0 + 1, y1), source_height)
    roi = frame[y0:y1, x0:x1]
    return roi if roi.size else None


def normalize_yolo_source_roi(
    frame,
    *,
    yolo_size: int = 640,
    target_size: int = MODEL_INPUT_SIZE,
    reference_height: int = TRAINING_REFERENCE_HEIGHT,
):
    """Extract the training-equivalent source ROI and resize it for YOLO."""
    if int(yolo_size) <= 0:
        raise ValueError("YOLO target size must be positive")
    roi = extract_training_equivalent_source_roi(
        frame,
        target_size=target_size,
        reference_height=reference_height,
    )
    if roi is None:
        return None
    interpolation = (
        cv2.INTER_AREA
        if roi.shape[0] > int(yolo_size) or roi.shape[1] > int(yolo_size)
        else cv2.INTER_LINEAR
    )
    return cv2.resize(roi, (int(yolo_size), int(yolo_size)), interpolation=interpolation)
