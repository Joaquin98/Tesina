"""Descriptores de posiciones y posturas a partir de una caché de detecciones.

Los descriptores son deliberadamente compactos. La intención inicial es medir
si la geometría colectiva agrega señal al modelo RGB, no entrenar un estimador
de pose desde cero.
"""

from __future__ import annotations

import math

import numpy as np


POSITION_DIM = 31
POSE_DIM = 30


def _safe_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else 0.0


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty(0, dtype=np.float32)
    delta = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    return dist[np.triu_indices(len(points), k=1)].astype(np.float32)


def position_descriptor(boxes: np.ndarray, person_conf: np.ndarray, valid: np.ndarray, confidence_threshold: float = 0.25) -> tuple[np.ndarray, bool]:
    """Resume centros, dispersión y compactación de las personas visibles."""
    boxes = np.asarray(boxes, dtype=np.float32)
    person_conf = np.asarray(person_conf, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    keep = valid & (person_conf >= confidence_threshold)
    selected = boxes[keep]
    if len(selected) == 0:
        return np.zeros(POSITION_DIM, dtype=np.float32), False

    centers = np.column_stack(((selected[:, 0] + selected[:, 2]) * 0.5, (selected[:, 1] + selected[:, 3]) * 0.5))
    widths = np.clip(selected[:, 2] - selected[:, 0], 0.0, 1.0)
    heights = np.clip(selected[:, 3] - selected[:, 1], 0.0, 1.0)
    extent = np.ptp(centers, axis=0) if len(centers) > 1 else np.zeros(2, dtype=np.float32)
    dists = _pairwise_distances(centers)
    nearest = np.full(len(centers), np.nan, dtype=np.float32)
    if len(centers) > 1:
        matrix = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
        matrix[matrix == 0] = np.nan
        finite_rows = np.any(np.isfinite(matrix), axis=1)
        nearest[finite_rows] = np.nanmin(matrix[finite_rows], axis=1)
    nearest = nearest[np.isfinite(nearest)]

    centered = centers - centers.mean(axis=0, keepdims=True)
    if len(centers) >= 2:
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        principal = eigenvectors[:, order[0]]
        angle = math.atan2(float(principal[1]), float(principal[0]))
        ratio = float(eigenvalues[1] / (eigenvalues[0] + 1e-6))
    else:
        ratio, angle = 0.0, 0.0

    values = np.array(
        [
            len(selected) / 30.0,
            float(centers[:, 0].mean()), float(centers[:, 1].mean()),
            float(centers[:, 0].std()), float(centers[:, 1].std()),
            _safe_quantile(centers[:, 0], 0.25), _safe_quantile(centers[:, 0], 0.50), _safe_quantile(centers[:, 0], 0.75),
            _safe_quantile(centers[:, 1], 0.25), _safe_quantile(centers[:, 1], 0.50), _safe_quantile(centers[:, 1], 0.75),
            float(widths.mean()), float(widths.std()), float(heights.mean()), float(heights.std()),
            float(extent[0]), float(extent[1]),
            float(dists.mean()) if len(dists) else 0.0,
            float(dists.std()) if len(dists) else 0.0,
            _safe_quantile(dists, 0.25), _safe_quantile(dists, 0.50), _safe_quantile(dists, 0.75),
            float(nearest.mean()) if len(nearest) else 0.0,
            float(nearest.std()) if len(nearest) else 0.0,
            _safe_quantile(nearest, 0.50),
            ratio,
            float(math.sin(2.0 * angle)), float(math.cos(2.0 * angle)),
            float((centers[:, 0] < np.median(centers[:, 0])).mean()),
            float((widths / np.maximum(heights, 1e-4)).mean()),
            float(person_conf[keep].mean()),
        ],
        dtype=np.float32,
    )
    if values.shape != (POSITION_DIM,):
        raise AssertionError(f"Dimensión de posición inesperada: {values.shape}")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), True


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        return float("nan")
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-6:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(math.acos(cosine) / math.pi)


def pose_descriptor(keypoints: np.ndarray, keypoint_conf: np.ndarray, person_conf: np.ndarray, valid: np.ndarray, confidence_threshold: float = 0.20) -> tuple[np.ndarray, bool]:
    """Resume postura corporal sin depender de identidades persistentes."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    keypoint_conf = np.asarray(keypoint_conf, dtype=np.float32)
    person_conf = np.asarray(person_conf, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    keep = valid & (person_conf >= confidence_threshold)
    if not np.any(keep):
        return np.zeros(POSE_DIM, dtype=np.float32), False

    kp = keypoints[keep]
    kc = keypoint_conf[keep]
    usable = kc >= confidence_threshold
    person_features: list[list[float]] = []
    for joints, conf in zip(kp, kc):
        def point(index: int) -> np.ndarray:
            return joints[index] if conf[index] >= confidence_threshold else np.array([np.nan, np.nan], dtype=np.float32)

        nose, ls, rs = point(0), point(5), point(6)
        le, re, lw, rw = point(7), point(8), point(9), point(10)
        lh, rh = point(11), point(12)
        lk, rk, la, ra = point(13), point(14), point(15), point(16)
        shoulder = np.nanmean(np.stack([ls, rs]), axis=0) if np.isfinite(np.stack([ls, rs])).any() else np.array([np.nan, np.nan])
        hip = np.nanmean(np.stack([lh, rh]), axis=0) if np.isfinite(np.stack([lh, rh])).any() else np.array([np.nan, np.nan])
        torso = hip - shoulder
        torso_length = float(np.linalg.norm(torso)) if np.all(np.isfinite(torso)) else float("nan")
        torso_vertical = abs(float(torso[1])) / (torso_length + 1e-6) if np.isfinite(torso_length) else float("nan")
        knee_left = _joint_angle(lh, lk, la)
        knee_right = _joint_angle(rh, rk, ra)
        wrist_left_up = float(lw[1] < ls[1]) if np.all(np.isfinite(lw)) and np.all(np.isfinite(ls)) else float("nan")
        wrist_right_up = float(rw[1] < rs[1]) if np.all(np.isfinite(rw)) and np.all(np.isfinite(rs)) else float("nan")
        crouched = float(np.nanmean([lk[1] - lh[1], rk[1] - rh[1]]) < 0.10) if np.all(np.isfinite([lk, lh, rk, rh])) else float("nan")
        body_height = float(np.nanmax(joints[:, 1]) - np.nanmin(joints[:, 1])) if np.any(np.isfinite(joints)) else float("nan")
        wrist_height = float(np.nanmean([lw[1], rw[1]])) if np.all(np.isfinite([lw, rw])) else float("nan")
        hip_height = float(np.nanmean([lh[1], rh[1]])) if np.all(np.isfinite([lh, rh])) else float("nan")
        nose_above = float(nose[1] < shoulder[1]) if np.all(np.isfinite([nose, shoulder])) else float("nan")
        arm_raise = float(np.mean([wrist_left_up, wrist_right_up])) if np.isfinite([wrist_left_up, wrist_right_up]).any() else float("nan")
        person_features.append([
            float(np.nanmean(conf)), float(np.mean(usable)), torso_vertical, torso_length,
            knee_left, knee_right, wrist_left_up, wrist_right_up,
            arm_raise, crouched,
            body_height, wrist_height, hip_height, nose_above,
        ])

    matrix = np.asarray(person_features, dtype=np.float32)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    mean = np.divide(np.nansum(matrix, axis=0), counts, out=np.zeros(matrix.shape[1], dtype=np.float32), where=counts > 0)
    centered = np.where(finite, matrix - mean[None, :], 0.0)
    std = np.sqrt(np.divide(np.sum(centered * centered, axis=0), counts, out=np.zeros(matrix.shape[1], dtype=np.float32), where=counts > 0))
    rates = finite.mean(axis=0)
    values = np.concatenate([
        np.array([len(matrix) / 30.0], dtype=np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
        rates.astype(np.float32),
    ])
    # 1 + 14 * 3 = 43, but project to a compact fixed vector used by the model.
    selected = np.array([
        values[0], values[1], values[2], values[3], values[4],
        values[5], values[6], values[7], values[8], values[9],
        values[10], values[11], values[12], values[13], values[14],
        values[15], values[16], values[17], values[18], values[19],
        values[20], values[21], values[22], values[23], values[24],
        values[25], values[26], values[27], values[28], values[29],
    ], dtype=np.float32)
    if selected.shape != (POSE_DIM,):
        raise AssertionError(f"Dimensión de pose inesperada: {selected.shape}")
    return np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0), True


def build_geometry_features(boxes: np.ndarray, person_conf: np.ndarray, keypoints: np.ndarray, keypoint_conf: np.ndarray, valid: np.ndarray, person_threshold: float = 0.25, keypoint_threshold: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genera las tres matrices guardadas por ``pose_cache.py``."""
    positions = []
    poses = []
    frame_mask = []
    for frame_boxes, frame_person_conf, frame_keypoints, frame_keypoint_conf, frame_valid in zip(boxes, person_conf, keypoints, keypoint_conf, valid):
        position, position_ok = position_descriptor(frame_boxes, frame_person_conf, frame_valid, person_threshold)
        pose, pose_ok = pose_descriptor(frame_keypoints, frame_keypoint_conf, frame_person_conf, frame_valid, keypoint_threshold)
        positions.append(position)
        poses.append(pose)
        frame_mask.append(position_ok or pose_ok)
    return np.asarray(positions, dtype=np.float32), np.asarray(poses, dtype=np.float32), np.asarray(frame_mask, dtype=bool)
