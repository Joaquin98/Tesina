import numpy as np

def decode_result(result, frame, max_people):
    boxes = np.zeros((max_people, 4), dtype=np.float32)
    person_conf = np.zeros(max_people, dtype=np.float32)
    keypoints = np.zeros((max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((max_people, 17), dtype=np.float32)
    valid = np.zeros(max_people, dtype=bool)
    if result.boxes is None or len(result.boxes) == 0:
        return (boxes, person_conf, keypoints, keypoint_conf, valid)
    raw_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    raw_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    raw_keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.xy is not None else None
    raw_keypoint_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.conf is not None else None
    order = np.argsort(raw_scores)[::-1][:max_people]
    width, height = (max(float(frame.shape[1]), 1.0), max(float(frame.shape[0]), 1.0))
    for out_i, src_i in enumerate(order):
        x1, y1, x2, y2 = raw_boxes[src_i]
        boxes[out_i] = np.clip([x1 / width, y1 / height, x2 / width, y2 / height], 0.0, 1.0)
        person_conf[out_i] = raw_scores[src_i]
        valid[out_i] = True
        if raw_keypoints is not None and src_i < len(raw_keypoints):
            keypoints[out_i] = np.clip(raw_keypoints[src_i] / np.array([width, height], dtype=np.float32), 0.0, 1.0)
        if raw_keypoint_conf is not None and src_i < len(raw_keypoint_conf):
            keypoint_conf[out_i] = raw_keypoint_conf[src_i]
    return (boxes, person_conf, keypoints, keypoint_conf, valid)
