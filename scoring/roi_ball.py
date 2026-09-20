"""NumPy-only geometry for the measured ankle-following ball detector.

Landmarks use normalized MediaPipe image coordinates, not world coordinates.
The ROI uses shoulders 11/12, hips 23/24 and ankles 27/28. Visibility is
reported, never thresholded. No tracking, interpolation or candidate selection
is performed here.
"""

import numpy as np


PARAMETERS = {
    'class_id': 32, 'imgsz': 640, 'conf': 0.10, 'iou': 0.7,
    'rect': False, 'half': False,
    'horizontal_margin_torso_lengths': 2.0,
    'vertical_margin_torso_lengths': 1.0,
    'roi_rule': 'Axis-aligned ankle bounding box expanded horizontally/vertically '
                'by the specified torso-length multipliers, clipped to image. '
                'Torso length = distance(mid shoulders11/12, mid hips23/24). '
                'Base margin is max(24px, torso length). No ball-track input.',
    'missing_pose_rule': 'Empty candidate list with missing_pose status; no temporal imputation.',
    'invalid_roi_rule': 'Malformed or nonfinite pose/visibility/timestamp, or an empty '
                        'clipped ROI, produces invalid_roi; no full-frame fallback.',
    'radius_rule': '(bbox_width+bbox_height)/4 in source pixels',
    'retention': 'All class32 candidates >=0.10 surviving standard YOLO NMS, no geometric selection.',
}


def roi_for(pose_landmarks, visibility, frame_number, frame_shape, timestamp=0.0):
    """Return a JSON-ready frame record; crop only records with status ``pending``.

    ``frame_shape`` is (height, width[, channels]); ``roi_xyxy`` uses exclusive
    right/bottom pixel bounds. Missing pose rows return ``missing_pose``;
    malformed/nonfinite inputs and empty clipped boxes return ``invalid_roi``.
    Invalid records never authorize inference, even when ROI bounds are present.
    """
    record = {
        'frame_number': int(frame_number), 'pose_timestamp_s': None,
        'roi_xyxy': None, 'torso_length_px': None, 'ankles_global_xy': None,
        'ankle_visibility': None, 'torso_visibility': None,
        'status': 'missing_pose', 'detections': [],
    }
    if pose_landmarks is None or visibility is None:
        return record
    record['status'] = 'invalid_roi'
    try:
        lm = np.asarray(pose_landmarks, dtype=float)
        vis = np.asarray(visibility, dtype=float)
        shape = np.asarray(frame_shape[:2], dtype=float)
        ts = float(timestamp)
    except (TypeError, ValueError, OverflowError):
        record['invalid_roi_reason'] = 'Malformed pose, visibility, timestamp or frame shape'
        return record
    if (lm.ndim != 2 or lm.shape[0] < 29 or lm.shape[1] < 2
            or vis.ndim != 1 or vis.shape[0] != lm.shape[0]
            or shape.shape != (2,) or not np.isfinite(shape).all()
            or (shape <= 0).any() or (shape != np.floor(shape)).any()
            or not np.isfinite(lm).all() or not np.isfinite(vis).all()
            or not np.isfinite(ts)):
        record['invalid_roi_reason'] = 'Malformed or nonfinite pose, visibility, timestamp or frame shape'
        return record
    height, width = map(int, shape)
    with np.errstate(over='ignore', invalid='ignore'):
        xy = lm[:, :2] * [width, height]
        length = float(np.linalg.norm(xy[[11, 12]].mean(axis=0)
                                      - xy[[23, 24]].mean(axis=0)))
        margin = max(24.0, length)
        ankles = xy[[27, 28]]
        horizontal = margin * PARAMETERS['horizontal_margin_torso_lengths']
        vertical = margin * PARAMETERS['vertical_margin_torso_lengths']
        bounds = np.array([
            ankles[:, 0].min() - horizontal, ankles[:, 1].min() - vertical,
            ankles[:, 0].max() + horizontal, ankles[:, 1].max() + vertical,
        ])
    if not np.isfinite(bounds).all() or not np.isfinite(length):
        record['invalid_roi_reason'] = 'Nonfinite pixel geometry'
        return record
    x1 = int(np.clip(np.floor(bounds[0]), 0, width))
    y1 = int(np.clip(np.floor(bounds[1]), 0, height))
    x2 = int(np.clip(np.ceil(bounds[2]), 0, width))
    y2 = int(np.clip(np.ceil(bounds[3]), 0, height))
    record.update({
        'pose_timestamp_s': ts, 'roi_xyxy': [x1, y1, x2, y2],
        'torso_length_px': length, 'ankles_global_xy': ankles.tolist(),
        'ankle_visibility': vis[[27, 28]].tolist(),
        'torso_visibility': vis[[11, 12, 23, 24]].tolist(),
    })
    if x2 <= x1 or y2 <= y1:
        record['invalid_roi_reason'] = 'Empty ROI after clipping to the image'
    else:
        record['status'] = 'pending'
    return record


def remap_detections(boxes_xyxy, confidences, class_ids, roi_xyxy):
    """Translate every post-NMS crop box to source pixels, preserving its order.

    No scale conversion is needed: YOLO's result boxes are already expressed in
    the original crop pixels. Radius is the legacy mean half-width/half-height.
    Invalid detector outputs raise instead of writing NaN or silently dropping
    candidates. Input arrays are not modified.
    """
    boxes = np.asarray(boxes_xyxy, dtype=float)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    confidence = np.asarray(confidences, dtype=float)
    classes = np.asarray(class_ids, dtype=float)
    roi = np.asarray(roi_xyxy, dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4
            or confidence.shape != (len(boxes),) or classes.shape != (len(boxes),)
            or roi.shape != (4,) or not np.isfinite(roi).all()
            or not np.isfinite(boxes).all() or not np.isfinite(confidence).all()
            or not np.isfinite(classes).all() or (classes != np.floor(classes)).any()
            or (boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError('Malformed or nonfinite detector output/ROI')
    ox, oy = roi[:2]
    detections = []
    for box, conf, cls in zip(boxes, confidence, classes):
        x1, y1, x2, y2 = map(float, box)
        x1 += float(ox)
        x2 += float(ox)
        y1 += float(oy)
        y2 += float(oy)
        detections.append({
            'class_id': int(cls), 'confidence': float(conf),
            'bbox_global_xyxy': [x1, y1, x2, y2],
            'center_global_xy': [(x1 + x2) / 2, (y1 + y2) / 2],
            'radius_px': (x2 - x1 + y2 - y1) / 4,
        })
    return detections
