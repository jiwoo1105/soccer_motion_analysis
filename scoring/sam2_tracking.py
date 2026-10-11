"""Local-weight YOLOv8 prompt + raw SAM2 mask tracking from sam2_ball_tracker.py.

Preserves its initial scan, lower-half/ankle gates, centroid/radius definition,
and five-frame loss/reinitialization policy. The research cache producer
extract_depth_metrics.run_sam2 supplies ankles from all valid source frames.
No model package is imported until
tracking is requested; this module never downloads weights.
"""
from pathlib import Path
import tempfile

import numpy as np


SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
PARAMETERS = dict(ball_class=32, confidence=.3, search_frames=20,
                  initial_scan_limit=120, max_ankle_dist=250., lost_run=5,
                  max_reinit=4, ankle_source_frames='all_valid_pose_frames', sam2_config=SAM2_CONFIG,
                  mask_threshold=0., smoothing='none')


def ankle_positions(pose):
    """Match the research cache producer's all-frame ankle coordinates in pixels."""
    result = {}
    for frame in range(pose['n']):
        xy = pose['lm'][frame, [27, 28], :2] * [pose['width'], pose['height']]
        if np.isfinite(xy).all():
            result[frame] = xy.tolist()
    return result


def _ankle_dist(frame, cx, cy, ankles):
    points = ankles.get(frame)
    if not points:
        return None
    return min(np.hypot(cx - x, cy - y) for x, y in points)


def _scan_for_ball(model, directory, total, ankles, start, end, device, max_ankle_dist=250.):
    import cv2
    best, best_dist = None, float('inf')
    for frame in range(max(0, start), min(end, total)):
        image = cv2.imread(str(Path(directory) / f'{frame:06d}.jpg'))
        if image is None:
            raise ValueError(f'Cannot read extracted source frame {frame}')
        h, w = image.shape[:2]
        for result in model(image, conf=.3, verbose=False, device=device):
            for box in result.boxes:
                if int(box.cls[0]) != 32:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cy < h * .5:
                    continue
                radius = ((x2 - x1) + (y2 - y1)) / 4
                distance = _ankle_dist(frame, cx, cy, ankles)
                if distance is None:
                    distance = np.hypot(cx - w / 2, cy - h * .8)
                elif distance > max_ankle_dist:
                    continue
                if distance < best_dist:
                    best_dist, best = distance, (frame, cx, cy, radius)
        if best is not None and best_dist < max_ankle_dist / 2:
            break
    return best


def _extract_frames(video, directory, pose):
    import cv2
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open source video for SAM2')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (pose['height'], pose['width']):
                raise ValueError('SAM2 decode dimensions differ from source pose')
            if not cv2.imwrite(str(Path(directory) / f'{count:06d}.jpg'), frame):
                raise OSError('Cannot write temporary SAM2 frame')
            count += 1
        if count != pose['n']:
            raise ValueError(f'Incomplete SAM2 decode: {count} of {pose["n"]} source frames')
    finally:
        capture.release()
    return count


def track_ball(video, pose, model, checkpoint, device='cpu'):
    """Return {original source index: (cx, cy, area-equivalent radius)}.

    ``model`` is the caller's YOLO instance loaded from a verified local file.
    SAM2 must already be installed and checkpoint must exist locally. Missing
    masks stay absent, with no interpolation, smoothing, or ROI fallback.
    Temporary JPEGs are removed on both success and failure.
    """
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != '.pt':
        raise FileNotFoundError(f'Local SAM2 checkpoint missing: {checkpoint}')
    checkpoint = checkpoint.resolve()
    device = f'cuda:{device}' if str(device).isdigit() else str(device)
    ankles = ankle_positions(pose)
    with tempfile.TemporaryDirectory(prefix='sam2-frames-') as directory:
        total = _extract_frames(video, directory, pose)
        limit = min(total, PARAMETERS['initial_scan_limit'])
        window, initial = PARAMETERS['search_frames'], None
        while window <= limit and initial is None:
            initial = _scan_for_ball(model, directory, total, ankles, 0, window, device)
            window *= 2
        if initial is None:
            # Original fallback retries the first <=120 frames without the gate.
            initial = _scan_for_ball(model, directory, total, ankles, 0, limit, device, float('inf'))
        if initial is None:
            return {}
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CONFIG, str(checkpoint), device=device)
        tracks = {}
        frame, cx, cy, _ = initial
        with torch.inference_mode():
            state = predictor.init_state(video_path=directory)
            for attempt in range(PARAMETERS['max_reinit'] + 1):
                predictor.reset_state(state)
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=frame, obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32), labels=np.array([1], dtype=np.int32))
                segment, bad_run, lost_at = {}, 0, None
                for index, ids, masks in predictor.propagate_in_video(state, start_frame_idx=frame):
                    for object_id, mask in zip(ids, masks):
                        if object_id != 1:
                            continue
                        pixels = mask[0].cpu().numpy() > 0.
                        area = int(pixels.sum())
                        if not area:
                            continue
                        ys, xs = np.where(pixels)
                        mx, my = float(xs.mean()), float(ys.mean())
                        distance = _ankle_dist(index, mx, my, ankles)
                        if distance is not None and distance > PARAMETERS['max_ankle_dist']:
                            bad_run += 1
                            if bad_run >= PARAMETERS['lost_run']:
                                lost_at = index - bad_run + 1
                        else:
                            bad_run = 0
                        segment[int(index)] = (mx, my, float(np.sqrt(area / np.pi)))
                    if lost_at is not None:
                        break
                if lost_at is not None:
                    segment = {index: value for index, value in segment.items() if index < lost_at}
                tracks.update(segment)
                if lost_at is None or attempt == PARAMETERS['max_reinit']:
                    break
                found = _scan_for_ball(model, directory, total, ankles, lost_at + 1, total, device)
                if found is None:
                    break
                frame, cx, cy, _ = found
        return tracks
