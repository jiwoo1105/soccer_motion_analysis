"""September 2026 experimental metrics on the original source-frame clock.

No model loading, grade fitting, output-directory imports or cache mutation.
Body alignment is a projected 2D quantity, not a 3D rotation angle.
"""
from pathlib import Path
import unicodedata

import numpy as np
from scipy.signal import find_peaks, savgol_filter

VERSION = '2026-09-20-relative-alignment-roi'
DEFAULT_WEIGHTS = {'headup': 1 / 3, 'trunk': 1 / 3, 'body': 1 / 3}


def nfc(value):
    return unicodedata.normalize('NFC', str(value))


def runs(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def load_pose(path, source_frames):
    """Expand cached poses without deleting absent source frames."""
    if not isinstance(source_frames, int) or isinstance(source_frames, bool) or source_frames <= 0:
        raise ValueError('source_frames must be a positive integer from source video metadata')
    with np.load(Path(path), allow_pickle=False) as z:
        frames = z['frame_numbers'].copy()
        if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
            raise ValueError('frame_numbers must be a one-dimensional integer array')
        if len(frames) and (np.any(frames < 0) or np.any(frames >= source_frames) or np.any(np.diff(frames) <= 0)):
            raise ValueError('source frame numbers must be unique, ascending, and inside the video')
        width, height = float(z['frame_width']), float(z['frame_height'])
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError('frame dimensions must be finite and positive')
        pose = dict(n=source_frames, width=width, height=height, frame_numbers=frames)
        for dest, key, shape in [('lm', 'landmarks', (33, 3)), ('world', 'world_landmarks', (33, 3)), ('vis', 'visibility', (33,))]:
            values = np.asarray(z[key], float)
            if values.shape != (len(frames),) + shape:
                raise ValueError(f'Unexpected {key} shape: {values.shape}')
            arr = np.full((source_frames,) + shape, np.nan)
            arr[frames] = values
            pose[dest] = arr
    return pose


def body_alignment(pose, window=9):
    """P95-P5 of the normalized determinant, using common 13-frame support."""
    if window not in (5, 9, 13):
        raise ValueError('Supported body smoothing windows are 5, 9 and 13')
    xy = pose['lm'][:, :, :2] * [pose['width'], pose['height']]
    sh, pe = xy[:, 12] - xy[:, 11], xy[:, 24] - xy[:, 23]
    spine = (xy[:, 11] + xy[:, 12] - xy[:, 23] - xy[:, 24]) / 2
    length = np.linalg.norm(spine, axis=1)
    ids = [11, 12, 23, 24]
    valid = (pose['vis'][:, ids] >= .5).all(axis=1) & np.isfinite(xy[:, ids]).all(axis=(1, 2)) & (length > 1)
    raw = (sh[:, 0] * pe[:, 1] - sh[:, 1] * pe[:, 0]) / np.maximum(length ** 2, 1e-9)
    raw[~valid] = np.nan
    signal = np.full(pose['n'], np.nan)
    mask = np.zeros(pose['n'], bool)
    for ix in runs(valid):
        if len(ix) < 13:
            continue
        signal[ix] = savgol_filter(raw[ix], window, 2)
        mask[ix[6:-6]] = True
    values = signal[mask]
    # One sample cannot establish a variation range. Coverage/duration still
    # require review: this structural minimum is not a validated quality cutoff.
    measurement = float(np.percentile(values, 95) - np.percentile(values, 5)) if len(values) >= 2 else None
    return dict(M=measurement, signal=signal, mask=mask, valid_frames=int(mask.sum()), source_frames=pose['n'])


def trunk_angle(pose):
    """Mean left/right knee-hip-shoulder angle, retaining the existing formula."""
    world, vis = pose['world'], pose['vis']
    ids = [25, 23, 11, 26, 24, 12]
    good = (vis[:, ids] >= .5).all(axis=1) & np.isfinite(world[:, ids]).all(axis=(1, 2))
    angles = []
    for knee, hip, shoulder in [(25, 23, 11), (26, 24, 12)]:
        u, v = world[:, knee] - world[:, hip], world[:, shoulder] - world[:, hip]
        nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        good &= (nu > 0) & (nv > 0)
        u, v = u / (nu[:, None] + 1e-10), v / (nv[:, None] + 1e-10)
        angles.append(np.degrees(np.arccos(np.clip(np.sum(u * v, axis=1), -1, 1))))
    return float(np.mean(np.mean(angles, axis=0)[good])) if good.any() else None


def choose_ball(record):
    """Largest geometrically plausible candidate near the ankles; not a truth label."""
    if record.get('status') != 'observed':
        return None
    length = record['torso_length_px']
    ankles = np.array(record['ankles_global_xy'], float)
    if not np.isfinite(length) or not np.isfinite(ankles).all():
        return None
    good = []
    for det in record['detections']:
        center, radius, box = np.array(det['center_global_xy']), det['radius_px'], det['bbox_global_xyxy']
        if not np.isfinite([*center, radius, *box]).all():
            continue
        ratio = (box[2] - box[0]) / max(box[3] - box[1], 1)
        distance = np.min(np.linalg.norm(ankles - center, axis=1)) / max(length, 1)
        if .15 <= radius / max(length, 1) <= .50 and .5 <= ratio <= 2 and distance <= 2.5:
            good.append(det)
    if not good:
        return None
    det = max(good, key=lambda item: item['radius_px'])
    return [*det['center_global_xy'], det['radius_px']]


def analyze_headup(pose, candidates):
    """Head-angle range around filtered ball direction-change proxies (±8 frames)."""
    n, world, vis = pose['n'], pose['world'], pose['vis']
    if candidates['counts']['source_frames'] != n:
        raise ValueError('Candidate/source frame counts disagree')
    h = (world[:, 2] + world[:, 5] - world[:, 11] - world[:, 12]) / 2
    hn = h / (np.linalg.norm(h, axis=1)[:, None] + 1e-10)
    vertical = np.array([0., -1., 0.]) / (1. + 1e-10)
    head = np.degrees(np.arccos(np.clip(hn @ vertical, -1, 1)))
    head_ok = (vis[:, [2, 5, 11, 12]] >= .5).all(axis=1) & np.isfinite(head) & (np.linalg.norm(h, axis=1) > 0)
    xy = pose['lm'][:, :, :2] * [pose['width'], pose['height']]
    selected = np.full((n, 3), np.nan)
    seen = set()
    for rec in candidates['frames']:
        frame = rec['frame_number']
        if not isinstance(frame, int) or not 0 <= frame < n or frame in seen:
            raise ValueError('Invalid or duplicate candidate frame index')
        seen.add(frame)
        value = choose_ball(rec)
        if value is not None:
            selected[frame] = value
    observed = np.isfinite(selected).all(axis=1)
    track, imputed = selected.copy(), np.zeros(n, bool)
    jumps = np.flatnonzero(np.linalg.norm(np.diff(selected[:, :2], axis=0), axis=1) > 80) + 1
    for ix in runs(~observed):
        if len(ix) <= 3 and ix[0] > 0 and ix[-1] + 1 < n:
            left, right = ix[0] - 1, ix[-1] + 1
            if observed[left] and observed[right] and np.linalg.norm(track[right, :2] - track[left, :2]) / (right - left) <= 40:
                track[ix] = track[left] + ((ix - left) / (right - left))[:, None] * (track[right] - track[left])
                imputed[ix] = True
    smooth = np.full(n, np.nan)
    peaks_all = []
    for ix in runs(np.isfinite(track[:, 0])):
        if len(ix) < 11:
            continue
        smooth[ix] = savgol_filter(track[ix, 0], 11, 2)
        for sign in (1, -1):
            peaks, props = find_peaks(sign * smooth[ix], distance=8, prominence=10)
            peaks_all.extend(dict(frame=int(ix[j]), sign=sign, prominence=float(pr)) for j, pr in zip(peaks, props['prominences']))
    chosen = []
    for event in sorted(peaks_all, key=lambda item: -item['prominence']):
        if all(abs(event['frame'] - previous['frame']) >= 8 for previous in chosen):
            chosen.append(event)
    chosen.sort(key=lambda event: event['frame'])
    events, rejected = [], []
    for event in chosen:
        t, reasons = event['frame'], []
        lo, hi = t - 13, t + 14
        if lo < 0 or hi > n:
            reasons.append('filter_edge')
        else:
            if not np.isfinite(track[lo:hi]).all(): reasons.append('ball_support_gap')
            if observed[t-8:t+9].mean() < .8: reasons.append('observed_under80pct')
            if not head_ok[t-8:t+9].all(): reasons.append('head_pose_invalid')
            if any(lo <= j < hi for j in jumps): reasons.append('ball_jump_in_support')
        if not observed[t]: reasons.append('imputed_center')
        distances = np.linalg.norm(xy[t, [27, 28]] - track[t, :2], axis=1)
        foot = int(np.argmin(distances)) + 27
        if not np.isfinite(distances).all() or min(distances) > 200: reasons.append('far_from_ankle')
        if not np.isfinite(vis[t, foot]) or vis[t, foot] < .5: reasons.append('ankle_visibility')
        event.update(reasons=reasons, distance=float(min(distances)), foot=foot,
                     head_range=float(np.ptp(head[max(0, t-8):min(n, t+9)])))
        (rejected if reasons else events).append(event)
    return dict(headup=float(np.mean([event['head_range'] for event in events])) if len(events) >= 2 else None,
                events=events, rejected=rejected, candidate_count=len(chosen),
                observed_fraction=float(observed.mean()), imputed_fraction=float(imputed.mean()),
                jumps=jumps, head_angles=head, head_valid=head_ok, observed_ball=selected, ball=track, smooth_x=smooth)


def head_timing_sensitivity(head):
    """Compare ±2-frame shifts on the same events with valid support at all shifts."""
    angles, valid = head['head_angles'], head['head_valid']
    centers = [event['frame'] for event in head['events']
               if event['frame'] >= 10 and event['frame'] + 10 < len(angles)
               and valid[event['frame']-10:event['frame']+11].all()]
    means = [float(np.mean([np.ptp(angles[t+s-8:t+s+9]) for t in centers]))
             if len(centers) >= 2 else None for s in range(-2, 3)]
    return dict(shifts=list(range(-2, 3)), common_event_count=len(centers), means=means)


def score_measurements(row, calibration):
    def proximity(value, parameters):
        if value is None or not np.isfinite(value): return None
        opt, tau = parameters['opt'], parameters['tau']
        if not np.isfinite([opt, tau]).all() or tau <= 0:
            raise ValueError('Calibration opt/tau must be finite; tau must be positive')
        return float(np.clip(10 * (1 - abs(value - opt) / tau), 0, 10))
    ref = calibration['body_reference_mean']
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError('body_reference_mean must be finite and positive')
    body = row['body_M']
    scores = dict(head_score=proximity(row['head_raw'], calibration['headup']),
                  trunk_score=proximity(row['trunk_raw'], calibration['trunk']),
                  body_score=float(np.clip(10 * body / ref, 0, 10)) if body is not None and np.isfinite(body) else None)
    scores['total'] = sum(scores.values()) / 3 if all(v is not None for v in scores.values()) else None
    return scores
