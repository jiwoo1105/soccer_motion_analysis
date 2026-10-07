"""Source-frame pose loading and ball direction-change head proxies.

Model-free helpers shared by cache replay and release scoring. The head helper
retains its original ±8 acceptance window; release_metrics expands accepted
events to ±16 for the current measurement.
"""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter


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
