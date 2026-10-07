"""Frozen provisional scores from source-frame pose and ball candidates.

Only NumPy/SciPy and the model-free historical head proxy helper are used.
No grades, video names, learned models, per-clip calibration, or report imports.
Missing samples are masked, never removed from the source clock. Diagnostics
contain NumPy arrays and NaNs; a report serializer must convert NaNs to null.
"""
from collections.abc import Mapping
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from .current_metrics import analyze_headup, load_pose, runs


_CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/scoring.json').read_text(encoding='utf-8'))
VERSION = _CONFIG['version']
DEFAULT_CALIBRATION = _CONFIG['calibration']
DEFAULT_WEIGHTS = _CONFIG['weights']
_KEYS = {'headup', 'trunk', 'shoulder'}


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _array(value, shape, name):
    try:
        result = np.asarray(value)
        if result.shape != shape or result.dtype.kind not in 'iuf':
            raise ValueError
        return result.astype(float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a numeric array of shape {shape}') from exc


def _pose(pose):
    if not isinstance(pose, Mapping) or not {'n', 'width', 'height', 'lm', 'world', 'vis'} <= pose.keys():
        raise ValueError('pose requires n, width, height, lm, world, and vis')
    n = _integer(pose['n'], 'pose.n', minimum=1)
    p = dict(n=n, width=_number(pose['width'], 'width', positive=True),
             height=_number(pose['height'], 'height', positive=True))
    for name, shape in (('lm', (n, 33, 3)), ('world', (n, 33, 3)), ('vis', (n, 33))):
        p[name] = _array(pose[name], shape, name)
        p[name][~np.isfinite(p[name])] = np.nan
    invalid_visibility = (p['vis'] < 0) | (p['vis'] > 1) | ~np.isfinite(p['vis'])
    p['vis'][invalid_visibility] = np.nan
    return p, invalid_visibility


def _exclusions(value, n):
    mask = np.zeros(n, dtype=bool)
    if value is None:
        return mask
    a = np.asarray(value)
    if a.ndim != 1:
        raise ValueError('excluded_frames must be source indices or an N-element boolean mask')
    if a.dtype.kind == 'b':
        if a.shape != (n,):
            raise ValueError('excluded_frames boolean mask must have N entries')
        return a.copy()
    if not a.size:
        return mask
    if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a >= n):
        raise ValueError('excluded_frames indices must be integers inside the source video')
    # Mixed bool/int lists must not turn True into source frame 1.
    if any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError('excluded_frames indices cannot be booleans')
    mask[a] = True
    return mask


def validate_parameters(calibration=None, weights=None):
    """Preflight anchors and weights without inference; return fresh (c, w).

    calibration accepts the complete JSON profile or its calibration field.
    Supply the profile's weights explicitly when using a custom profile.
    """
    calibration = validate_calibration(calibration)
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if not isinstance(weights, Mapping) or set(weights) != _KEYS:
        raise ValueError('weights must have exactly headup, trunk, shoulder keys')
    w = {k: _number(weights[k], f'weights.{k}') for k in ('headup', 'trunk', 'shoulder')}
    if any(v < 0 for v in w.values()) or not np.isclose(sum(w.values()), 1., rtol=0, atol=1e-9):
        raise ValueError('weights must be nonnegative and sum to 1; they are not normalized')
    return calibration, w


def validate_calibration(calibration=None):
    """Validate before inference; return a fresh three-metric calibration dict.

    Accept either the complete configs/scoring.json document or its
    ``calibration`` field. Weights are separately validated by measure_video.
    This only validates supplied anchors; it never learns or refits them.
    """
    calibration = DEFAULT_CALIBRATION if calibration is None else calibration
    if isinstance(calibration, Mapping) and 'calibration' in calibration:
        calibration = calibration['calibration']
    if not isinstance(calibration, Mapping) or set(calibration) != _KEYS:
        raise ValueError('calibration must have exactly headup, trunk, shoulder keys')
    fields = {'headup': ('min_deg', 'max_deg'), 'trunk': ('reference_flexion_deg',),
              'shoulder': ('reference_speed_deg_s',)}
    c = {}
    for key, expected in fields.items():
        if not isinstance(calibration[key], Mapping) or set(calibration[key]) != set(expected):
            raise ValueError(f'calibration.{key} requires {expected}')
        c[key] = {field: _number(calibration[key][field], f'calibration.{key}.{field}') for field in expected}
    if not 0 <= c['headup']['min_deg'] < c['headup']['max_deg'] <= 180:
        raise ValueError('headup anchors require 0 <= min_deg < max_deg <= 180')
    if not 0 < c['trunk']['reference_flexion_deg'] <= 180:
        raise ValueError('trunk reference_flexion_deg must be in (0, 180]')
    if c['shoulder']['reference_speed_deg_s'] <= 0:
        raise ValueError('shoulder reference_speed_deg_s must be positive')
    return c


def _candidates(candidates, n):
    """Validate record shapes; nonfinite detections remain ordinary missing data."""
    if candidates is None:
        return {'counts': {'source_frames': n}, 'frames': []}, np.zeros(n, bool)
    if not isinstance(candidates, Mapping) or not isinstance(candidates.get('counts'), Mapping):
        raise ValueError('candidates requires counts.source_frames and frames')
    if _integer(candidates['counts'].get('source_frames'), 'candidate source_frames', minimum=1) != n:
        raise ValueError('Candidate/source frame counts disagree')
    if not isinstance(candidates.get('frames'), (list, tuple)):
        raise ValueError('candidate frames must be a list')
    records, seen, invalid = [], set(), np.zeros(n, bool)
    for record in candidates['frames']:
        if not isinstance(record, Mapping) or not isinstance(record.get('status'), str):
            raise ValueError('candidate record requires frame_number and status')
        frame = _integer(record.get('frame_number'), 'candidate frame_number')
        if frame >= n or frame in seen:
            raise ValueError('Invalid or duplicate candidate frame index')
        seen.add(frame)
        rec = dict(record, frame_number=frame)
        if rec['status'] == 'observed':
            length = record.get('torso_length_px')
            if isinstance(length, (bool, np.bool_)) or not isinstance(length, Real):
                raise ValueError('candidate torso_length_px must be numeric')
            ankles = _array(record.get('ankles_global_xy'), (2, 2), 'ankles_global_xy')
            if not isinstance(record.get('detections'), (list, tuple)):
                raise ValueError('candidate detections must be a list')
            detections = []
            for det in record['detections']:
                if not isinstance(det, Mapping):
                    raise ValueError('each detection must be a mapping')
                center = _array(det.get('center_global_xy'), (2,), 'center_global_xy')
                box = _array(det.get('bbox_global_xyxy'), (4,), 'bbox_global_xyxy')
                radius = det.get('radius_px')
                if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
                    raise ValueError('radius_px must be numeric')
                if (not np.isfinite([*center, *box, radius]).all() or radius <= 0
                        or box[2] <= box[0] or box[3] <= box[1]):
                    invalid[frame] = True
                    continue
                detections.append(dict(center_global_xy=center, bbox_global_xyxy=box, radius_px=float(radius)))
            rec.update(torso_length_px=float(length), ankles_global_xy=ankles, detections=detections)
            if not np.isfinite(length) or length <= 0 or not np.isfinite(ankles).all():
                rec['status'] = 'invalid_geometry'
                invalid[frame] = True
        records.append(rec)
    return {'counts': {'source_frames': n}, 'frames': records}, invalid


def _unit(vector):
    length = np.linalg.norm(vector, axis=-1)
    # Match the native-mask reference's zero-vector convention: a length
    # collapse still has a finite step, so its >1.5 ratio removes BOTH ends.
    return np.divide(vector, np.maximum(length[..., None], 1e-12),
                     out=np.full_like(vector, np.nan), where=np.isfinite(length[..., None]))


def _angle(a, b):
    # atan2(cross, dot) is stable at rest; acos(dot) amplifies roundoff near 1.
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


def _head(pose, candidates):
    old = analyze_headup(pose, candidates)
    events, rejected, support = [], [], np.zeros(pose['n'], bool)
    for event in old['events']:
        t = event['frame']
        lo, hi, reasons = t - 16, t + 16, []
        if lo < 0 or hi >= pose['n']:
            reasons.append('source_edge')
        elif not old['head_valid'][lo:hi + 1].all():
            reasons.append('head_pose_invalid')
        entry = dict(event, start=lo, end=hi, reasons=reasons)
        if reasons:
            # The old ±8 range is not a valid measurement of this ±16 window.
            entry.update(head_range=None, range_deg=None)
            rejected.append(entry)
        else:
            value = float(np.ptp(old['head_angles'][lo:hi + 1]))
            entry.update(head_range=value, range_deg=value)
            events.append(entry)
            support[lo:hi + 1] = True
    raw = float(np.mean([e['range_deg'] for e in events])) if len(events) >= 2 else None
    observed = np.isfinite(old['observed_ball']).all(axis=1)
    imputed = ~observed & np.isfinite(old['ball']).all(axis=1)
    diagnostics = dict(old, headup=raw, events=events, legacy_events=old['events'],
                       legacy_rejected=old['rejected'], rejected_expanded=rejected,
                       rejected=old['rejected'] + rejected, observed=observed, imputed=imputed,
                       support=support, window_halfwidth_frames=16,
                       counts=dict(candidate_events=old['candidate_count'], legacy_events=len(old['events']),
                                   valid_events=len(events), rejected_legacy_events=len(old['rejected']),
                                   rejected_expanded_events=len(rejected), observed_frames=int(observed.sum()),
                                   imputed_frames=int(imputed.sum()), jump_transitions=len(old['jumps'])))
    reason = None if raw is not None else f'Need at least 2 accepted ball proxies with complete ±16 head support; found {len(events)}.'
    return dict(raw=raw, unit='deg', reason=reason, valid_events=len(events),
                valid_frames=int(support.sum()), valid_samples=len(events),
                candidate_events=old['candidate_count']), diagnostics


def _trunk(pose):
    w, vis = pose['world'], pose['vis']
    ids = [25, 23, 11, 26, 24, 12]
    finite = np.isfinite(w[:, ids]).all(axis=(1, 2))
    visible = (vis[:, ids] >= .5).all(axis=1)
    nondegenerate = np.ones(pose['n'], bool)
    sides = []
    for knee, hip, shoulder in ((25, 23, 11), (26, 24, 12)):
        u, v = w[:, knee] - w[:, hip], w[:, shoulder] - w[:, hip]
        nondegenerate &= (np.linalg.norm(u, axis=1) > 0) & (np.linalg.norm(v, axis=1) > 0)
        sides.append(_angle(_unit(u), _unit(v)))
    bilateral = np.column_stack(sides)
    valid = finite & visible & nondegenerate & np.isfinite(bilateral).all(axis=1)
    theta = bilateral.mean(axis=1)
    theta[~valid] = np.nan
    raw = float(theta[valid].mean()) if valid.any() else None
    count = int(valid.sum())
    metric = dict(raw=raw, unit='deg', reason=None if raw is not None else 'No valid bilateral knee–hip–shoulder frames.',
                  valid_frames=count, valid_samples=count)
    return metric, dict(valid=valid, finite=finite, visible=visible, nondegenerate=nondegenerate,
                        angles=theta, bilateral_angles=bilateral, flexion=180 - theta,
                        counts=dict(valid_frames=count, invalid_frames=pose['n'] - count))


def _shoulder(pose, excluded):
    """Native shoulder mask and common SG9/13/21 support; measure SG13 XYZ."""
    world, lm, vis = pose['world'][:, [11, 12]], pose['lm'][:, [11, 12]], pose['vis'][:, [11, 12]]
    finite = np.isfinite(world).all(axis=(1, 2)) & np.isfinite(lm).all(axis=(1, 2))
    visible = (vis >= .5).all(axis=1)
    inside = ((lm[:, :, :2] >= 0) & (lm[:, :, :2] <= 1)).all(axis=(1, 2))
    vector = world[:, 1] - world[:, 0]
    length = np.linalg.norm(vector, axis=1)
    geometry = (length > 1e-5) & (np.linalg.norm(vector[:, [0, 2]], axis=1) > .05 * length)
    base_valid = finite & visible & inside & geometry
    unit = _unit(vector)
    step = _angle(unit[:-1], unit[1:])
    ratio = np.maximum(length[:-1] / np.maximum(length[1:], 1e-12),
                       length[1:] / np.maximum(length[:-1], 1e-12))
    flagged = np.isfinite(step) & ((step > 45) | (ratio > 1.5))
    auto = np.zeros(pose['n'], bool)
    bad = np.flatnonzero(flagged)
    auto[bad] = True
    auto[bad + 1] = True
    native = base_valid & ~auto
    valid = native & ~excluded
    smooth_unit = np.full_like(unit, np.nan)
    norms = {window: np.full(pose['n'], np.nan) for window in (9, 13, 21)}
    support, trimmed = np.zeros(pose['n'], bool), np.zeros(pose['n'], bool)
    run_lengths = []
    for ix in runs(valid):
        run_lengths.append(len(ix))
        if len(ix) < 24:
            continue
        common = np.ones(len(ix), bool)
        for window in (9, 13, 21):
            filtered = savgol_filter(unit[ix], window, 2, axis=0)
            norms[window][ix] = np.linalg.norm(filtered, axis=1)
            common &= norms[window][ix] > .2
            if window == 13:
                smooth_unit[ix] = _unit(filtered)
        trimmed[ix[11:-11]] = True
        support[ix[11:-11]] = common[11:-11]
    pair = np.r_[False, support[:-1] & support[1:]]
    return dict(vector=vector, length=length, raw_unit=unit, unit=smooth_unit,
                finite=finite, visible=visible, inside=inside, geometry=geometry,
                base_valid=base_valid, native_valid=native, valid=valid, auto_excluded=auto,
                excluded=excluded.copy(), step_deg=step, length_ratio=ratio,
                flagged_transitions=bad, smooth_norms=norms, trimmed_support=trimmed,
                support=support, pair_support=pair, run_lengths=run_lengths)


def measure_video(pose, candidates, fps, calibration=None, weights=None, excluded_frames=None):
    """Return ``metrics``, ``total``, ``weights``, ``quality``, and diagnostics.

    pose: n (positive source-frame count), width/height, lm/world (N,33,3),
    vis (N,33). Nonfinite coordinates and vis outside [0,1] are missing data.
    candidates: redetect_balls counts/frames schema, or None for no ball data.
    calibration: full config document or its three calibration mappings; no fitting.
    weights: dictionary with headup/trunk/shoulder, nonnegative, summing to 1.
    excluded_frames: source-frame integer indices or an N-element bool mask,
    applied to all metrics; no video aliases or historical exclusions are used.

    Head raw is the mean ±16 event range in degrees (at least two events).
    Trunk raw is the bilateral angle theta, not 180-theta. Shoulder raw is mean
    geodesic deg/s over adjacent supported source frames. Any missing component
    makes total None, even if its weight is zero. Valid scores have reason=None.
    """
    fps = _number(fps, 'fps', positive=True)
    calibration, weights = validate_parameters(calibration, weights)
    p, invalid_visibility = _pose(pose)
    excluded = _exclusions(excluded_frames, p['n'])
    candidates, invalid_candidates = _candidates(candidates, p['n'])
    # Native shoulder diagnostics precede explicit exclusions; other measures
    # receive excluded visibility without mutating the caller's source arrays.
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        shoulder = _shoulder(p, excluded)
        p['vis'][excluded] = np.nan
        head_metric, head = _head(p, candidates)
        trunk_metric, trunk = _trunk(p)
        speed = np.r_[np.nan, _angle(shoulder['unit'][:-1], shoulder['unit'][1:]) * fps]
    shoulder['pair_support'] &= np.isfinite(speed)
    speed[~shoulder['pair_support']] = np.nan
    shoulder['geo_speed'] = speed
    count = int(shoulder['pair_support'].sum())
    raw = float(speed[shoulder['pair_support']].mean()) if count else None
    shoulder['counts'] = dict(native_frames=int(shoulder['native_valid'].sum()),
                              valid_frames=int(shoulder['valid'].sum()),
                              auto_excluded_frames=int(shoulder['auto_excluded'].sum()),
                              flagged_transitions=len(shoulder['flagged_transitions']),
                              excluded_frames=int(excluded.sum()),
                              supported_frames=int(shoulder['support'].sum()), valid_pairs=count)
    shoulder_metric = dict(raw=raw, unit='deg/s', valid_frames=int(shoulder['support'].sum()),
                           valid_pairs=count, valid_samples=count,
                           reason=None if raw is not None else 'No adjacent supported shoulder frames: need a valid run of at least 24 frames, 11-frame end guards, and common SG9/13/21 support.')
    head['invalid_candidate_geometry'] = invalid_candidates
    head['counts']['invalid_candidate_geometry_frames'] = int(invalid_candidates.sum())
    metrics = dict(headup=head_metric, trunk=trunk_metric, shoulder=shoulder_metric)
    for key, metric in metrics.items():
        value = metric['raw']
        score = None
        if value is not None:
            if key == 'headup':
                low, high = calibration[key]['min_deg'], calibration[key]['max_deg']
                score = 3 + 7 * np.clip((value - low) / (high - low), 0, 1)
            elif key == 'trunk':
                score = np.clip(10 * (180 - value) / calibration[key]['reference_flexion_deg'], 0, 10)
            else:
                score = np.clip(10 * value / calibration[key]['reference_speed_deg_s'], 0, 10)
        metric['score'] = float(score) if score is not None else None
        metric['source_frames'] = p['n']
    total = sum(weights[k] * m['score'] for k, m in metrics.items()) if all(m['score'] is not None for m in metrics.values()) else None
    warnings = [
        'Provisional frozen relative scores; not validated skill grades. No grade training or per-video refitting.',
        'Automatic numeric masks do not validate subject identity or left/right landmark identity; inspect the source video.',
        f'Head ball proxies retain fixed pixel thresholds (80 px jumps, 40 px/frame interpolation, 200 px ankle distance, 10 px prominence); source width {p["width"]:g} px and camera framing can change acceptance.',
        'Head ranges include upward and downward motion; ball direction changes are proxies, not confirmed contacts or gaze.',
    ]
    if excluded.any():
        warnings.append(f'{int(excluded.sum())} explicitly excluded source frames were masked for all metrics.')
    for key, metric in metrics.items():
        if metric['reason']:
            warnings.append(f'{key}: {metric["reason"]}')
    return dict(version=VERSION, metrics=metrics, total=total, weights=weights, calibration=calibration,
                quality=dict(warnings=warnings, source_frames=p['n'], excluded_frames=int(excluded.sum()),
                             invalid_visibility_values=int(invalid_visibility.sum()),
                             missing_metrics=[k for k, m in metrics.items() if m['raw'] is None]),
                diagnostics=dict(headup=head, trunk=trunk, shoulder=shoulder,
                                 excluded_frames=excluded, invalid_visibility=invalid_visibility))
