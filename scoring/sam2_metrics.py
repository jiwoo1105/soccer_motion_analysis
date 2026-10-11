"""Original YOLOv8/SAM2 turning-point detector, without inference imports.

The legacy detector smooths the available SAM2 observations (SG11, degree2),
then restores source indices. It therefore compresses ball gaps while finding
peaks; this limitation is reported, not silently presented as contact GT.
Shoulder smoothing/speeds themselves never compress the source clock.
"""
from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def normalize_tracks(tracks, n):
    if not isinstance(tracks, Mapping):
        raise ValueError('ball_tracks must map source frame indices to (cx, cy, radius)')
    result = np.full((n, 3), np.nan)
    seen = set()
    for key, value in tracks.items():
        if isinstance(key, str) and key.isdecimal() and str(int(key)) == key:
            frame = int(key)
        elif isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)):
            frame = int(key)
        else:
            raise ValueError('SAM2 frame keys must be canonical source-frame integers')
        if frame < 0 or frame >= n or frame in seen:
            raise ValueError('SAM2 frame index outside source or duplicated')
        seen.add(frame)
        a = np.asarray(value)
        if a.shape != (3,) or a.dtype.kind not in 'iuf':
            raise ValueError('SAM2 observation must be numeric (cx, cy, radius)')
        if np.isfinite(a).all():
            if a[2] <= 0:
                raise ValueError('SAM2 radius must be positive')
            result[frame] = a
    return result


def detect_sam2_touches(pose, tracks):
    """Retain the original extrema + nearest-ankle <=200px rule.

    Peak distance8 applies to each polarity in the observed sequence, exactly
    as in the original detector; opposite-polarity events may be closer.
    Head visibility is deliberately not a shoulder event-selection criterion.
    """
    ball = normalize_tracks(tracks, pose['n'])
    observed = np.isfinite(ball).all(axis=1)
    frames = np.flatnonzero(observed)
    smooth = np.full(pose['n'], np.nan)
    accepted, rejected = [], []
    if len(frames) >= 10:
        width = min(11, len(frames) if len(frames)%2 else len(frames)-1)
        values = savgol_filter(ball[frames, 0], width, 2)
        smooth[frames] = values
        padded = np.r_[values[0], values, values[-1]]
        indices = set()
        for sign in (1, -1):
            ix = find_peaks(sign*padded, distance=8, prominence=10)[0]-1
            indices.update(int(i) for i in ix if 0 <= i < len(frames))
        for i in sorted(indices):
            frame = int(frames[i])
            xy = pose['lm'][frame, [27,28], :2] * [pose['width'],pose['height']]
            distance = np.linalg.norm(xy-ball[frame,:2],axis=1)
            why = []
            if not np.isfinite(distance).all():
                why.append('ankle_pose_missing')
            elif distance.min() > 200:
                why.append('far_from_ankle')
            event = dict(frame=frame, reasons=why, distance=float(distance.min()) if np.isfinite(distance).all() else None,
                         foot=int(np.argmin(distance))+27 if np.isfinite(distance).all() else None)
            (rejected if why else accepted).append(event)
    return dict(events=accepted,rejected=rejected,ball=ball,observed_ball=ball.copy(),
                observed=observed,smooth_x=smooth,candidate_count=len(accepted)+len(rejected),
                source='yolov8_sam2',missing_ball_frames=int((~observed).sum()),
                gap_policy='legacy extrema use compacted observed ball positions; source indices restored')


def head_from_sam2(pose, detected):
    """Original head angle definition, legacy complete ±8 gate, then ±16 range.

    Keeping the legacy ball/pose support gate makes its effect explicit.
    Expanded head support must be complete; no neighboring eye samples are
    invented. At least two accepted windows are required for the video mean.
    """
    w = pose['world'];n=pose['n']
    h=(w[:,2]+w[:,5]-w[:,11]-w[:,12])/2
    length=np.linalg.norm(h,axis=1)
    hn=h/(length[:,None]+1e-10)
    angles=np.degrees(np.arccos(np.clip(hn@(np.array([0.,-1.,0.])/(1+1e-10)),-1,1)))
    valid=(pose['vis'][:,[2,5,11,12]]>=.5).all(axis=1)&np.isfinite(angles)&(length>0)
    accepted,rejected=[],list(detected['rejected'])
    support=np.zeros(n,bool)
    for e in detected['events']:
        t=e['frame'];lo,hi=t-16,t+16;why=[]
        if lo<0 or hi>=n:
            why.append('source_edge')
        else:
            if not valid[lo:hi+1].all():why.append('head_pose_invalid')
            if not detected['observed'][t-8:t+9].all():why.append('ball_support_gap')
        event=dict(e,start=lo,end=hi,reasons=why,range_deg=None,head_range=None)
        if why:rejected.append(event)
        else:
            event['range_deg']=event['head_range']=float(np.ptp(angles[lo:hi+1]))
            accepted.append(event);support[lo:hi+1]=True
    raw=float(np.mean([e['range_deg'] for e in accepted])) if len(accepted)>=2 else None
    diag=dict(detected,head_angles=angles,head_valid=valid,headup=raw,events=accepted,
              rejected=rejected,support=support,imputed=np.zeros(n,bool),jumps=[],
              legacy_events=detected['events'],legacy_rejected=detected['rejected'],
              window_halfwidth_frames=16,event_source='yolov8_sam2',
              counts=dict(candidate_events=detected['candidate_count'],valid_events=len(accepted),
                          observed_frames=int(detected['observed'].sum()),imputed_frames=0,
                          jump_transitions=0,rejected_events=len(rejected)))
    reason=None if raw is not None else f'Need at least 2 SAM2 touch candidates with complete ±16 head and legacy ±8 ball support; found {len(accepted)}.'
    metric=dict(raw=raw,unit='deg',reason=reason,valid_events=len(accepted),valid_frames=int(support.sum()),
                valid_samples=len(accepted),candidate_events=detected['candidate_count'])
    return metric,diag
