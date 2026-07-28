# -*- coding: utf-8 -*-
"""터치 감지 진단 — 왜 터치가 적게 잡히는가

기준1(239프레임)에서 터치가 2개만 잡혀 헤드업·비율이 N/A가 됐다.
어느 단계에서 걸러지는지 확인한다:

    SAM2 추적 프레임 수  →  방향전환 후보 수  →  발 근접 통과 수  →  윈도우 유효 수

실행:
    python3.9 -m scoring.diagnose_touches                 # 캐시된 영상 전부
    python3.9 -m scoring.diagnose_touches --plot          # 공 X 궤적 그래프도 저장
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import savgol_filter, find_peaks

from scoring.extraction_cache import load, is_cached, VIDEOS
from scoring.touch_filter import filter_touches


def turn_candidates(ball_tracks, prominence=10, distance=8):
    """공 X 방향 전환 후보 (발 근접 조건 적용 전)"""
    frames = sorted(ball_tracks.keys())
    if len(frames) < 10:
        return np.array([]), np.array([]), np.array([])

    frames_arr = np.array(frames)
    ball_x = np.array([ball_tracks[f][0] for f in frames])

    win = min(11, len(ball_x) if len(ball_x) % 2 == 1 else len(ball_x) - 1)
    smooth = savgol_filter(ball_x, win, 2) if win >= 3 else ball_x

    padded = np.concatenate([[smooth[0]], smooth, [smooth[-1]]])
    peaks, _ = find_peaks(padded, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-padded, distance=distance, prominence=prominence)
    idx = np.array(sorted(set(list(peaks - 1) + list(troughs - 1))))
    idx = idx[(idx >= 0) & (idx < len(smooth))]

    return frames_arr, smooth, idx


def foot_distances(pose_frames, ball_tracks, frames_arr, turn_idx):
    """각 방향전환 지점에서 공-발 최소 거리 (px)"""
    frame_map = {pf.frame_number: pf for pf in pose_frames}
    out = []
    for ti in turn_idx:
        fn = int(frames_arr[ti])
        if fn not in frame_map:
            out.append((fn, None))
            continue
        pf = frame_map[fn]
        cx, cy, _ = ball_tracks[fn]
        fw, fh = pf.frame_width, pf.frame_height
        lm = pf.landmarks
        d = min(
            np.hypot(cx - lm[27][0] * fw, cy - lm[27][1] * fh),
            np.hypot(cx - lm[28][0] * fw, cy - lm[28][1] * fh),
        )
        out.append((fn, float(d)))
    return out


def diagnose(stem, prominence=10, max_foot_dist=200):
    from extract_depth_metrics import detect_touches_by_ball_direction

    pose_frames, ball_tracks = load(stem)
    n_pose = len(pose_frames)
    n_ball = len(ball_tracks)

    frames_arr, smooth, turn_idx = turn_candidates(ball_tracks, prominence=prominence)
    dists = foot_distances(pose_frames, ball_tracks, frames_arr, turn_idx)
    near = [(f, d) for f, d in dists if d is not None and d <= max_foot_dist]

    touches = detect_touches_by_ball_direction(
        pose_frames, ball_tracks, max_foot_dist=max_foot_dist)
    valid, reasons = filter_touches(touches, pose_frames, ball_tracks)

    ball_coverage = n_ball / n_pose if n_pose else 0
    print(f"\n[{stem}]")
    print(f"  포즈 {n_pose}프레임 / 공추적 {n_ball}프레임 (커버리지 {ball_coverage:.0%})")
    print(f"  방향전환 후보 {len(turn_idx)}개 → 발근접(≤{max_foot_dist}px) {len(near)}개 "
          f"→ 터치 {len(touches)}개 → 윈도우 유효 {len(valid)}개")
    if dists:
        ds = [d for _, d in dists if d is not None]
        print(f"  전환지점 공-발 거리: min {min(ds):.0f} / 중앙 {np.median(ds):.0f} "
              f"/ max {max(ds):.0f} px")
        print(f"    상세: " + ', '.join(
            f"f{f}={d:.0f}" + ('' if d <= max_foot_dist else '✗') for f, d in dists))
    for fn, why in reasons.items():
        print(f"    윈도우 탈락 f{fn}: {why}")

    return {
        'video': stem, 'n_pose': n_pose, 'n_ball': n_ball,
        'coverage': ball_coverage, 'n_turns': len(turn_idx),
        'n_near': len(near), 'n_touch': len(touches), 'n_valid': len(valid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prominence', type=float, default=10)
    ap.add_argument('--max-foot-dist', type=float, default=200)
    args = ap.parse_args()

    stems = [Path(v).stem for v in VIDEOS if is_cached(Path(v).stem)]
    if not stems:
        print('캐시된 영상이 없다. 먼저 scoring.extraction_cache를 돌려라.')
        return

    rows = [diagnose(s, args.prominence, args.max_foot_dist) for s in stems]

    print(f"\n{'='*76}")
    print(f"{'영상':<14}{'포즈':>6}{'공':>6}{'커버':>7}{'전환':>6}{'발근접':>7}{'터치':>6}{'유효':>6}")
    print('-' * 76)
    for r in rows:
        print(f"{r['video']:<14}{r['n_pose']:>6}{r['n_ball']:>6}{r['coverage']:>6.0%}"
              f"{r['n_turns']:>6}{r['n_near']:>7}{r['n_touch']:>6}{r['n_valid']:>6}")


if __name__ == '__main__':
    main()
