# -*- coding: utf-8 -*-
"""세 지표의 원시값 계산 (캐시된 포즈·공추적에서)

    헤드업     : 유효 터치 ±8f, head angle mean_range (world_landmarks Y축)
    상체각도   : 전체 프레임, 무릎-엉덩이-어깨 각 평균 (world_landmarks)
    어깨-골반  : 유효 터치 ±8f, 어깨진폭 / 골반진폭 비율 (2D 폭)

포즈 추출과 SAM2 공추적은 `scoring.extraction_cache`가 미리 해둔다. 이 모듈은
캐시만 읽으므로 초 단위로 반복 실행할 수 있다.

실행:
    python3.9 -m scoring.extraction_cache     # 먼저 (영상당 약 10분)
    python3.9 -m scoring.raw_metrics
"""
import os
import sys
import re
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.head_pose_analyzer import HeadPoseAnalyzer
from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
from analysis.ball_motion_analyzer import BallMotionData
from scoring import rotation_signal as rs
from scoring.extraction_cache import load, is_cached, VIDEOS
from scoring.touch_filter import filter_touches, touch_quality, WINDOW

MAX_FOOT_DIST = 200


def score_tier(stem):
    """파일명에서 사람이 매긴 점수대를 읽는다. 기준영상은 None"""
    if '기준' in stem:
        return None
    m = re.search(r'(\d+)-\d+', stem)
    return int(m.group(1)) if m else None


def _rotation_metrics(pose_frames, valid_touch_frames):
    """2D 폭 기반 어깨/골반 진폭과 비율

    평활·드리프트 제거는 영상 전체에서 하고, 극값 선택만 터치 윈도우로 제한한다.
    짧은 구간만 잘라서 savgol을 걸면 경계에서 가짜 스윙이 생기기 때문이다.
    """
    frames = [pf.frame_number for pf in pose_frames]
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    n = len(frames)

    sh_w = np.array([abs(pf.landmarks[12][0] - pf.landmarks[11][0]) * pf.frame_width
                     for pf in pose_frames])
    pe_w = np.array([abs(pf.landmarks[24][0] - pf.landmarks[23][0]) * pf.frame_width
                     for pf in pose_frames])

    sh_ang, sh_bad, sh_run = rs.clean_angle(rs.width_to_angle(sh_w))
    pe_ang, pe_bad, pe_run = rs.clean_angle(rs.width_to_angle(pe_w))

    centers = [frame_to_idx[f] for f in valid_touch_frames if f in frame_to_idx]
    allowed = rs.touch_window_indices(centers, n, WINDOW) if centers else None

    sh_amp, sh_ext = rs.amplitude(sh_ang, allowed)
    pe_amp, pe_ext = rs.amplitude(pe_ang, allowed)
    ratio = sh_amp / pe_amp if (sh_amp and pe_amp and pe_amp > 0) else None

    # 참고용 — 터치 윈도우 제한 없이 영상 전체에서 잰 값
    sh_amp_all, _ = rs.amplitude(sh_ang)
    pe_amp_all, _ = rs.amplitude(pe_ang)
    ratio_all = (sh_amp_all / pe_amp_all
                 if (sh_amp_all and pe_amp_all and pe_amp_all > 0) else None)

    return {
        'shoulder_amp': sh_amp,
        'pelvis_amp': pe_amp,
        'ratio': ratio,
        'shoulder_amp_all': sh_amp_all,
        'pelvis_amp_all': pe_amp_all,
        'ratio_all': ratio_all,
        'shoulder_spikes': int(sh_bad.sum()),
        'pelvis_spikes': int(pe_bad.sum()),
        'shoulder_max_run': int(sh_run),
        'pelvis_max_run': int(pe_run),
        'cross_corr': rs.max_cross_correlation(sh_ang, pe_ang),
        'shoulder_extrema': len(sh_ext),
        'pelvis_extrema': len(pe_ext),
    }


def compute(stem, max_foot_dist=MAX_FOOT_DIST):
    """캐시된 영상 1편 → 원시값 dict"""
    from extract_depth_metrics import detect_touches_by_ball_direction

    pose_frames, ball_tracks = load(stem)
    touches = detect_touches_by_ball_direction(
        pose_frames, ball_tracks, max_foot_dist=max_foot_dist)
    valid, reasons = filter_touches(touches, pose_frames, ball_tracks)

    trunk_data = TrunkPoseAnalyzer().analyze(pose_frames)
    trunk = float(trunk_data.mean_angle) if trunk_data else None

    # 헤드업은 터치별 range의 평균이라 유효 터치가 2개는 있어야 한다
    headup = None
    if len(valid) >= 2:
        ball_data = BallMotionData(
            frame_numbers=np.array([pf.frame_number for pf in pose_frames]),
            positions=np.array([[0, 0]] * len(pose_frames)),
            touch_frames=[t.frame_number for t in valid],
            touch_count=len(valid),
            touch_events=valid,
        )
        head_data = HeadPoseAnalyzer(min_visibility_threshold=0.5).analyze(
            pose_frames, ball_data)
        if head_data and head_data.touch_window_mean_range is not None:
            headup = float(head_data.touch_window_mean_range)

    rot = _rotation_metrics(pose_frames, [t.frame_number for t in valid])

    return {
        'video': stem,
        'tier': score_tier(stem),
        'frames': len(pose_frames),
        'ball_frames': len(ball_tracks),
        'touch_total': len(touches),
        'touch_valid': len(valid),
        'touch_quality': touch_quality(len(valid), len(touches)),
        'touch_frames': [t.frame_number for t in valid],
        'drop_reasons': {str(k): v for k, v in reasons.items()},
        'headup': headup,
        'trunk': trunk,
        **rot,
    }


def load_all(max_foot_dist=MAX_FOOT_DIST):
    """캐시된 영상 전부의 원시값"""
    out = []
    for v in VIDEOS:
        stem = Path(v).stem
        if not is_cached(stem):
            continue
        out.append(compute(stem, max_foot_dist))
    return out


def print_table(results):
    f = lambda v, d=2: (f'{v:.{d}f}' if v is not None else 'N/A')
    print(f"\n{'='*92}")
    print(f"{'영상':<14}{'점수대':>5}{'헤드업':>9}{'상체각':>9}"
          f"{'어깨진폭':>9}{'골반진폭':>9}{'비율':>8}{'터치':>9}{'교차상관':>9}")
    print('-' * 92)
    for r in sorted(results, key=lambda r: (r['tier'] is None, r['tier'] or 0)):
        print(f"{r['video']:<14}{r['tier'] if r['tier'] else '기준':>5}"
              f"{f(r['headup'],1):>9}{f(r['trunk'],1):>9}"
              f"{f(r['shoulder_amp'],1):>9}{f(r['pelvis_amp'],1):>9}"
              f"{f(r['ratio']):>8}"
              f"{str(r['touch_valid'])+'/'+str(r['touch_total']):>9}"
              f"{f(r['cross_corr']):>9}")


def phase1_gate(results):
    """어깨-골반 교차상관 평균 — 0.9 이상이면 몸통 분리 미관측"""
    corrs = [r['cross_corr'] for r in results if r['cross_corr'] is not None]
    if not corrs:
        return None
    mean_corr = float(np.mean(corrs))
    print(f"\n[Phase 1 게이트] 어깨-골반 교차상관 평균 = {mean_corr:.3f} (n={len(corrs)})")
    for r in sorted(results, key=lambda r: -r['cross_corr']):
        print(f"    {r['video']:<14}{r['cross_corr']:.3f}")
    if mean_corr >= 0.9:
        print('  ⚠ 0.9 이상 — 이 동작에서 몸통 분리가 관측되지 않음. 지표 재검토 필요.')
    else:
        print('  0.9 미만 — Phase 2 진행 가능')
    return mean_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-foot-dist', type=float, default=MAX_FOOT_DIST)
    args = ap.parse_args()

    results = load_all(args.max_foot_dist)
    if not results:
        print('캐시된 영상이 없다. python3.9 -m scoring.extraction_cache 를 먼저 돌려라.')
        return

    print_table(results)
    phase1_gate(results)

    for r in results:
        for part in ('shoulder', 'pelvis'):
            if r[f'{part}_max_run'] >= 6:
                print(f"  ⚠ {r['video']} {part}: 연속 이상치 {r[f'{part}_max_run']}프레임 "
                      f"— Hampel 윈도우 과반 오염, 놓친 스파이크 가능")


if __name__ == '__main__':
    main()
