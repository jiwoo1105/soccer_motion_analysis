# -*- coding: utf-8 -*-
"""세 지표의 원시값 계산 (캐시된 포즈·공추적에서)

    헤드업     : 유효 터치 ±8f, head angle mean_range (world_landmarks Y축)
    상체각도   : 전체 프레임, 무릎-엉덩이-어깨 각 평균 (world_landmarks)
    어깨·골반  : 영상 전체, z축 회전 진폭 (atan2 → Hampel → savgol
                 → 드리프트 제거 → find_peaks, 연속 극값 차의 평균)

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
from scoring import rotation_measurements as rm
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


def _rotation_metrics(pose_frames):
    """어깨/골반 회전 진폭 — 영상에서 직접 계산한다

    z축 파이프라인(atan2 → Hampel → savgol → 드리프트 제거 → find_peaks)을
    매번 돌린다. 확정 측정값 표를 읽어오면 표에 없는 새 영상은 점수가 나오지
    않는다. 표(`rotation_measurements.ROTATION_AMPLITUDE`)는 이 계산이 어긋나지
    않는지 확인하는 회귀 테스트용으로만 남겨둔다 — 12편에서 평균 0.07° 일치.

    2D 폭 방식을 보류한 이유는 rotation_measurements 모듈 주석 참조.
    """
    wl = np.array([pf.world_landmarks for pf in pose_frames])
    sh_amp, sh_spikes, sh_run, sh_ext = rs.measure(wl, 11, 12)
    pe_amp, pe_spikes, pe_run, pe_ext = rs.measure(wl, 23, 24)

    return {
        'shoulder': sh_amp,
        'pelvis': pe_amp,
        'shoulder_spikes': int(sh_spikes),
        'pelvis_spikes': int(pe_spikes),
        'shoulder_max_run': int(sh_run),
        'pelvis_max_run': int(pe_run),
        'shoulder_extrema': sh_ext,
        'pelvis_extrema': pe_ext,
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

    rot = _rotation_metrics(pose_frames)

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
        # 7-3은 좌우 랜드마크 스왑으로 복구 불가 — 모든 지표에서 제외
        if stem in rm.EXCLUDED or not is_cached(stem):
            continue
        out.append(compute(stem, max_foot_dist))
    return out


def print_table(results):
    f = lambda v, d=2: (f'{v:.{d}f}' if v is not None else 'N/A')
    print(f"\n{'='*74}")
    print(f"{'영상':<14}{'점수대':>5}{'헤드업':>9}{'상체각':>9}"
          f"{'어깨진폭':>9}{'골반진폭':>9}{'터치':>9}")
    print('-' * 74)
    for r in sorted(results, key=lambda r: (r['tier'] is None, r['tier'] or 0)):
        print(f"{r['video']:<14}{r['tier'] if r['tier'] else '기준':>5}"
              f"{f(r['headup'],1):>9}{f(r['trunk'],1):>9}"
              f"{f(r['shoulder'],1):>9}{f(r['pelvis'],1):>9}"
              f"{str(r['touch_valid'])+'/'+str(r['touch_total']):>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-foot-dist', type=float, default=MAX_FOOT_DIST)
    args = ap.parse_args()

    results = load_all(args.max_foot_dist)
    if not results:
        print('캐시된 영상이 없다. python3.9 -m scoring.extraction_cache 를 먼저 돌려라.')
        return

    print_table(results)


if __name__ == '__main__':
    main()
