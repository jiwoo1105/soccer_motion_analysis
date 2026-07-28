# -*- coding: utf-8 -*-
"""터치 유효성 판정

헤드업과 어깨-골반은 터치 ±8프레임 윈도우에서 계산되므로, 그 윈도우가 온전하지
않은 터치를 섞으면 지표가 오염된다. 터치 단위로 걸러내되 영상을 통째로 버리지는
않고, 유효 비율을 신뢰도 플래그로 함께 보고한다.
"""
WINDOW = 8
MIN_VISIBILITY = 0.5

# 헤드업 각도에 필요한 랜드마크 (눈 2개 + 어깨 2개)
HEAD_LANDMARKS = (2, 5, 11, 12)


def filter_touches(touches, pose_frames, ball_tracks, window=WINDOW):
    """유효 터치만 남긴다.

    유효 조건 (전부 만족):
      1. 공-발 2D 거리 ≤ 200px — 터치 감지 단계에서 이미 적용됨
      2. ±window 프레임 윈도우가 영상 범위 안에 완전히 들어옴
      3. 윈도우 내 모든 프레임에 pose 랜드마크 존재 (visibility ≥ 0.5)
      4. 윈도우 내 SAM2 공 추적 끊김 없음

    Returns:
        (valid_touches, reasons) — reasons는 탈락 터치의 {frame: 사유} 맵
    """
    frame_map = {pf.frame_number: pf for pf in pose_frames}
    if not frame_map:
        return [], {}

    lo_bound = min(frame_map)
    hi_bound = max(frame_map)

    valid = []
    reasons = {}

    for t in touches:
        tf = t.frame_number
        lo, hi = tf - window, tf + window

        if lo < lo_bound or hi > hi_bound:
            reasons[tf] = '윈도우가 영상 범위를 벗어남'
            continue

        window_frames = range(lo, hi + 1)

        missing_pose = [f for f in window_frames if f not in frame_map]
        if missing_pose:
            reasons[tf] = f'윈도우 내 pose 결손 {len(missing_pose)}프레임'
            continue

        low_vis = [
            f for f in window_frames
            if any(frame_map[f].visibility[i] < MIN_VISIBILITY for i in HEAD_LANDMARKS)
        ]
        if low_vis:
            reasons[tf] = f'윈도우 내 visibility 미달 {len(low_vis)}프레임'
            continue

        missing_ball = [f for f in window_frames if f not in ball_tracks]
        if missing_ball:
            reasons[tf] = f'윈도우 내 공 추적 끊김 {len(missing_ball)}프레임'
            continue

        valid.append(t)

    return valid, reasons


def touch_quality(valid_count, total_count):
    """유효 터치 비율. 터치가 하나도 없으면 0.0"""
    if total_count == 0:
        return 0.0
    return valid_count / total_count
