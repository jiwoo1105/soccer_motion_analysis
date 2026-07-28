# analysis/dribble_cycle_analyzer.py
"""
90도 측면 촬영 기준 공 크기 변화로 드리블 방향 판별

원리:
  반지름 증가 (trough→peak): 공이 카메라 쪽으로 옴 → NEAR 터치
  반지름 감소 (peak→trough): 공이 카메라 반대로 감 → FAR 터치

  NEAR↔FAR 교대 횟수 = 드리블 사이클 수
  x좌표는 사용하지 않음 (대각선 드리블에서 무의미)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy.signal import find_peaks, savgol_filter


@dataclass
class HalfCycle:
    start_frame: int    # 시작 프레임 (trough 또는 peak)
    end_frame:   int    # 종료 프레임 (peak 또는 trough)
    direction:   str    # 'NEAR' (반지름↑) or 'FAR' (반지름↓)
    r_change:    float  # 반지름 변화량 (px)


@dataclass
class DribbleCycleData:
    frame_nums:    np.ndarray
    radii_raw:     np.ndarray
    radii_smooth:  np.ndarray
    x_series:      np.ndarray
    peak_frames:   List[int]
    trough_frames: List[int]
    half_cycles:   List[HalfCycle]   # half-cycle 목록
    cycle_count:   int               # 완전한 사이클 수 (peak 개수 - 1)
    near_count:    int               # 카메라 쪽으로 온 횟수
    far_count:     int               # 카메라 반대로 간 횟수


def analyze_dribble_cycles(all_ball_positions: dict,
                            fps: float = 30.0,
                            smooth_window: int = 9,
                            peak_prominence: float = 1.5,
                            peak_distance: int = 20) -> Optional[DribbleCycleData]:
    if not all_ball_positions or len(all_ball_positions) < 10:
        return None

    sorted_frames = sorted(all_ball_positions.keys())
    frame_nums = np.array(sorted_frames)
    radii_raw  = np.array([all_ball_positions[f][2] for f in sorted_frames], dtype=float)
    x_series   = np.array([all_ball_positions[f][0] for f in sorted_frames], dtype=float)

    win = min(smooth_window, len(radii_raw) if len(radii_raw) % 2 == 1 else len(radii_raw) - 1)
    win = max(win, 3)
    radii_smooth = savgol_filter(radii_raw, window_length=win, polyorder=2)

    peak_idx,   _ = find_peaks( radii_smooth, prominence=peak_prominence, distance=peak_distance)
    trough_idx, _ = find_peaks(-radii_smooth, prominence=peak_prominence, distance=peak_distance)

    peak_frames   = [int(frame_nums[i]) for i in peak_idx]
    trough_frames = [int(frame_nums[i]) for i in trough_idx]

    # peak/trough를 시간 순으로 합쳐서 반-사이클 구성
    events = (
        [(int(frame_nums[i]), 'peak',   radii_smooth[i]) for i in peak_idx] +
        [(int(frame_nums[i]), 'trough', radii_smooth[i]) for i in trough_idx]
    )
    events.sort(key=lambda e: e[0])

    half_cycles = []
    for i in range(len(events) - 1):
        f1, t1, r1 = events[i]
        f2, t2, r2 = events[i + 1]
        if t1 == t2:          # peak-peak or trough-trough 연속이면 스킵
            continue
        direction = 'NEAR' if t2 == 'peak' else 'FAR'
        half_cycles.append(HalfCycle(
            start_frame=f1,
            end_frame=f2,
            direction=direction,
            r_change=float(r2 - r1),
        ))

    near_count = sum(1 for h in half_cycles if h.direction == 'NEAR')
    far_count  = sum(1 for h in half_cycles if h.direction == 'FAR')

    return DribbleCycleData(
        frame_nums=frame_nums,
        radii_raw=radii_raw,
        radii_smooth=radii_smooth,
        x_series=x_series,
        peak_frames=peak_frames,
        trough_frames=trough_frames,
        half_cycles=half_cycles,
        cycle_count=max(0, len(peak_frames) - 1),
        near_count=near_count,
        far_count=far_count,
    )
