# -*- coding: utf-8 -*-
"""어깨/골반 회전각 처리 파이프라인 (z축)

확정 측정값은 `scoring/rotation_measurements.py`에 있다. 이 모듈은 그 값을
어떻게 얻었는지 기록하고 재현할 수 있게 남겨둔 것이다 — 채점 경로에서는
표를 직접 읽으므로 매 실행마다 돌지 않는다.

    atan2(vz, vx) → unwrap → Hampel 이상치 제거 → 선형 보간 → savgol(41,2)
      → 드리프트 제거(savgol 81,2) → find_peaks(distance=10, prominence=3)
      → 연속 극값 차의 평균 = 진폭

이상치 제거를 드리프트 제거보다 **먼저** 해야 한다. 스파이크가 남은 채로
baseline을 뽑으면 주변까지 오염된다.
"""
import numpy as np
from scipy.signal import savgol_filter, find_peaks

SPIKE_THRESH = 10.0     # Hampel 임계 (°)
HAMPEL_HALF_WIN = 5     # Hampel 윈도우 반폭 (전후 ±5프레임)
SAVGOL_WIN = 41
DRIFT_WIN = 81
PEAK_DISTANCE = 10
PEAK_PROMINENCE = 3
EDGE_GUARD = 5          # 창이 잘리는 앞뒤 프레임은 이상치 판정에서 제외


def z_axis_angle(world_landmarks, left_idx, right_idx):
    """world landmark z축 기반 회전각 (°)

    Args:
        world_landmarks: (n_frames, 33, 3)
        left_idx, right_idx: 어깨는 (11, 12), 골반은 (23, 24)
    """
    wl = np.asarray(world_landmarks)
    vx = wl[:, right_idx, 0] - wl[:, left_idx, 0]
    vz = wl[:, right_idx, 2] - wl[:, left_idx, 2]
    raw = np.degrees(np.arctan2(vz, vx))
    return np.degrees(np.unwrap(np.radians(raw)))


def detect_spikes_hampel(angles, thresh=SPIKE_THRESH, half_win=HAMPEL_HALF_WIN):
    """Hampel 이상치 판별 — 전후 ±half_win 윈도우 중앙값 대비 편차

    중앙값 기준이라 윈도우 안에 스파이크가 있어도 오염되지 않고, 기준이 실제
    움직임을 따라가므로 빠른 회전 중에도 오탐이 없다.
    avg / lastvalid / old 방식은 선행 분석에서 기각됐다.

    Returns:
        bad: bool 배열
        max_run: 최장 연속 플래그 길이. 윈도우 11개 중 과반(6개)이 오염되면
                 중앙값이 스파이크 쪽으로 넘어가 놓치므로, 6 이상이면 경고 대상이다.
    """
    a = np.asarray(angles, dtype=float)
    n = len(a)
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        if abs(a[i] - np.median(a[lo:hi])) > thresh:
            bad[i] = True

    # 경계효과 제거 — 창이 잘리는 앞뒤 구간은 판정에서 제외
    bad[:EDGE_GUARD] = False
    bad[-EDGE_GUARD:] = False

    max_run = run = 0
    for flag in bad:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)

    return bad, max_run


def clean_angle(angles):
    """이상치 제거 → 보간 → savgol 평활 → 드리프트 제거

    Returns:
        detrended, bad, max_run
    """
    a = np.asarray(angles, dtype=float)
    n = len(a)

    bad, max_run = detect_spikes_hampel(a)
    cleaned = a.copy()
    good = np.where(~bad)[0]
    if len(good) >= 2 and bad.any():
        cleaned[bad] = np.interp(np.where(bad)[0], good, cleaned[good])

    win = min(SAVGOL_WIN, n if n % 2 == 1 else n - 1)
    smoothed = savgol_filter(cleaned, win, 2) if win >= 3 else cleaned

    dwin = min(DRIFT_WIN, n if n % 2 == 1 else n - 1)
    baseline = savgol_filter(smoothed, dwin, 2) if dwin >= 3 else np.zeros(n)

    return smoothed - baseline, bad, max_run


def amplitude(angles):
    """연속 극값 차의 평균 = 회전 진폭 (°)

    Returns:
        (진폭, 극값 인덱스 리스트). 극값이 2개 미만이면 (None, ext).
    """
    a = np.asarray(angles, dtype=float)
    peaks, _ = find_peaks(a, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    valleys, _ = find_peaks(-a, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    ext = sorted(list(peaks) + list(valleys))

    if len(ext) < 2:
        return None, ext

    diffs = [abs(a[ext[i + 1]] - a[ext[i]]) for i in range(len(ext) - 1)]
    return float(np.mean(diffs)), ext


def measure(world_landmarks, left_idx, right_idx):
    """전체 파이프라인 — 확정 측정값을 재현할 때 쓴다"""
    angle = z_axis_angle(world_landmarks, left_idx, right_idx)
    detrended, bad, max_run = clean_angle(angle)
    amp, ext = amplitude(detrended)
    return amp, bad.sum(), max_run, len(ext)
