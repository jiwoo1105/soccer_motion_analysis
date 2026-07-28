# -*- coding: utf-8 -*-
"""어깨/골반 2D 폭 → 회전각 → 진폭 (Phase 1 신호원)

기존 `extract_rotation_score.py`는 world landmark의 z축으로 atan2(vz, vx)를 썼는데,
z는 촬영 각도에 따라 눌린다(8점대 두 영상이 어깨·골반 모두 최저권으로 나온 원인).
그래서 image landmark의 2D 폭만 쓴다.

처리 순서는 기존 파이프라인을 유지한다. 이상치 제거를 드리프트 제거보다 먼저
해야 한다 — 스파이크가 남은 채로 baseline을 뽑으면 주변까지 오염된다.

    폭 → 각도 → unwrap → Hampel 이상치 제거 → 선형 보간 → savgol(41,2)
       → 드리프트 제거(savgol 81,2) → find_peaks(distance=10, prominence=3)
       → 연속 극값 차의 평균 = 진폭
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


def width_to_angle(width):
    """2D 폭 → 회전각 (°)

    어깨선이 카메라와 정면일 때 화면상 폭이 최대이고, 몸을 돌릴수록 폭이
    cos(θ)에 비례해 줄어든다. 따라서 θ = arccos(w / w_max) 다.

    w_max는 최대값 대신 **95 백분위수**를 쓴다. 랜드마크가 한 프레임 튀어서
    폭이 과대추정되면 그 뒤 전 구간의 각도가 통째로 밀리기 때문이다.

    한계 — 이 변환은 **부호를 잃는다.** 좌회전과 우회전이 모두 양수 θ로 접힌다.
    회전 방향은 구분할 수 없고 회전량(진폭)만 측정된다. 이번 지표는 진폭 비율만
    쓰므로 문제되지 않지만, 방향이 필요한 지표에는 이 신호를 쓰면 안 된다.
    """
    w = np.asarray(width, dtype=float)
    w_max = float(np.percentile(w, 95))
    if w_max <= 0:
        return np.zeros_like(w)
    return np.degrees(np.arccos(np.clip(w / w_max, 0.0, 1.0)))


def detect_spikes_hampel(angles, thresh=SPIKE_THRESH, half_win=HAMPEL_HALF_WIN):
    """Hampel 이상치 판별 — 전후 ±half_win 윈도우 중앙값 대비 편차

    중앙값 기준이라 윈도우 안에 스파이크가 있어도 오염되지 않고, 기준이 실제
    움직임을 따라가므로 빠른 회전 중에도 오탐이 없다.
    (avg / lastvalid / old 방식은 선행 분석에서 기각됐다.)

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

    # 최장 연속 구간 길이
    max_run = run = 0
    for flag in bad:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)

    return bad, max_run


def clean_angle(angles):
    """이상치 제거 → 보간 → 평활 → 드리프트 제거

    Returns:
        detrended, bad, max_run
    """
    a = np.asarray(angles, dtype=float)
    n = len(a)

    # unwrap: 2D 폭 기반 각도는 0~90° 범위라 ±180° 경계를 넘지 않으므로
    # 사실상 no-op다. z축 파이프라인과 순서를 맞추기 위해 남겨둔다.
    a = np.degrees(np.unwrap(np.radians(a)))

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


def find_extrema(angles):
    """극대 + 극소를 시간순으로 반환"""
    a = np.asarray(angles, dtype=float)
    peaks, _ = find_peaks(a, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    valleys, _ = find_peaks(-a, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    return sorted(list(peaks) + list(valleys))


def amplitude(angles, allowed=None):
    """연속 극값 차의 평균 = 회전 진폭 (°)

    Args:
        angles: 드리프트 제거된 각도 시계열
        allowed: 허용 인덱스 집합. None이면 전체 구간.

    Returns:
        (진폭, 극값 인덱스 리스트). 극값이 2개 미만이면 (None, ext).

    주의 — 평활·드리프트 제거는 반드시 **영상 전체**에서 하고, 구간 제한은
    극값을 고르는 단계에서만 적용한다. 짧은 구간만 잘라서 savgol을 걸면
    경계에서 가짜 스윙이 생긴다.
    """
    a = np.asarray(angles, dtype=float)
    ext = find_extrema(a)

    if allowed is not None:
        ext = [e for e in ext if e in allowed]

    if len(ext) < 2:
        return None, ext

    diffs = [abs(a[ext[i + 1]] - a[ext[i]]) for i in range(len(ext) - 1)]
    return float(np.mean(diffs)), ext


def touch_window_indices(centers, n, window=8):
    """터치 중심 인덱스들의 ±window 합집합"""
    allowed = set()
    for c in centers:
        allowed.update(range(max(0, c - window), min(n, c + window + 1)))
    return allowed


def max_cross_correlation(x, y):
    """두 시계열의 정규화 교차상관 최대값 (Phase 1 게이트용)

    어깨와 골반이 사실상 같은 신호면 1에 가깝다. 평균이 0.9 이상이면
    이 동작에서 몸통 분리가 관측되지 않는다는 뜻이다.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return 0.0
    return float(np.max(np.correlate(a, b, mode='full')) / denom)
