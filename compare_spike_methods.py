# -*- coding: utf-8 -*-
"""
스파이크(이상치) 판별 기준 비교 스크립트

기존 방식 (old): 직전 프레임(t-1)과의 차이 > 10° → 이상치
개선 방식 (new): 가장 최근 '정상' 프레임(t_a)과의 차이 > 10° → 이상치

- 영상마다 MediaPipe 추출은 1회만 수행하고, 두 방식으로 각각 처리해서
  어깨/골반 회전 점수(S_rot)를 나란히 비교 출력한다.
- 기존 extract_rotation_score.py는 건드리지 않는다.

실행: python3 compare_spike_methods.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.signal import savgol_filter, find_peaks

# 기존 스크립트의 추출 함수 재사용 (MediaPipe world landmarks → 어깨/골반 벡터)
from extract_rotation_score import extract_rotation_data, VIDEOS


# ── 이상치 판별 두 방식 ──────────────────────────────────────────
def detect_spikes_prev(unwrapped, thresh=10):
    """기존: 직전 프레임(t-1) 대비 차이 > thresh → 이상치"""
    bad = np.zeros(len(unwrapped), dtype=bool)
    for i in range(1, len(unwrapped)):
        if abs(unwrapped[i] - unwrapped[i - 1]) > thresh:
            bad[i] = True
    return bad


def detect_spikes_lastvalid(unwrapped, thresh=10):
    """(기각됨) 가장 최근 '정상' 프레임(t_a) 대비 차이 > thresh → 이상치

    실험 결과: 스파이크 직후 실제 회전이 진행 중이면 기준이 동결되어
    정상 프레임을 대량 오탐(영상당 60~300 플래그) → 채택 불가.
    """
    bad = np.zeros(len(unwrapped), dtype=bool)
    last_valid = unwrapped[0]
    for i in range(1, len(unwrapped)):
        if abs(unwrapped[i] - last_valid) > thresh:
            bad[i] = True
        else:
            last_valid = unwrapped[i]
    return bad


def detect_spikes_trailmean(unwrapped, thresh=10, n_prev=5):
    """교수님 원안: 이전 n_prev개 프레임의 '평균'과의 차이 > thresh → 이상치

    (과거 방향 sliding window + 평균 기준)
    - 윈도우에 스파이크가 포함되면 평균이 오염되어 기준이 밀리는 약점 검증용
    - 기준이 항상 과거라서 빠른 회전 구간에서 지연 오탐 가능성도 검증
    """
    n = len(unwrapped)
    bad = np.zeros(n, dtype=bool)
    for i in range(1, n):
        lo = max(0, i - n_prev)
        ref = np.mean(unwrapped[lo:i])
        if abs(unwrapped[i] - ref) > thresh:
            bad[i] = True
    return bad


def detect_spikes_hampel(unwrapped, thresh=10, half_win=5):
    """개선 v2 (Hampel 방식): 주변 윈도우의 중앙값 대비 차이 > thresh → 이상치

    - 교수님 제안(이전 N프레임 평균 비교)의 강건한 버전:
      평균 대신 중앙값 → 윈도우 안에 스파이크가 있어도 기준이 오염되지 않음
    - 중심 윈도우(앞뒤 half_win 프레임) → 기준이 실제 움직임을 따라가므로
      빠른 회전 중에도 정상 프레임을 오탐하지 않음 (오프라인 처리라 가능)
    """
    n = len(unwrapped)
    bad = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - half_win), min(n, i + half_win + 1)
        med = np.median(unwrapped[lo:hi])
        if abs(unwrapped[i] - med) > thresh:
            bad[i] = True
    return bad


# ── 공통 파이프라인 (판별 방식만 갈아끼움) ────────────────────────
def process_angle(vx, vz, detect_fn, spike_thresh=10, savgol_win=41, drift_win=81):
    """atan2 → unwrap → 이상치 판별(detect_fn) → 선형 보간 → savgol → 드리프트 제거"""
    raw = np.degrees(np.arctan2(vz, vx))
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))

    bad_mask = detect_fn(unwrapped, spike_thresh)

    cleaned = unwrapped.copy()
    good_idx = np.where(~bad_mask)[0]
    if len(good_idx) >= 2:
        cleaned[bad_mask] = np.interp(
            np.where(bad_mask)[0], good_idx, cleaned[good_idx]
        )

    n = len(cleaned)
    win = min(savgol_win, n if n % 2 == 1 else n - 1)
    smoothed = savgol_filter(cleaned, win, 2) if win >= 3 else cleaned

    dwin = min(drift_win, n if n % 2 == 1 else n - 1)
    baseline = savgol_filter(smoothed, dwin, 2) if dwin >= 3 else np.zeros(n)
    detrended = smoothed - baseline

    return detrended, bad_mask


def peak_to_peak_score(angles, distance=10, prominence=3):
    """극점 탐색 → 연속 극값 차의 평균 (기존과 동일)"""
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    extrema = sorted(list(peaks) + list(valleys))
    if len(extrema) < 2:
        return None, 0
    diffs = [abs(angles[extrema[i + 1]] - angles[extrema[i]])
             for i in range(len(extrema) - 1)]
    return float(np.mean(diffs)), len(extrema)


def fmt(v):
    return f"{v:.1f}" if v is not None else "N/A"


def main():
    rows = []
    for video in VIDEOS:
        if not os.path.exists(video):
            print(f"  파일 없음: {video}")
            continue
        from pathlib import Path
        stem = Path(video).stem
        print(f"\n처리 중: {stem}")

        frames, sh_vx, sh_vz, pe_vx, pe_vz = extract_rotation_data(video)
        if len(frames) < 20:
            print("  프레임 부족, 건너뜀")
            continue

        row = {'video': stem}
        for part, vx, vz in [('sh', sh_vx, sh_vz), ('pe', pe_vx, pe_vz)]:
            for tag, fn in [('old', detect_spikes_prev),
                            ('new', detect_spikes_lastvalid),
                            ('avg', detect_spikes_trailmean),
                            ('hmp', detect_spikes_hampel)]:
                ang, bad = process_angle(vx, vz, fn)
                score, n_ext = peak_to_peak_score(ang)
                row[f'{part}_{tag}'] = score
                row[f'{part}_{tag}_bad'] = int(bad.sum())
                row[f'{part}_{tag}_ext'] = n_ext
        rows.append(row)

        print(f"  어깨: old={fmt(row['sh_old'])}° ({row['sh_old_bad']})  "
              f"new={fmt(row['sh_new'])}° ({row['sh_new_bad']})  "
              f"avg={fmt(row['sh_avg'])}° ({row['sh_avg_bad']})  "
              f"hampel={fmt(row['sh_hmp'])}° ({row['sh_hmp_bad']})")
        print(f"  골반: old={fmt(row['pe_old'])}° ({row['pe_old_bad']})  "
              f"new={fmt(row['pe_new'])}° ({row['pe_new_bad']})  "
              f"avg={fmt(row['pe_avg'])}° ({row['pe_avg_bad']})  "
              f"hampel={fmt(row['pe_hmp'])}° ({row['pe_hmp_bad']})")

    # ── 요약 표 1: 점수 (old / 평균윈도우 / Hampel) ────────────
    print(f"\n{'='*104}")
    print(f"{'영상':<15} {'어깨old':>8} {'어깨avg':>8} {'어깨hmp':>8} "
          f"{'골반old':>8} {'골반avg':>8} {'골반hmp':>8}")
    print('-' * 104)
    for r in rows:
        print(f"{r['video']:<15} {fmt(r['sh_old']):>8} {fmt(r['sh_avg']):>8} {fmt(r['sh_hmp']):>8} "
              f"{fmt(r['pe_old']):>8} {fmt(r['pe_avg']):>8} {fmt(r['pe_hmp']):>8}")

    # ── 요약 표 2: 플래그 수 (오탐 규모 비교) ──────────────────
    print(f"\n{'영상':<15} {'어깨old':>8} {'어깨avg':>8} {'어깨hmp':>8} "
          f"{'골반old':>8} {'골반avg':>8} {'골반hmp':>8}   (플래그 프레임 수)")
    print('-' * 104)
    for r in rows:
        print(f"{r['video']:<15} {r['sh_old_bad']:>8} {r['sh_avg_bad']:>8} {r['sh_hmp_bad']:>8} "
              f"{r['pe_old_bad']:>8} {r['pe_avg_bad']:>8} {r['pe_hmp_bad']:>8}")

    # CSV 저장
    import csv
    out = 'output/spike_method_comparison.csv'
    os.makedirs('output', exist_ok=True)
    if rows:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 저장: {out}")


if __name__ == "__main__":
    main()
