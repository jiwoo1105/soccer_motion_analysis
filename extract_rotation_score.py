"""
어깨/골반 회전 점수 추출
arctan2 → unwrap → Hampel 스파이크 제거(±5프레임 중앙값, 10° 임계)
→ savgol → 베이스라인 드리프트 제거 → peak-to-peak 평균으로 회전량 점수화

드리프트 제거 이유:
  Mediapipe world_landmarks의 X좌표가 프레임마다 천천히 틀린 방향으로 흘러감.
  이건 급격한 점프(스파이크)가 아니라 느린 오차 누적이라 savgol로도 제거 안 됨.
  베이스라인(큰 윈도우 savgol)을 구해서 빼면 드리프트만 제거하고
  실제 회전 움직임(빠른 변화)만 남길 수 있음.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEOS = [
    "input/in_in/인,인 3-1.MOV",
    "input/in_in/인,인 3-2.MOV",
    "input/in_in/인,인 5-1.MOV",
    "input/in_in/인,인 6-1.MOV",
    "input/in_in/인,인 7-1.MOV",
    "input/in_in/인,인 7-2.MOV",
    "input/in_in/인,인 7-3.MOV",
    "input/in_in/인,인 8-1.MOV",
    "input/in_in/인,인 8-2.MOV",
    "input/in_in/인,인 9-1.MOV",
    "input/in_in/인,인 9-2.MOV",
    "input/in_in/인,인 기준1.MOV",
    "input/in_in/인,인 기준2.MOV",
]


def extract_rotation_data(video_path):
    """영상에서 어깨/골반 world_landmarks 추출"""
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)

    sh_vx, sh_vz = [], []
    pe_vx, pe_vz = [], []
    valid_frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_world_landmarks:
                wl = results.pose_world_landmarks.landmark
                sh_vx.append(wl[12].x - wl[11].x)
                sh_vz.append(wl[12].z - wl[11].z)
                pe_vx.append(wl[24].x - wl[23].x)
                pe_vz.append(wl[24].z - wl[23].z)
                valid_frames.append(idx)
            idx += 1
    cap.release()

    return (np.array(valid_frames),
            np.array(sh_vx), np.array(sh_vz),
            np.array(pe_vx), np.array(pe_vz))


def process_angle(vx, vz, spike_thresh=10, savgol_win=41, drift_win=81,
                  hampel_half_win=5):
    """arctan2 → unwrap → 스파이크 제거(Hampel) → savgol → 베이스라인 드리프트 제거
    Returns: detrended, bad_mask, unwrapped(raw), smoothed(드리프트 제거 전), baseline

    스파이크 판별 (Hampel 방식, 2026-07 변경):
      각 프레임을 ±hampel_half_win 프레임 윈도우의 중앙값과 비교,
      편차가 spike_thresh(°)를 초과하면 이상치.
      - 기존 직전 프레임(t-1) 비교 방식은 지속형 스파이크의 중간 프레임을
        놓치고 스파이크 직후 정상 프레임을 오판하는 문제가 있었음.
      - 중앙값 기준은 윈도우 안에 스파이크가 있어도 오염되지 않고,
        기준이 실제 움직임을 따라가므로 빠른 회전 중에도 오탐이 없음.
      - 13개 영상 비교 실험: 11개는 기존과 ±1° 이내,
        2개(7-3 어깨, 3-1 골반)는 잔존 스파이크 제거로 하향 보정.
    """
    raw = np.degrees(np.arctan2(vz, vx))
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))

    # 스파이크 제거 + 보간 (Hampel: 윈도우 중앙값 대비 편차)
    cleaned = unwrapped.copy()
    n_frames = len(cleaned)
    bad_mask = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        lo = max(0, i - hampel_half_win)
        hi = min(n_frames, i + hampel_half_win + 1)
        med = np.median(cleaned[lo:hi])
        if abs(cleaned[i] - med) > spike_thresh:
            bad_mask[i] = True
    good_idx = np.where(~bad_mask)[0]
    if len(good_idx) >= 2:
        cleaned[bad_mask] = np.interp(
            np.where(bad_mask)[0], good_idx, cleaned[good_idx]
        )

    # savgol
    n = len(cleaned)
    win = min(savgol_win, n if n % 2 == 1 else n - 1)
    smoothed = savgol_filter(cleaned, win, 2) if win >= 3 else cleaned

    # 베이스라인 드리프트 제거
    dwin = min(drift_win, n if n % 2 == 1 else n - 1)
    baseline = savgol_filter(smoothed, dwin, 2) if dwin >= 3 else np.zeros(n)
    detrended = smoothed - baseline

    return detrended, bad_mask, unwrapped, smoothed, baseline


def find_peak_to_peak_score(angles, distance=10, prominence=3):
    """peak-to-peak 평균 점수 계산"""
    # 극대 찾기
    peaks, peak_props = find_peaks(angles, distance=distance, prominence=prominence)
    # 극소 찾기
    valleys, valley_props = find_peaks(-angles, distance=distance, prominence=prominence)

    # 극대 + 극소 합쳐서 시간순 정렬
    all_extrema = sorted(list(peaks) + list(valleys))

    if len(all_extrema) < 2:
        return None, [], []

    # 연속 extrema 간 각도 차이
    diffs = []
    for i in range(len(all_extrema) - 1):
        diff = abs(angles[all_extrema[i+1]] - angles[all_extrema[i]])
        diffs.append(diff)

    score = float(np.mean(diffs))
    return score, all_extrema, diffs


def process_video(video_path):
    """한 영상 처리 → 회전 점수"""
    from pathlib import Path
    stem = Path(video_path).stem
    print(f"\n  처리 중: {stem}")

    frames, sh_vx, sh_vz, pe_vx, pe_vz = extract_rotation_data(video_path)
    n = len(frames)
    if n < 20:
        print(f"    프레임 부족: {n}")
        return None

    # 어깨/골반 각도 처리
    sh_angle, sh_bad, sh_raw, sh_before_detrend, sh_baseline = process_angle(sh_vx, sh_vz)
    pe_angle, pe_bad, pe_raw, pe_before_detrend, pe_baseline = process_angle(pe_vx, pe_vz)

    # peak-to-peak 점수
    sh_score, sh_extrema, sh_diffs = find_peak_to_peak_score(sh_angle)
    pe_score, pe_extrema, pe_diffs = find_peak_to_peak_score(pe_angle)

    print(f"    어깨: score={sh_score:.1f}° ({len(sh_extrema)}개 극값, {sh_bad.sum()}개 스파이크 제거)")
    print(f"    골반: score={pe_score:.1f}° ({len(pe_extrema)}개 극값, {pe_bad.sum()}개 스파이크 제거)")

    return {
        'video': stem,
        'shoulder_score': sh_score,
        'pelvis_score': pe_score,
        'sh_extrema_count': len(sh_extrema),
        'pe_extrema_count': len(pe_extrema),
        'sh_diffs': sh_diffs,
        'pe_diffs': pe_diffs,
        'frames': frames,
        'sh_angle': sh_angle,
        'pe_angle': pe_angle,
        'sh_raw': sh_raw,
        'pe_raw': pe_raw,
        'sh_before_detrend': sh_before_detrend,
        'pe_before_detrend': pe_before_detrend,
        'sh_baseline': sh_baseline,
        'pe_baseline': pe_baseline,
        'sh_bad': sh_bad,
        'pe_bad': pe_bad,
        'sh_extrema': sh_extrema,
        'pe_extrema': pe_extrema,
    }


def plot_detail(result, output_dir='output/graphs'):
    """개별 영상 상세 그래프: 위=처리 과정, 아래=드리프트 제거 결과"""
    os.makedirs(output_dir, exist_ok=True)
    stem = result['video']
    frames = result['frames']

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    for col, (label, raw, before_dt, baseline, final, extrema, bad, score, color) in enumerate([
        ('어깨', result['sh_raw'], result['sh_before_detrend'], result['sh_baseline'],
         result['sh_angle'], result['sh_extrema'], result['sh_bad'], result['shoulder_score'], 'blue'),
        ('골반', result['pe_raw'], result['pe_before_detrend'], result['pe_baseline'],
         result['pe_angle'], result['pe_extrema'], result['pe_bad'], result['pelvis_score'], 'red'),
    ]):
        # 위: 처리 과정 (raw → savgol → 베이스라인 표시)
        ax = axes[0][col]
        ax.plot(frames, raw, '-', color='gray', alpha=0.3, lw=1, label='① raw')
        if bad.any():
            ax.scatter(frames[bad], raw[bad], color='orange', s=20, marker='x', zorder=4,
                       label=f'스파이크 ({bad.sum()}개)')
        ax.plot(frames, before_dt, '-', color=color, alpha=0.5, lw=1.5, label='② 10°제거+savgol41')
        ax.plot(frames, baseline, 'g--', lw=2, label='③ 베이스라인(드리프트)')
        ax.set_ylabel('각도 (°)')
        ax.set_title(f'{label}: 처리 과정 (초록점선 = 제거할 드리프트)', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # 아래: 드리프트 제거 후 결과 + peak
        ax = axes[1][col]
        ax.plot(frames, final, '-', color=color, lw=2, label='드리프트 제거 후')
        ax.axhline(0, color='gray', lw=0.5, ls='--')

        if extrema:
            ax.scatter(frames[extrema], final[extrema],
                       color='red', s=80, zorder=5, edgecolors='black', linewidths=0.5)
            for i in range(len(extrema) - 1):
                e1, e2 = extrema[i], extrema[i + 1]
                diff = abs(final[e2] - final[e1])
                mid_x = frames[(e1 + e2) // 2]
                mid_y = (final[e1] + final[e2]) / 2
                ax.annotate(f'{diff:.1f}°', (mid_x, mid_y),
                            fontsize=9, color='black', fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

        score_str = f'{score:.1f}' if score else 'N/A'
        ax.set_ylabel('각도 변화 (°)')
        ax.set_xlabel('Frame')
        ax.set_title(f'{label} 회전 | score = {score_str}° (peak-to-peak 평균)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(stem, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rotation_score_{stem}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    results = []
    for v in VIDEOS:
        if not os.path.exists(v):
            print(f"  파일 없음: {v}")
            continue
        r = process_video(v)
        if r:
            results.append(r)

    # 전체 결과 테이블
    print(f"\n{'='*70}")
    print(f"{'영상':<20} {'어깨 점수':>10} {'골반 점수':>10} {'어깨 극값':>8} {'골반 극값':>8}")
    print(f"{'-'*70}")
    for r in results:
        sh = f"{r['shoulder_score']:.1f}°" if r['shoulder_score'] else "N/A"
        pe = f"{r['pelvis_score']:.1f}°" if r['pelvis_score'] else "N/A"
        print(f"{r['video']:<20} {sh:>10} {pe:>10} {r['sh_extrema_count']:>8} {r['pe_extrema_count']:>8}")

    # 전체 영상 그래프 생성
    for r in results:
        plot_detail(r)
        print(f"  그래프 저장: output/graphs/rotation_score_{r['video']}.png")


if __name__ == "__main__":
    main()
