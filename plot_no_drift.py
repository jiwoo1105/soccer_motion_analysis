"""
10° threshold + savgol만 적용한 그래프 (드리프트 제거 없이)
raw + 10°제거+savgol + peak 표시
output/graphs/no_drift/ 폴더에 저장
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from extract_rotation_score import extract_rotation_data, VIDEOS

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'output/graphs/no_drift'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_no_drift(vx, vz, spike_thresh=10, savgol_win=41):
    """10° 스파이크 제거 + savgol만 (드리프트 제거 없음)"""
    raw = np.degrees(np.arctan2(vz, vx))
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))

    cleaned = unwrapped.copy()
    bad_mask = np.zeros(len(cleaned), dtype=bool)
    for i in range(1, len(cleaned)):
        if abs(cleaned[i] - cleaned[i-1]) > spike_thresh:
            bad_mask[i] = True
    good_idx = np.where(~bad_mask)[0]
    if len(good_idx) >= 2:
        cleaned[bad_mask] = np.interp(np.where(bad_mask)[0], good_idx, cleaned[good_idx])

    n = len(cleaned)
    win = min(savgol_win, n if n % 2 == 1 else n - 1)
    smoothed = savgol_filter(cleaned, win, 2) if win >= 3 else cleaned

    return unwrapped, smoothed, bad_mask


def get_score(angles, distance=10, prominence=3):
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    extrema = sorted(list(peaks) + list(valleys))
    if len(extrema) < 2:
        return None, [], []
    diffs = [abs(angles[extrema[i+1]] - angles[extrema[i]]) for i in range(len(extrema)-1)]
    return float(np.mean(diffs)), extrema, diffs


from pathlib import Path

all_results = []

for v in VIDEOS:
    if not os.path.exists(v):
        continue
    stem = Path(v).stem
    print(f"  처리 중: {stem}")

    frames, sh_vx, sh_vz, pe_vx, pe_vz = extract_rotation_data(v)
    n = len(frames)
    if n < 20:
        continue

    sh_raw, sh_smooth, sh_bad = process_no_drift(sh_vx, sh_vz)
    pe_raw, pe_smooth, pe_bad = process_no_drift(pe_vx, pe_vz)

    sh_score, sh_ext, sh_diffs = get_score(sh_smooth)
    pe_score, pe_ext, pe_diffs = get_score(pe_smooth)

    print(f"    어깨: {sh_score:.1f}° | 골반: {pe_score:.1f}°")

    all_results.append({
        'video': stem, 'frames': frames,
        'sh_raw': sh_raw, 'pe_raw': pe_raw,
        'sh_smooth': sh_smooth, 'pe_smooth': pe_smooth,
        'sh_bad': sh_bad, 'pe_bad': pe_bad,
        'sh_score': sh_score, 'pe_score': pe_score,
        'sh_ext': sh_ext, 'pe_ext': pe_ext,
    })

    # ── 그래프 ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    for ax_idx, (label, raw, smooth, bad, score, ext, color) in enumerate([
        ('어깨', sh_raw, sh_smooth, sh_bad, sh_score, sh_ext, 'blue'),
        ('골반', pe_raw, pe_smooth, pe_bad, pe_score, pe_ext, 'red'),
    ]):
        ax = axes[ax_idx]

        # raw
        ax.plot(frames, raw, '-', color='gray', alpha=0.4, lw=1, label='① raw (unwrap)')
        # 스파이크
        if bad.any():
            ax.scatter(frames[bad], raw[bad], color='orange', s=25, marker='x',
                       zorder=4, label=f'스파이크 제거 ({bad.sum()}개)')
        # 10° + savgol
        ax.plot(frames, smooth, '-', color=color, lw=2.5, label='② 10° 제거 + savgol41')

        # peak
        if ext:
            ax.scatter(frames[ext], smooth[ext],
                       color='red', s=80, zorder=5, edgecolors='black', linewidths=0.5)
            for i in range(len(ext) - 1):
                e1, e2 = ext[i], ext[i + 1]
                diff = abs(smooth[e2] - smooth[e1])
                mid_x = frames[(e1 + e2) // 2]
                mid_y = (smooth[e1] + smooth[e2]) / 2
                ax.annotate(f'{diff:.1f}°', (mid_x, mid_y),
                            fontsize=9, color='black', fontweight='bold', ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

        score_str = f'{score:.1f}' if score else 'N/A'
        ax.set_ylabel('각도 (°)')
        ax.set_title(f'{label} 회전 | score = {score_str}° (peak-to-peak 평균)  [드리프트 제거 없음]', fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Frame')
    plt.suptitle(f'{stem}  (10° threshold + savgol41, 드리프트 제거 없음)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rotation_no_drift_{stem}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    저장: {OUTPUT_DIR}/rotation_no_drift_{stem}.png")

# 전체 결과 테이블
print(f"\n{'='*50}")
print(f"{'영상':<16} {'어깨':>8} {'골반':>8}")
print(f"{'-'*50}")
for r in all_results:
    print(f"{r['video']:<16} {r['sh_score']:>7.1f}° {r['pe_score']:>7.1f}°")
