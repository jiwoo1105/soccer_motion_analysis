"""
점수대별 어깨-골반 각도 분석 + 높은 각도 원인 분석
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks

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


def extract_all(video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    sh_vx, sh_vz, pe_vx, pe_vz = [], [], [], []
    sh_w2d, pe_w2d = [], []

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_world_landmarks and results.pose_landmarks:
                wl = results.pose_world_landmarks.landmark
                il = results.pose_landmarks.landmark
                sh_vx.append(wl[12].x - wl[11].x)
                sh_vz.append(wl[12].z - wl[11].z)
                pe_vx.append(wl[24].x - wl[23].x)
                pe_vz.append(wl[24].z - wl[23].z)
                sh_w2d.append(abs(il[12].x - il[11].x) * w_frame)
                pe_w2d.append(abs(il[24].x - il[23].x) * w_frame)
    cap.release()
    return (np.array(sh_vx), np.array(sh_vz),
            np.array(pe_vx), np.array(pe_vz),
            np.array(sh_w2d), np.array(pe_w2d))


def process_angle(vx, vz, spike_thresh=10, savgol_win=41):
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
    return smoothed, bad_mask, unwrapped


def get_score(angles, distance=10, prominence=3):
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    extrema = sorted(list(peaks) + list(valleys))
    if len(extrema) < 2:
        return None, [], []
    diffs = [abs(angles[extrema[i+1]] - angles[extrema[i]]) for i in range(len(extrema)-1)]
    return float(np.mean(diffs)), extrema, diffs


def main():
    from pathlib import Path

    all_results = []
    for v in VIDEOS:
        if not os.path.exists(v):
            continue
        stem = Path(v).stem
        print(f"  처리 중: {stem}")
        sh_vx, sh_vz, pe_vx, pe_vz, sh_w2d, pe_w2d = extract_all(v)
        n = len(sh_vx)
        if n < 20:
            continue

        sh_angle, sh_bad, sh_raw = process_angle(sh_vx, sh_vz)
        pe_angle, pe_bad, pe_raw = process_angle(pe_vx, pe_vz)

        sh_score, sh_ext, sh_diffs = get_score(sh_angle)
        pe_score, pe_ext, pe_diffs = get_score(pe_angle)

        # 벡터 길이 (노이즈 민감도 지표)
        sh_vec_len = np.mean(np.sqrt(np.array(sh_vx)**2 + np.array(sh_vz)**2))
        pe_vec_len = np.mean(np.sqrt(np.array(pe_vx)**2 + np.array(pe_vz)**2))

        # raw 변화량 (노이즈 크기)
        sh_raw_diff_mean = np.mean(np.abs(np.diff(sh_raw)))
        pe_raw_diff_mean = np.mean(np.abs(np.diff(pe_raw)))

        # 2D 폭 변화량 (실제 움직임 기준)
        sh_w2d_range = sh_w2d.max() - sh_w2d.min()
        pe_w2d_range = pe_w2d.max() - pe_w2d.min()

        all_results.append({
            'video': stem,
            'sh_score': sh_score, 'pe_score': pe_score,
            'sh_diffs': sh_diffs, 'pe_diffs': pe_diffs,
            'sh_spike': sh_bad.sum(), 'pe_spike': pe_bad.sum(),
            'sh_vec_len': sh_vec_len, 'pe_vec_len': pe_vec_len,
            'sh_raw_noise': sh_raw_diff_mean, 'pe_raw_noise': pe_raw_diff_mean,
            'sh_w2d_range': sh_w2d_range, 'pe_w2d_range': pe_w2d_range,
            'n_frames': n,
        })

    # ── 1. 점수대별 정렬 + 상세 테이블 ──
    all_results.sort(key=lambda x: x['sh_score'] if x['sh_score'] else 0)

    print(f"\n{'='*110}")
    print(f"{'영상':<16} {'어깨':>6} {'골반':>6} │ {'벡터길이':>10} {'벡터길이':>10} │ {'raw노이즈':>9} {'raw노이즈':>9} │ {'2D폭변화':>8} {'2D폭변화':>8} │ {'스파이크':>6} {'스파이크':>6}")
    print(f"{'':16} {'점수':>6} {'점수':>6} │ {'어깨':>10} {'골반':>10} │ {'어깨':>9} {'골반':>9} │ {'어깨':>8} {'골반':>8} │ {'어깨':>6} {'골반':>6}")
    print(f"{'-'*110}")

    for r in all_results:
        sh = f"{r['sh_score']:.1f}°" if r['sh_score'] else "N/A"
        pe = f"{r['pe_score']:.1f}°" if r['pe_score'] else "N/A"
        print(f"{r['video']:<16} {sh:>6} {pe:>6} │ "
              f"{r['sh_vec_len']:.4f}     {r['pe_vec_len']:.4f}     │ "
              f"{r['sh_raw_noise']:.2f}°/f   {r['pe_raw_noise']:.2f}°/f   │ "
              f"{r['sh_w2d_range']:.0f}px    {r['pe_w2d_range']:.0f}px    │ "
              f"{r['sh_spike']:>5} {r['pe_spike']:>6}")

    # ── 2. 노이즈 기여도 분석 ──
    print(f"\n{'='*70}")
    print("노이즈 기여도 분석")
    print(f"{'='*70}")

    sh_scores = [r['sh_score'] for r in all_results if r['sh_score']]
    pe_scores = [r['pe_score'] for r in all_results if r['pe_score']]
    sh_noises = [r['sh_raw_noise'] for r in all_results if r['sh_score']]
    pe_noises = [r['pe_raw_noise'] for r in all_results if r['pe_score']]
    sh_w2ds = [r['sh_w2d_range'] for r in all_results if r['sh_score']]
    pe_w2ds = [r['pe_w2d_range'] for r in all_results if r['pe_score']]
    sh_vlens = [r['sh_vec_len'] for r in all_results if r['sh_score']]
    pe_vlens = [r['pe_vec_len'] for r in all_results if r['pe_score']]

    from scipy.stats import pearsonr
    r1, _ = pearsonr(sh_scores, sh_w2ds)
    r2, _ = pearsonr(pe_scores, pe_w2ds)
    r3, _ = pearsonr(sh_scores, sh_noises)
    r4, _ = pearsonr(pe_scores, pe_noises)

    print(f"  어깨 점수 vs 2D폭 변화량 상관: r={r1:.3f}  ← 높으면 실제 움직임 반영")
    print(f"  골반 점수 vs 2D폭 변화량 상관: r={r2:.3f}")
    print(f"  어깨 점수 vs raw 노이즈 상관:  r={r3:.3f}  ← 높으면 노이즈가 점수 지배")
    print(f"  골반 점수 vs raw 노이즈 상관:  r={r4:.3f}")

    # 골반 점수가 어깨보다 항상 높은 비율
    ratios = [r['pe_score']/r['sh_score'] for r in all_results if r['sh_score'] and r['pe_score']]
    print(f"\n  골반/어깨 점수 비율: 평균 {np.mean(ratios):.2f}배 (범위 {min(ratios):.2f}~{max(ratios):.2f})")
    print(f"  → 골반 벡터가 어깨의 {np.mean(pe_vlens)/np.mean(sh_vlens)*100:.0f}%로 짧아서 노이즈 증폭")

    # ── 3. 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    names = [r['video'].replace('인,인 ', '') for r in all_results]

    # (1) 어깨 vs 골반 점수 비교
    ax = axes[0][0]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, [r['sh_score'] for r in all_results], w, label='어깨', color='blue', alpha=0.7)
    ax.bar(x + w/2, [r['pe_score'] for r in all_results], w, label='골반', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('peak-to-peak 평균 (°)')
    ax.set_title('① 점수대별 어깨 vs 골반')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # (2) 개별 peak-to-peak 분포 (boxplot)
    ax = axes[0][1]
    sh_all_diffs = [r['sh_diffs'] for r in all_results]
    pe_all_diffs = [r['pe_diffs'] for r in all_results]
    positions_sh = np.arange(len(names)) * 2
    positions_pe = np.arange(len(names)) * 2 + 0.6
    bp1 = ax.boxplot(sh_all_diffs, positions=positions_sh, widths=0.5,
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp2 = ax.boxplot(pe_all_diffs, positions=positions_pe, widths=0.5,
                      patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    ax.set_xticks(np.arange(len(names)) * 2 + 0.3)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('peak-to-peak 각도 (°)')
    ax.set_title('② 개별 peak-to-peak 분포 (파랑=어깨, 빨강=골반)')
    ax.grid(True, alpha=0.3, axis='y')

    # (3) 점수 vs 2D 폭 변화 (실제 움직임 관계)
    ax = axes[1][0]
    ax.scatter([r['sh_w2d_range'] for r in all_results],
               [r['sh_score'] for r in all_results],
               color='blue', s=60, label=f'어깨 (r={r1:.2f})')
    ax.scatter([r['pe_w2d_range'] for r in all_results],
               [r['pe_score'] for r in all_results],
               color='red', s=60, label=f'골반 (r={r2:.2f})')
    for r in all_results:
        ax.annotate(r['video'].replace('인,인 ', ''),
                    (r['sh_w2d_range'], r['sh_score']), fontsize=6, color='blue')
    ax.set_xlabel('2D 폭 변화량 (px) — 실제 움직임')
    ax.set_ylabel('arctan2 점수 (°)')
    ax.set_title('③ arctan2 점수 vs 실제 움직임(2D폭)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (4) 점수 vs raw 노이즈
    ax = axes[1][1]
    ax.scatter([r['sh_raw_noise'] for r in all_results],
               [r['sh_score'] for r in all_results],
               color='blue', s=60, label=f'어깨 (r={r3:.2f})')
    ax.scatter([r['pe_raw_noise'] for r in all_results],
               [r['pe_score'] for r in all_results],
               color='red', s=60, label=f'골반 (r={r4:.2f})')
    ax.set_xlabel('raw 프레임간 노이즈 (°/frame)')
    ax.set_ylabel('arctan2 점수 (°)')
    ax.set_title('④ arctan2 점수 vs 노이즈 크기')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('output/graphs', exist_ok=True)
    plt.savefig('output/graphs/score_tier_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n그래프 저장: output/graphs/score_tier_analysis.png")
    plt.close()


if __name__ == "__main__":
    main()
