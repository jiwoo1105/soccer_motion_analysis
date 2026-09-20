"""
10° 스파이크 제거 → savgol 스무딩 조합 테스트
2D 폭과 비교
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH = "input/in_in/인,인 3-1.MOV"

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

sh_vx, sh_vz = [], []
pe_vx, pe_vz = [], []
sh_width_2d, pe_width_2d = [], []
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
        if results.pose_world_landmarks and results.pose_landmarks:
            wl = results.pose_world_landmarks.landmark
            il = results.pose_landmarks.landmark
            sh_vx.append(wl[12].x - wl[11].x)
            sh_vz.append(wl[12].z - wl[11].z)
            pe_vx.append(wl[24].x - wl[23].x)
            pe_vz.append(wl[24].z - wl[23].z)
            sh_width_2d.append(abs(il[12].x - il[11].x) * w_frame)
            pe_width_2d.append(abs(il[24].x - il[23].x) * w_frame)
            valid_frames.append(idx)
        idx += 1
cap.release()

frames = np.array(valid_frames)
n = len(frames)
sh_vx = np.array(sh_vx); sh_vz = np.array(sh_vz)
pe_vx = np.array(pe_vx); pe_vz = np.array(pe_vz)
sh_w2d = np.array(sh_width_2d)
pe_w2d = np.array(pe_width_2d)

# Raw arctan2
sh_raw = np.degrees(np.arctan2(sh_vz, sh_vx))
pe_raw = np.degrees(np.arctan2(pe_vz, pe_vx))

# unwrap (180° 점프 제거)
sh_unwrap = np.degrees(np.unwrap(np.radians(sh_raw)))
pe_unwrap = np.degrees(np.unwrap(np.radians(pe_raw)))


def spike_remove_then_smooth(angles, threshold, savgol_win):
    """1단계: threshold 이상 점프 프레임 제거 + 보간, 2단계: savgol"""
    cleaned = angles.copy()
    bad_mask = np.zeros(len(angles), dtype=bool)

    for i in range(1, len(angles)):
        if abs(cleaned[i] - cleaned[i-1]) > threshold:
            bad_mask[i] = True

    good_idx = np.where(~bad_mask)[0]
    if len(good_idx) >= 2:
        cleaned[bad_mask] = np.interp(
            np.where(bad_mask)[0], good_idx, cleaned[good_idx]
        )

    # savgol
    win = min(savgol_win, len(cleaned) if len(cleaned) % 2 == 1 else len(cleaned) - 1)
    if win >= 3:
        smoothed = savgol_filter(cleaned, win, 2)
    else:
        smoothed = cleaned

    return cleaned, smoothed, bad_mask


# 여러 조합 테스트
configs = [
    (5, 11), (5, 21), (5, 41),
    (10, 11), (10, 21), (10, 41),
    (15, 21), (15, 41),
]

print(f"{'설정':<20} {'어깨 r':>8} {'어깨 range':>12} {'골반 r':>8} {'골반 range':>12} {'어깨 제거':>8} {'골반 제거':>8}")
print("-" * 85)

best_sh_r = -1
best_pe_r = -1
best_sh_config = None
best_pe_config = None
best_sh_result = None
best_pe_result = None

for thresh, swin in configs:
    _, sh_result, sh_bad = spike_remove_then_smooth(sh_unwrap, thresh, swin)
    _, pe_result, pe_bad = spike_remove_then_smooth(pe_unwrap, thresh, swin)

    r_sh, _ = pearsonr(sh_result, sh_w2d)
    r_pe, _ = pearsonr(pe_result, pe_w2d)
    sh_range = sh_result.max() - sh_result.min()
    pe_range = pe_result.max() - pe_result.min()

    name = f"t{thresh}°+sav{swin}"
    print(f"{name:<20} {r_sh:>8.3f} {sh_range:>10.1f}° {r_pe:>8.3f} {pe_range:>10.1f}° {sh_bad.sum():>8} {pe_bad.sum():>8}")

    if r_sh > best_sh_r:
        best_sh_r = r_sh
        best_sh_config = name
        best_sh_result = sh_result
    if r_pe > best_pe_r:
        best_pe_r = r_pe
        best_pe_config = name
        best_pe_result = pe_result

# raw 비교용
r_sh_raw, _ = pearsonr(sh_raw, sh_w2d)
r_pe_raw, _ = pearsonr(pe_raw, pe_w2d)
print(f"\n{'raw (비교용)':<20} {r_sh_raw:>8.3f} {sh_raw.max()-sh_raw.min():>10.1f}° {r_pe_raw:>8.3f} {pe_raw.max()-pe_raw.min():>10.1f}°")

print(f"\n최적: 어깨={best_sh_config} (r={best_sh_r:.3f}), 골반={best_pe_config} (r={best_pe_r:.3f})")

# ── 시각화: 최적 결과 vs 2D 폭 ──
# 가장 좋은 설정 하나로 통일 (t10+sav41 사용)
_, sh_final, sh_bad = spike_remove_then_smooth(sh_unwrap, 10, 41)
_, pe_final, pe_bad = spike_remove_then_smooth(pe_unwrap, 10, 41)

fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# 어깨
ax1 = axes[0]
ax1.plot(frames, sh_unwrap, 'b-', alpha=0.15, label='raw (unwrap)')
ax1.plot(frames, sh_final, 'b-', lw=2, label='10° 제거 + savgol41')
ax1.scatter(frames[sh_bad], sh_unwrap[sh_bad], color='blue', s=30, marker='x', zorder=5, label=f'제거됨 ({sh_bad.sum()}개)')
ax1b = ax1.twinx()
ax1b.plot(frames, sh_w2d, 'r-', alpha=0.7, lw=1.5, label='2D 폭 (기준)')
ax1b.set_ylabel('2D 폭 (px)', color='red')
ax1.set_ylabel('arctan2 각도 (°)', color='blue')
r_sh, _ = pearsonr(sh_final, sh_w2d)
ax1.set_title(f'어깨: 10° 제거 + savgol41 (r={r_sh:.3f}, range={sh_final.max()-sh_final.min():.1f}°)')
lines = ax1.get_lines() + ax1b.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
ax1.grid(True, alpha=0.3)

# 골반
ax2 = axes[1]
ax2.plot(frames, pe_unwrap, 'b-', alpha=0.15, label='raw (unwrap)')
ax2.plot(frames, pe_final, 'b-', lw=2, label='10° 제거 + savgol41')
ax2.scatter(frames[pe_bad], pe_unwrap[pe_bad], color='blue', s=30, marker='x', zorder=5, label=f'제거됨 ({pe_bad.sum()}개)')
ax2b = ax2.twinx()
ax2b.plot(frames, pe_w2d, 'r-', alpha=0.7, lw=1.5, label='2D 폭 (기준)')
ax2b.set_ylabel('2D 폭 (px)', color='red')
ax2.set_ylabel('arctan2 각도 (°)', color='blue')
r_pe, _ = pearsonr(pe_final, pe_w2d)
ax2.set_title(f'골반: 10° 제거 + savgol41 (r={r_pe:.3f}, range={pe_final.max()-pe_final.min():.1f}°)')
ax2.set_xlabel('Frame')
lines2 = ax2.get_lines() + ax2b.get_lines()
ax2.legend(lines2, [l.get_label() for l in lines2], loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/graphs/spike_then_smooth.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/spike_then_smooth.png")
plt.close()
