"""
arctan2(Z, X) 방식 살리기: 다양한 후처리 비교
- unwrap + 강한 savgol
- unwrap + moving median (큰 윈도우)
- unwrap + lowpass (butterworth)
모든 결과를 2D 폭(ground truth)과 비교
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
from scipy.ndimage import uniform_filter1d

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
sh_vx = np.array(sh_vx)
sh_vz = np.array(sh_vz)
pe_vx = np.array(pe_vx)
pe_vz = np.array(pe_vz)
sh_w2d = np.array(sh_width_2d)
pe_w2d = np.array(pe_width_2d)
n = len(frames)

# Raw arctan2
sh_raw = np.degrees(np.arctan2(sh_vz, sh_vx))
pe_raw = np.degrees(np.arctan2(pe_vz, pe_vx))

# ── 방법 1: np.unwrap + savgol (큰 윈도우) ──
sh_unwrap = np.unwrap(np.radians(sh_raw))
pe_unwrap = np.unwrap(np.radians(pe_raw))

for win_size in [21, 41, 61]:
    win = min(win_size, n if n % 2 == 1 else n - 1)
    if win >= 3:
        sh_savgol = np.degrees(savgol_filter(sh_unwrap, win, 2))
        pe_savgol = np.degrees(savgol_filter(pe_unwrap, win, 2))

sh_sav21 = np.degrees(savgol_filter(sh_unwrap, min(21, n if n%2==1 else n-1), 2))
sh_sav41 = np.degrees(savgol_filter(sh_unwrap, min(41, n if n%2==1 else n-1), 2))
sh_sav61 = np.degrees(savgol_filter(sh_unwrap, min(61, n if n%2==1 else n-1), 2))

pe_sav21 = np.degrees(savgol_filter(pe_unwrap, min(21, n if n%2==1 else n-1), 2))
pe_sav41 = np.degrees(savgol_filter(pe_unwrap, min(41, n if n%2==1 else n-1), 2))
pe_sav61 = np.degrees(savgol_filter(pe_unwrap, min(61, n if n%2==1 else n-1), 2))

# ── 방법 2: moving median (큰 윈도우) ──
med_win = min(31, n if n % 2 == 1 else n - 1)
sh_med = np.degrees(medfilt(sh_unwrap, med_win))
pe_med = np.degrees(medfilt(pe_unwrap, med_win))

# ── 방법 3: butterworth lowpass ──
fps = 30
cutoff = 1.5  # 1.5Hz (0.67초 주기 이상만 통과)
b, a = butter(2, cutoff / (fps/2), btype='low')
sh_butter = np.degrees(filtfilt(b, a, sh_unwrap))
pe_butter = np.degrees(filtfilt(b, a, pe_unwrap))

# ── 상관계수 계산 (2D 폭과) ──
from scipy.stats import pearsonr

methods = {
    'raw': (sh_raw, pe_raw),
    'unwrap+sav21': (sh_sav21, pe_sav21),
    'unwrap+sav41': (sh_sav41, pe_sav41),
    'unwrap+sav61': (sh_sav61, pe_sav61),
    'unwrap+median31': (sh_med, pe_med),
    'unwrap+butter1.5Hz': (sh_butter, pe_butter),
}

print(f"{'방법':<22} {'어깨 r':>8} {'어깨 range':>12} {'골반 r':>8} {'골반 range':>12}")
print("-" * 65)
for name, (sh, pe) in methods.items():
    r_sh, _ = pearsonr(sh, sh_w2d)
    r_pe, _ = pearsonr(pe, pe_w2d)
    sh_range = sh.max() - sh.min()
    pe_range = pe.max() - pe.min()
    print(f"{name:<22} {r_sh:>8.3f} {sh_range:>10.1f}° {r_pe:>8.3f} {pe_range:>10.1f}°")

# ── 시각화: 가장 좋은 방법들 비교 ──
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# 어깨
ax1 = axes[0]
ax1.plot(frames, sh_raw, 'b-', alpha=0.15, label='raw')
ax1.plot(frames, sh_sav41, 'b-', lw=1.5, label='unwrap+savgol41')
ax1.plot(frames, sh_butter, 'c-', lw=1.5, label='unwrap+butter1.5Hz')
ax1b = ax1.twinx()
ax1b.plot(frames, sh_w2d, 'r-', alpha=0.7, lw=1.5, label='2D 폭 (기준)')
ax1b.set_ylabel('2D 폭 (px)', color='red')
ax1.set_ylabel('arctan2 각도 (°)', color='blue')
ax1.set_title('어깨: arctan2 후처리 vs 2D 폭')
lines1 = ax1.get_lines() + ax1b.get_lines()
ax1.legend(lines1, [l.get_label() for l in lines1], loc='upper right')
ax1.grid(True, alpha=0.3)

# 골반
ax2 = axes[1]
ax2.plot(frames, pe_raw, 'b-', alpha=0.15, label='raw')
ax2.plot(frames, pe_sav41, 'b-', lw=1.5, label='unwrap+savgol41')
ax2.plot(frames, pe_butter, 'c-', lw=1.5, label='unwrap+butter1.5Hz')
ax2b = ax2.twinx()
ax2b.plot(frames, pe_w2d, 'r-', alpha=0.7, lw=1.5, label='2D 폭 (기준)')
ax2b.set_ylabel('2D 폭 (px)', color='red')
ax2.set_ylabel('arctan2 각도 (°)', color='blue')
ax2.set_title('골반: arctan2 후처리 vs 2D 폭')
ax2.set_xlabel('Frame')
lines2 = ax2.get_lines() + ax2b.get_lines()
ax2.legend(lines2, [l.get_label() for l in lines2], loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/graphs/arctan2_fix_compare.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/arctan2_fix_compare.png")
plt.close()
