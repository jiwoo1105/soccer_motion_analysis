"""
arctan2 vs 2D 폭 방식 직접 비교
같은 영상에서 두 방식이 같은 패턴을 보이면 arctan2도 신뢰 가능
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH = "input/in_in/인,인 3-1.MOV"

mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 데이터 수집
sh_vx, sh_vz = [], []
pe_vx, pe_vz = [], []
sh_width_2d, pe_width_2d = [], []  # 2D 폭 (px)
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
sh_deg = np.degrees(np.arctan2(sh_vz, sh_vx))
pe_deg = np.degrees(np.arctan2(pe_vz, pe_vx))
sh_w2d = np.array(sh_width_2d)
pe_w2d = np.array(pe_width_2d)

# ── 시각화: 둘을 겹쳐서 비교 ──
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# 어깨
ax1 = axes[0]
color1 = 'blue'
ax1.set_ylabel('arctan2 각도 (°)', color=color1)
ax1.plot(frames, sh_deg, color=color1, alpha=0.7, label='arctan2 (3D)')
ax1.tick_params(axis='y', labelcolor=color1)

ax1b = ax1.twinx()
color2 = 'red'
ax1b.set_ylabel('2D 폭 (px)', color=color2)
ax1b.plot(frames, sh_w2d, color=color2, alpha=0.7, label='2D 폭')
ax1b.tick_params(axis='y', labelcolor=color2)

ax1.set_title('어깨: arctan2(3D) vs 2D 폭 — 패턴이 같으면 arctan2 신뢰 가능')
lines1 = ax1.get_lines() + ax1b.get_lines()
ax1.legend(lines1, [l.get_label() for l in lines1], loc='upper right')
ax1.grid(True, alpha=0.3)

# 골반
ax2 = axes[1]
ax2.set_ylabel('arctan2 각도 (°)', color=color1)
ax2.plot(frames, pe_deg, color=color1, alpha=0.7, label='arctan2 (3D)')
ax2.tick_params(axis='y', labelcolor=color1)

ax2b = ax2.twinx()
ax2b.set_ylabel('2D 폭 (px)', color=color2)
ax2b.plot(frames, pe_w2d, color=color2, alpha=0.7, label='2D 폭')
ax2b.tick_params(axis='y', labelcolor=color2)

ax2.set_title('골반: arctan2(3D) vs 2D 폭 — 패턴이 같으면 arctan2 신뢰 가능')
lines2 = ax2.get_lines() + ax2b.get_lines()
ax2.legend(lines2, [l.get_label() for l in lines2], loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Frame')

plt.tight_layout()
plt.savefig('output/graphs/compare_arctan2_vs_2d.png', dpi=150, bbox_inches='tight')
print("저장: output/graphs/compare_arctan2_vs_2d.png")

# 상관계수
from scipy.stats import pearsonr
# 2D 폭이 클수록 정면 → arctan2 90° 근처, 폭 작을수록 옆 → 0° or 180°
# 반전 관계일 수 있으니 절대값 상관
r_sh, p_sh = pearsonr(sh_deg, sh_w2d)
r_pe, p_pe = pearsonr(pe_deg, pe_w2d)
print(f"\n상관계수 (arctan2 vs 2D폭):")
print(f"  어깨: r={r_sh:.3f} (p={p_sh:.4f})")
print(f"  골반: r={r_pe:.3f} (p={p_pe:.4f})")

plt.close()
