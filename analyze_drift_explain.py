"""
드리프트 원인 설명용 깔끔한 그래프
1. 3D arctan2 vs 2D 폭 비교 → 드리프트 구간 표시
2. 드리프트 구간의 X좌표 변화 상세
3. 프레임 이미지로 "사람은 안 움직였다" 증명
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH = "input/in_in/인,인 3-1.MOV"

# ── 데이터 수집 ──
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

data = []
with mp_pose.Pose(
    static_image_mode=False, model_complexity=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
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
            data.append({
                'frame': idx,
                'sh_vx': wl[12].x - wl[11].x,
                'sh_vz': wl[12].z - wl[11].z,
                'sh_w2d': abs(il[12].x - il[11].x) * w_frame,
            })
        idx += 1
cap.release()

n = len(data)
frames = np.array([d['frame'] for d in data])
sh_vx = np.array([d['sh_vx'] for d in data])
sh_vz = np.array([d['sh_vz'] for d in data])
sh_w2d = np.array([d['sh_w2d'] for d in data])

# arctan2 + unwrap + 10° 제거 + savgol
raw = np.degrees(np.arctan2(sh_vz, sh_vx))
unwrapped = np.degrees(np.unwrap(np.radians(raw)))
cleaned = unwrapped.copy()
for i in range(1, len(cleaned)):
    if abs(cleaned[i] - cleaned[i-1]) > 10:
        cleaned[i] = cleaned[i-1]
win = min(41, n if n % 2 == 1 else n - 1)
angle_3d = savgol_filter(cleaned, win, 2)

# 2D 폭 smoothing
sh_w2d_smooth = savgol_filter(sh_w2d, win, 2)

# 드리프트 구간 찾기: 3D 변화 큰데 2D 변화 작은 구간
window = 30
drift_score = np.zeros(n)
for i in range(n - window):
    change_3d = abs(angle_3d[i + window] - angle_3d[i])
    change_2d = abs(sh_w2d_smooth[i + window] - sh_w2d_smooth[i])
    # 3D는 크게 변하는데 2D는 적게 변하면 → 드리프트
    if change_2d < 15:  # 2D 폭 변화 15px 이하 = 거의 안 움직임
        drift_score[i] = change_3d

# 가장 큰 드리프트 구간
max_idx = np.argmax(drift_score)
drift_start = max_idx
drift_end = min(max_idx + window, n - 1)

print(f"최대 드리프트 구간: frame {frames[drift_start]}~{frames[drift_end]}")
print(f"  3D 변화: {angle_3d[drift_start]:.1f}° → {angle_3d[drift_end]:.1f}° = {abs(angle_3d[drift_end]-angle_3d[drift_start]):.1f}°")
print(f"  2D 폭 변화: {sh_w2d_smooth[drift_start]:.1f}px → {sh_w2d_smooth[drift_end]:.1f}px = {abs(sh_w2d_smooth[drift_end]-sh_w2d_smooth[drift_start]):.1f}px")

# ── 프레임 이미지 캡처 (드리프트 구간 시작, 중간, 끝) ──
sample_points = [drift_start, (drift_start + drift_end) // 2, drift_end]
cap = cv2.VideoCapture(VIDEO_PATH)
sample_imgs = {}
for si in sample_points:
    fn = data[si]['frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if ret:
        sample_imgs[si] = frame.copy()
cap.release()

# ── 시각화 ──
fig = plt.figure(figsize=(20, 22))
gs = fig.add_gridspec(4, 3, height_ratios=[3, 3, 3, 2], hspace=0.35, wspace=0.3)

# ───────────────────────────────────────────
# 1행: 3D 각도 vs 2D 폭 비교 + 드리프트 구간 표시
# ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.axvspan(frames[drift_start], frames[drift_end], alpha=0.2, color='red',
            label=f'드리프트 구간 (F{frames[drift_start]}~F{frames[drift_end]})')

ax1.plot(frames, angle_3d, 'b-', lw=2, label='3D arctan2 (10°+savgol)')
ax1.set_ylabel('3D 각도 (°)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

ax1b = ax1.twinx()
ax1b.plot(frames, sh_w2d_smooth, 'r-', lw=2, label='2D 어깨 폭 (px)')
ax1b.set_ylabel('2D 어깨 폭 (px)', color='red', fontsize=12)
ax1b.tick_params(axis='y', labelcolor='red')

# 드리프트 구간 수치 표시
mid = (drift_start + drift_end) // 2
ax1.annotate(
    f'3D: {abs(angle_3d[drift_end]-angle_3d[drift_start]):.0f}° 변화\n2D: {abs(sh_w2d_smooth[drift_end]-sh_w2d_smooth[drift_start]):.0f}px 변화\n→ 사람은 안 움직였는데\n   3D만 크게 변함 = 드리프트',
    xy=(frames[mid], angle_3d[mid]),
    xytext=(frames[mid] + 40, angle_3d[mid] - 30),
    fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='orange', lw=2),
    arrowprops=dict(arrowstyle='->', color='orange', lw=2))

lines1 = ax1.get_lines() + ax1b.get_lines()
patches = ax1.patches
ax1.legend(lines1, [l.get_label() for l in lines1], loc='upper right', fontsize=10)
ax1.set_title('① 3D 각도 vs 2D 폭 — 빨간 영역이 드리프트 (3D만 변하고 2D는 안 변하는 구간)',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# ───────────────────────────────────────────
# 2행: 드리프트 구간의 X, Z 좌표 변화 상세
# ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
pad = 10
view_start = max(0, drift_start - pad)
view_end = min(n, drift_end + pad)
view = slice(view_start, view_end)

ax2.axvspan(frames[drift_start], frames[drift_end], alpha=0.15, color='red')

ax2.plot(frames[view], sh_vx[view], 'b-', lw=2.5, marker='o', markersize=3, label='벡터 X (문제)')
ax2.plot(frames[view], sh_vz[view], 'r-', lw=2.5, marker='s', markersize=3, label='벡터 Z (안정)')
ax2.axhline(0, color='gray', lw=1, ls='--')

# X가 0 넘는 지점 표시
zero_cross = None
for i in range(view_start, view_end - 1):
    if sh_vx[i] * sh_vx[i+1] < 0:  # 부호 바뀜
        zero_cross = i
        break

if zero_cross:
    ax2.axvline(frames[zero_cross], color='purple', lw=2, ls='--', alpha=0.7)
    ax2.annotate('X가 0을 넘음\n→ arctan2 각도 급변',
                 xy=(frames[zero_cross], 0),
                 xytext=(frames[zero_cross] + 5, 0.15),
                 fontsize=11, fontweight='bold', color='purple',
                 bbox=dict(boxstyle='round', fc='lavender', ec='purple'),
                 arrowprops=dict(arrowstyle='->', color='purple', lw=2))

ax2.set_ylabel('벡터 값', fontsize=12)
ax2.set_xlabel('Frame', fontsize=12)
ax2.set_title('② 드리프트 구간 상세 — X(파란)가 천천히 양수→음수로 넘어감, Z(빨간)는 안정',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# ───────────────────────────────────────────
# 3행: 왜 이런 일이 생기는지 (단안 카메라 한계)
# ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])

# 누적 X 변화 그래프
cumulative_x = np.cumsum(np.diff(sh_vx[view_start:view_end]))
cumulative_x = np.concatenate([[0], cumulative_x])
ax3.axvspan(frames[drift_start], frames[drift_end], alpha=0.15, color='red')
ax3.fill_between(frames[view], cumulative_x, 0, where=cumulative_x < 0,
                 alpha=0.3, color='blue', label='X 감소 누적')
ax3.fill_between(frames[view], cumulative_x, 0, where=cumulative_x > 0,
                 alpha=0.3, color='red', label='X 증가 누적')
ax3.plot(frames[view], cumulative_x, 'k-', lw=2)
ax3.axhline(0, color='gray', lw=1, ls='--')

ax3.set_ylabel('X 변화 누적량', fontsize=12)
ax3.set_xlabel('Frame', fontsize=12)
ax3.set_title('③ X 변화 누적 — 매 프레임 작은 오차가 한 방향으로 쌓임 = 드리프트',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# 설명 박스
ax3.text(0.02, 0.95,
         '원인: 단안 카메라(1대)로 뒤에서 촬영\n'
         '→ 양 어깨가 비슷한 깊이에 있음\n'
         '→ Mediapipe가 X(앞뒤)를 매 프레임 추측\n'
         '→ 추측이 한쪽으로 치우쳐 누적\n'
         '→ 30프레임 뒤 108° 오차',
         transform=ax3.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', fc='lightyellow', ec='orange', lw=2))

# ───────────────────────────────────────────
# 4행: 프레임 이미지 (사람은 안 움직였다 증명)
# ───────────────────────────────────────────
labels = ['드리프트 시작', '드리프트 중간', '드리프트 끝']
for i, (si, label) in enumerate(zip(sample_points, labels)):
    ax = fig.add_subplot(gs[3, i])
    if si in sample_imgs:
        ax.imshow(cv2.cvtColor(sample_imgs[si], cv2.COLOR_BGR2RGB))
    fn = data[si]['frame']
    angle_val = angle_3d[si]
    w2d_val = sh_w2d_smooth[si]
    ax.set_title(f'{label}\nF{fn} | 3D={angle_val:.0f}° | 2D폭={w2d_val:.0f}px',
                 fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Mediapipe 드리프트 원인 분석 — 인,인 3-1',
             fontsize=18, fontweight='bold', y=0.98)

os.makedirs('output/graphs', exist_ok=True)
plt.savefig('output/graphs/drift_explanation.png', dpi=150, bbox_inches='tight')
print(f"\n저장: output/graphs/drift_explanation.png")
plt.close()
