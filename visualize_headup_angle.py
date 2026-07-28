"""
특정 프레임의 헤드업 각도 시각화
- 실제 영상 프레임 위에 skeleton + head_vector + 각도 오버레이
- 3D 뷰에서 head_vector와 수직축 비교
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH  = "input/in_in/인,인 기준2.MOV"
FRAME_INDEX = 133
SAVE_PATH   = "output/graphs/headup_angle_기준2_frame133.png"

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── 프레임 추출 ────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"총 프레임: {total}")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"프레임 {FRAME_INDEX}를 읽을 수 없습니다.")
    exit(1)

# ── MediaPipe 포즈 추출 ────────────────────────────────────────────
with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                  min_detection_confidence=0.5) as pose:
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if not results.pose_world_landmarks:
    print("포즈 감지 실패")
    exit(1)

lms    = results.pose_world_landmarks.landmark
lms_2d = results.pose_landmarks.landmark
wl     = np.array([[lm.x, lm.y, lm.z] for lm in lms])

h, w = frame.shape[:2]

# ── 헤드업 각도 계산 ───────────────────────────────────────────────
eye_center      = (wl[2] + wl[5]) / 2
shoulder_center = (wl[11] + wl[12]) / 2
head_vector     = eye_center - shoulder_center

vertical = np.array([0, -1, 0])
cos_a    = np.dot(head_vector, vertical) / (np.linalg.norm(head_vector) + 1e-10)
head_angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

print(f"헤드업 각도: {head_angle:.2f}도")

# ── 2D 이미지 좌표 (픽셀) ──────────────────────────────────────────
eye_2d      = np.array([(lms_2d[2].x + lms_2d[5].x) / 2 * w,
                         (lms_2d[2].y + lms_2d[5].y) / 2 * h])
shoulder_2d = np.array([(lms_2d[11].x + lms_2d[12].x) / 2 * w,
                         (lms_2d[11].y + lms_2d[12].y) / 2 * h])

# ── 시각화 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])

# ── 왼쪽: 실제 영상 + 오버레이 ────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# skeleton 그리기
annotated = frame_rgb.copy()
mp_drawing.draw_landmarks(
    annotated,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
)

# head_vector 화살표 (어깨중앙 → 눈중앙)
cv2.arrowedLine(annotated,
                tuple(shoulder_2d.astype(int)),
                tuple(eye_2d.astype(int)),
                (255, 50, 50), 4, tipLength=0.2)

# 수직 기준선 (어깨중앙에서 위로)
vec_len = np.linalg.norm(eye_2d - shoulder_2d)
vertical_end = (shoulder_2d + np.array([0, -vec_len])).astype(int)
cv2.arrowedLine(annotated,
                tuple(shoulder_2d.astype(int)),
                tuple(vertical_end),
                (50, 50, 255), 2, tipLength=0.15)

# 각도 텍스트
mid = ((shoulder_2d + eye_2d) / 2).astype(int)
cv2.putText(annotated, f"{head_angle:.1f}deg",
            (mid[0] + 10, mid[1]), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 50, 50), 3, cv2.LINE_AA)

ax1.imshow(annotated)
ax1.set_title(f'Frame {FRAME_INDEX}  |  헤드업 각도: {head_angle:.1f}°\n'
              f'빨간 화살표=head_vector  파란 화살표=수직기준', fontsize=11)
ax1.axis('off')

# ── 오른쪽: 3D world_landmarks + head_vector ──────────────────────
ax2 = fig.add_subplot(gs[1], projection='3d')

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

xs = wl[:, 0]
ys = -wl[:, 1]  # Y 반전
zs = wl[:, 2]

for i, j in POSE_CONNECTIONS:
    ax2.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]],
             color='gray', linewidth=1.2, alpha=0.6)
ax2.scatter(xs, zs, ys, color='gray', s=20, alpha=0.6)

# head_vector (world 좌표, Y 반전 적용)
sc = np.array([shoulder_center[0], -shoulder_center[1], shoulder_center[2]])
ec = np.array([eye_center[0],      -eye_center[1],      eye_center[2]])
hv = ec - sc

ax2.quiver(sc[0], sc[2], sc[1], hv[0], hv[2], hv[1],
           color='red', linewidth=3, arrow_length_ratio=0.15, label='head_vector')

# 수직 기준 (같은 길이)
hv_len = np.linalg.norm(hv)
ax2.quiver(sc[0], sc[2], sc[1], 0, 0, hv_len,
           color='blue', linewidth=2, linestyle='dashed',
           arrow_length_ratio=0.15, label='수직기준', alpha=0.8)

# 어깨중앙, 눈중앙 점
ax2.scatter(*[sc[0]], *[sc[2]], *[sc[1]], color='red', s=80, zorder=5)
ax2.scatter(*[ec[0]], *[ec[2]], *[ec[1]], color='red', s=80, zorder=5)

ax2.set_xlabel('X (좌우)',  fontsize=9, labelpad=8)
ax2.set_ylabel('Z (깊이)', fontsize=9, labelpad=8)
ax2.set_zlabel('Y (상하)', fontsize=9, labelpad=8)
ax2.set_title(f'3D 뷰  |  각도: {head_angle:.1f}°\n빨강=head_vector  파랑=수직기준', fontsize=10)
ax2.legend(fontsize=9)
ax2.view_init(elev=10, azim=-80)

x_r = xs.max()-xs.min()
z_r = zs.max()-zs.min()
y_r = ys.max()-ys.min()
ax2.set_box_aspect([x_r, z_r, y_r])

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"저장: {SAVE_PATH}")
plt.show()
