"""
특정 프레임의 3D pose world landmarks 시각화 (인터랙티브)
사용법: python3.9 "visualize_3d_pose 9_1 62frame .py"
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 설정 ──────────────────────────────────────────
VIDEO_PATH  = "input/in_in/인,인 9-1.MOV"
FRAME_INDEX = 219
SAVE_PATH   = "output/graphs/3d_pose_frame219.png"
# ─────────────────────────────────────────────────

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"총 프레임: {total}")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"프레임 {FRAME_INDEX}를 읽을 수 없습니다.")
    exit(1)

print(f"프레임 {FRAME_INDEX} 로드 완료")

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
) as pose:
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if not results.pose_world_landmarks:
    print("포즈를 감지하지 못했습니다.")
    exit(1)

# MediaPipe landmarks → numpy (33, 3)
lms = results.pose_world_landmarks.landmark
world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lms])

xs = world_landmarks[:, 0]
ys = -world_landmarks[:, 1]  # Y 반전 (위가 양수)
zs = world_landmarks[:, 2]

print("포즈 감지 완료 → 3D 시각화 생성 중...")

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 연결선
for i, j in POSE_CONNECTIONS:
    ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]],
            color='black', linewidth=2)

# 키포인트
ax.scatter(xs, zs, ys, color='red', s=50, zorder=5)

# 축 라벨
ax.set_xlabel('X  (좌우)', fontsize=12, fontweight='bold', labelpad=12)
ax.set_ylabel('Z  (깊이)', fontsize=12, fontweight='bold', labelpad=12)
ax.set_zlabel('Y  (상하)', fontsize=12, fontweight='bold', labelpad=12)

# 실제 데이터 범위 비율로 박스 크기 설정 (왜곡 없음)
x_range = xs.max() - xs.min()
z_range = zs.max() - zs.min()
y_range = ys.max() - ys.min()
ax.set_box_aspect([x_range, z_range, y_range])

ax.set_title(f"{VIDEO_PATH}  |  Frame {FRAME_INDEX}", fontsize=11)
ax.view_init(elev=10, azim=-80)

plt.tight_layout()

if SAVE_PATH:
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"저장: {SAVE_PATH}")

# 인터랙티브 (마우스 드래그로 회전)
plt.show()
