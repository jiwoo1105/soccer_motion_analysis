"""
특정 프레임의 상체각도 시각화
무릎 - 엉덩이(꼭지점) - 어깨 각도
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
FRAME_INDEX = 178
SAVE_PATH   = f"output/graphs/trunk_angle_기준2_frame{FRAME_INDEX}.png"

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── 프레임 추출 ────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"프레임 {FRAME_INDEX}를 읽을 수 없습니다.")
    exit(1)

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

# ── 상체각도 계산 ──────────────────────────────────────────────────
def calc_angle(p1, p2, p3):
    """p1-p2-p3, p2가 꼭지점"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

left_angle  = calc_angle(wl[25], wl[23], wl[11])   # 왼무릎-왼힙-왼어깨
right_angle = calc_angle(wl[26], wl[24], wl[12])   # 오른무릎-오른힙-오른어깨
trunk_angle = (left_angle + right_angle) / 2

print(f"왼쪽 상체각도:  {left_angle:.2f}도")
print(f"오른쪽 상체각도: {right_angle:.2f}도")
print(f"평균 상체각도:  {trunk_angle:.2f}도")

# ── 2D 픽셀 좌표 추출 ─────────────────────────────────────────────
def px(idx):
    return np.array([lms_2d[idx].x * w, lms_2d[idx].y * h])

L_knee = px(25); L_hip = px(23); L_sh = px(11)
R_knee = px(26); R_hip = px(24); R_sh = px(12)

# ── 시각화 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])

# ── 왼쪽: 실제 영상 오버레이 ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()

mp_drawing.draw_landmarks(
    annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=2),
)

# 왼쪽: 무릎→힙→어깨 선 (파랑)
for a, b in [(L_knee, L_hip), (L_hip, L_sh)]:
    cv2.line(annotated, tuple(a.astype(int)), tuple(b.astype(int)), (50, 100, 255), 4)
for pt in [L_knee, L_hip, L_sh]:
    cv2.circle(annotated, tuple(pt.astype(int)), 8, (50, 100, 255), -1)
cv2.putText(annotated, f"L:{left_angle:.1f}",
            tuple((L_hip + np.array([-80, 0])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 100, 255), 2)

# 오른쪽: 무릎→힙→어깨 선 (빨강)
for a, b in [(R_knee, R_hip), (R_hip, R_sh)]:
    cv2.line(annotated, tuple(a.astype(int)), tuple(b.astype(int)), (255, 80, 50), 4)
for pt in [R_knee, R_hip, R_sh]:
    cv2.circle(annotated, tuple(pt.astype(int)), 8, (255, 80, 50), -1)
cv2.putText(annotated, f"R:{right_angle:.1f}",
            tuple((R_hip + np.array([10, 0])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 50), 2)

# 평균 각도 상단 표시
cv2.putText(annotated, f"Trunk avg: {trunk_angle:.1f}deg",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 50), 3)

ax1.imshow(annotated)
ax1.set_title(f'Frame {FRAME_INDEX}  |  상체각도 평균: {trunk_angle:.1f}°\n'
              f'파랑=왼쪽({left_angle:.1f}°)  빨강=오른쪽({right_angle:.1f}°)', fontsize=11)
ax1.axis('off')

# ── 오른쪽: 3D world_landmarks ────────────────────────────────────
ax2 = fig.add_subplot(gs[1], projection='3d')

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

xs = wl[:, 0]; ys = -wl[:, 1]; zs = wl[:, 2]

for i, j in POSE_CONNECTIONS:
    ax2.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]],
             color='lightgray', linewidth=1.2, alpha=0.5)
ax2.scatter(xs, zs, ys, color='lightgray', s=15, alpha=0.5)

def pt3(idx):
    return np.array([wl[idx][0], wl[idx][2], -wl[idx][1]])

# 왼쪽 무릎-힙-어깨 (파랑)
for a, b in [(25, 23), (23, 11)]:
    p1, p2 = pt3(a), pt3(b)
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
             color='royalblue', linewidth=4)
for idx in [25, 23, 11]:
    p = pt3(idx)
    ax2.scatter(*[p[0]], *[p[1]], *[p[2]], color='royalblue', s=80, zorder=5)

# 오른쪽 무릎-힙-어깨 (빨강)
for a, b in [(26, 24), (24, 12)]:
    p1, p2 = pt3(a), pt3(b)
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
             color='tomato', linewidth=4)
for idx in [26, 24, 12]:
    p = pt3(idx)
    ax2.scatter(*[p[0]], *[p[1]], *[p[2]], color='tomato', s=80, zorder=5)

# 꼭지점(힙) 라벨
for idx, label in [(23, 'L힙'), (24, 'R힙')]:
    p = pt3(idx)
    ax2.text(p[0]+0.02, p[1], p[2], label, fontsize=9)

# 각도 호(arc) + 각도값 표시
def draw_angle_arc(ax, hip_idx, knee_idx, sh_idx, angle, color, r=0.1):
    hip = pt3(hip_idx)
    v1  = pt3(knee_idx) - hip;  v1 /= np.linalg.norm(v1)
    v2  = pt3(sh_idx)   - hip;  v2 /= np.linalg.norm(v2)
    # v1 → v2 사이 호: slerp로 샘플링
    arc_pts = []
    for t in np.linspace(0, 1, 30):
        v = v1 * (1 - t) + v2 * t
        v /= np.linalg.norm(v)
        arc_pts.append(hip + r * v)
    arc_pts = np.array(arc_pts)
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
            color=color, linewidth=2.5, zorder=6)
    # 각도 텍스트 (호 중간점 근처)
    mid = arc_pts[len(arc_pts)//2]
    offset = (mid - hip) * 1.8
    ax.text(hip[0]+offset[0], hip[1]+offset[1], hip[2]+offset[2],
            f'{angle:.1f}°', fontsize=11, color=color, fontweight='bold')

draw_angle_arc(ax2, 23, 25, 11, left_angle,  'royalblue')
draw_angle_arc(ax2, 24, 26, 12, right_angle, 'tomato')

ax2.set_xlabel('X (좌우)',  fontsize=9, labelpad=8)
ax2.set_ylabel('Z (깊이)', fontsize=9, labelpad=8)
ax2.set_zlabel('Y (상하)', fontsize=9, labelpad=8)
ax2.set_title(f'3D 뷰  |  상체각도: {trunk_angle:.1f}°\n'
              f'파랑=왼쪽  빨강=오른쪽  꼭지점=엉덩이', fontsize=10)

x_r = xs.max()-xs.min(); z_r = zs.max()-zs.min(); y_r = ys.max()-ys.min()
ax2.set_box_aspect([x_r, z_r, y_r])
ax2.view_init(elev=10, azim=-80)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"저장: {SAVE_PATH}")
plt.show()
