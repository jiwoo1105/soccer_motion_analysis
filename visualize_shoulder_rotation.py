"""
203 vs 219 프레임 어깨 회전 비교 (3D 시각화)
"""

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH = "input/in_in/인,인 9-1.MOV"
FRAME_A = 203
FRAME_B = 219

mp_pose = mp.solutions.pose


def get_shoulders(frame_idx):
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lms = results.pose_world_landmarks.landmark
    LS = np.array([lms[11].x, -lms[11].y, lms[11].z])  # Y 반전
    RS = np.array([lms[12].x, -lms[12].y, lms[12].z])
    return LS, RS


LS_A, RS_A = get_shoulders(FRAME_A)
LS_B, RS_B = get_shoulders(FRAME_B)

# 어깨 벡터 (3D)
vec_A = RS_A - LS_A
vec_B = RS_B - LS_B

# XZ 평면 투영 벡터 (Y 제거)
vxz_A = np.array([vec_A[0], 0, vec_A[2]])
vxz_B = np.array([vec_B[0], 0, vec_B[2]])

# 회전각 계산
cos_a = np.dot(vxz_A, vxz_B) / (np.linalg.norm(vxz_A) * np.linalg.norm(vxz_B))
angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

print(f"Frame {FRAME_A} 어깨벡터 (XZ): ({vec_A[0]:.3f}, {vec_A[2]:.3f})")
print(f"Frame {FRAME_B} 어깨벡터 (XZ): ({vec_B[0]:.3f}, {vec_B[2]:.3f})")
print(f"회전각: {angle:.2f}도")

# ── XZ 평면 실제 어깨 위치 ──────────────────────────────────────────
# 각 프레임 XZ 좌표 (Y 제거)
ls_A = np.array([LS_A[0], LS_A[2]])
rs_A = np.array([RS_A[0], RS_A[2]])
ls_B = np.array([LS_B[0], LS_B[2]])
rs_B = np.array([RS_B[0], RS_B[2]])

mid_A = (ls_A + rs_A) / 2
mid_B = (ls_B + rs_B) / 2

# 방향 단위벡터 (XZ)
nxz_A = np.array([vec_A[0], vec_A[2]])
nxz_A /= np.linalg.norm(nxz_A)
nxz_B = np.array([vec_B[0], vec_B[2]])
nxz_B /= np.linalg.norm(nxz_B)

# 두 직선 교차점 계산
# mid_A + t*nxz_A = mid_B + s*nxz_B
# [nxz_A | -nxz_B] [t, s]^T = mid_B - mid_A
M = np.array([[nxz_A[0], -nxz_B[0]],
              [nxz_A[1], -nxz_B[1]]])
rhs = mid_B - mid_A
try:
    ts = np.linalg.solve(M, rhs)
    intersect = mid_A + ts[0] * nxz_A
    has_intersect = True
except np.linalg.LinAlgError:
    has_intersect = False  # 평행한 경우

fig, ax = plt.subplots(figsize=(9, 9))

# 어깨선 연장해서 그리기
ext = 0.5  # 양쪽으로 연장 길이
for sign in [-1, 1]:
    ax.plot([ls_A[0] + sign*ext*nxz_A[0], rs_A[0] + sign*ext*nxz_A[0]],
            [ls_A[1] + sign*ext*nxz_A[1], rs_A[1] + sign*ext*nxz_A[1]],
            color='blue', lw=1.5, linestyle='--', alpha=0.4)
    ax.plot([ls_B[0] + sign*ext*nxz_B[0], rs_B[0] + sign*ext*nxz_B[0]],
            [ls_B[1] + sign*ext*nxz_B[1], rs_B[1] + sign*ext*nxz_B[1]],
            color='red', lw=1.5, linestyle='--', alpha=0.4)

# 실제 어깨선 (왼어깨 → 오른어깨)
ax.plot([ls_A[0], rs_A[0]], [ls_A[1], rs_A[1]], color='blue', lw=3, label=f'Frame {FRAME_A}')
ax.plot([ls_B[0], rs_B[0]], [ls_B[1], rs_B[1]], color='red',  lw=3, label=f'Frame {FRAME_B}')

# 어깨 점 표시
ax.scatter(*ls_A, color='blue', s=80, zorder=5)
ax.scatter(*rs_A, color='blue', s=80, zorder=5)
ax.scatter(*ls_B, color='red',  s=80, zorder=5)
ax.scatter(*rs_B, color='red',  s=80, zorder=5)
ax.text(ls_A[0]-0.03, ls_A[1]-0.03, f'L{FRAME_A}', fontsize=9, color='blue')
ax.text(rs_A[0]+0.01, rs_A[1]+0.01, f'R{FRAME_A}', fontsize=9, color='blue')
ax.text(ls_B[0]-0.03, ls_B[1]-0.03, f'L{FRAME_B}', fontsize=9, color='red')
ax.text(rs_B[0]+0.01, rs_B[1]+0.01, f'R{FRAME_B}', fontsize=9, color='red')

# 교차점 + 각도 호
if has_intersect:
    ax.scatter(*intersect, color='green', s=120, zorder=7)
    ax.text(intersect[0]+0.02, intersect[1]+0.02, '교차점', fontsize=10, color='green')

    theta_A = np.degrees(np.arctan2(nxz_A[1], nxz_A[0]))
    theta_B = np.degrees(np.arctan2(nxz_B[1], nxz_B[0]))
    thetas = np.linspace(np.radians(min(theta_A, theta_B)),
                         np.radians(max(theta_A, theta_B)), 50)
    r = 0.1
    ax.plot(intersect[0] + r*np.cos(thetas),
            intersect[1] + r*np.sin(thetas), color='gray', lw=2, linestyle='--')
    mid_theta = np.radians((theta_A + theta_B) / 2)
    ax.text(intersect[0] + 0.13*np.cos(mid_theta),
            intersect[1] + 0.13*np.sin(mid_theta),
            f'{angle:.1f}°', fontsize=13, color='gray', fontweight='bold', ha='center')

ax.set_xlabel('X  (좌우)', fontsize=12, fontweight='bold')
ax.set_ylabel('Z  (깊이)', fontsize=12, fontweight='bold')
ax.set_title(f'어깨 회전 비교 (XZ 위에서 내려다보기)\nFrame {FRAME_A} → {FRAME_B}  |  회전각: {angle:.1f}도', fontsize=13)
ax.axhline(0, color='lightgray', lw=0.8)
ax.axvline(0, color='lightgray', lw=0.8)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/graphs/shoulder_rotation_203_219.png", dpi=150, bbox_inches='tight')
plt.show()
