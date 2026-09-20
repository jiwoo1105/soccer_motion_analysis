"""
X 좌표 복원 실험: Z(안정) + 기하학적 제약조건으로 X 재계산
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
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 데이터 수집
sh_vx_raw, sh_vz_raw = [], []
pe_vx_raw, pe_vz_raw = [], []
sh_img_lx, sh_img_rx = [], []  # image landmarks X (좌우 판별용)
pe_img_lx, pe_img_rx = [], []
valid_frames = []

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_world_landmarks and results.pose_landmarks:
            wl = results.pose_world_landmarks.landmark
            il = results.pose_landmarks.landmark  # image landmarks

            # world: 어깨 12(R) - 11(L), 골반 24(R) - 23(L)
            sh_vx_raw.append(wl[12].x - wl[11].x)
            sh_vz_raw.append(wl[12].z - wl[11].z)
            pe_vx_raw.append(wl[24].x - wl[23].x)
            pe_vz_raw.append(wl[24].z - wl[23].z)

            # image: 좌우 판별용
            sh_img_lx.append(il[11].x)
            sh_img_rx.append(il[12].x)
            pe_img_lx.append(il[23].x)
            pe_img_rx.append(il[24].x)

            valid_frames.append(frame_idx)
        frame_idx += 1

cap.release()

sh_vx = np.array(sh_vx_raw)
sh_vz = np.array(sh_vz_raw)
pe_vx = np.array(pe_vx_raw)
pe_vz = np.array(pe_vz_raw)
frames = np.array(valid_frames)
n = len(frames)

print(f"유효 프레임: {n}/{total}")

# ── 방법: X 기하학적 복원 ──
# W = sqrt(X² + Z²) → 프레임마다 거의 일정해야 함
sh_W = np.sqrt(sh_vx**2 + sh_vz**2)
pe_W = np.sqrt(pe_vx**2 + pe_vz**2)

print(f"\n어깨 벡터 길이(W): mean={sh_W.mean():.4f}, std={sh_W.std():.4f}, range=[{sh_W.min():.4f}, {sh_W.max():.4f}]")
print(f"골반 벡터 길이(W): mean={pe_W.mean():.4f}, std={pe_W.std():.4f}, range=[{pe_W.min():.4f}, {pe_W.max():.4f}]")

# W_median 사용
sh_W_med = np.median(sh_W)
pe_W_med = np.median(pe_W)

# X 복원: X = sqrt(W² - Z²), 부호는 image landmarks에서
sh_x_sq = np.clip(sh_W_med**2 - sh_vz**2, 0, None)
sh_vx_recon = np.sqrt(sh_x_sq)
# 부호: image에서 R.x > L.x이면 X 양수 (오른쪽이 오른쪽에 있으면)
sh_sign = np.sign(np.array(sh_img_rx) - np.array(sh_img_lx))
sh_vx_recon *= sh_sign

pe_x_sq = np.clip(pe_W_med**2 - pe_vz**2, 0, None)
pe_vx_recon = np.sqrt(pe_x_sq)
pe_sign = np.sign(np.array(pe_img_rx) - np.array(pe_img_lx))
pe_vx_recon *= pe_sign

# arctan2 비교
sh_deg_raw = np.degrees(np.arctan2(sh_vz, sh_vx))
sh_deg_recon = np.degrees(np.arctan2(sh_vz, sh_vx_recon))
pe_deg_raw = np.degrees(np.arctan2(pe_vz, pe_vx))
pe_deg_recon = np.degrees(np.arctan2(pe_vz, pe_vx_recon))

# 변화량 비교
sh_diff_raw = np.abs(np.diff(sh_deg_raw))
sh_diff_recon = np.abs(np.diff(sh_deg_recon))
pe_diff_raw = np.abs(np.diff(pe_deg_raw))
pe_diff_recon = np.abs(np.diff(pe_deg_recon))

print(f"\n=== 어깨 비교 ===")
print(f"  raw   range: {sh_deg_raw.max()-sh_deg_raw.min():.1f}°, 평균변화: {sh_diff_raw.mean():.2f}°/f")
print(f"  복원  range: {sh_deg_recon.max()-sh_deg_recon.min():.1f}°, 평균변화: {sh_diff_recon.mean():.2f}°/f")

print(f"\n=== 골반 비교 ===")
print(f"  raw   range: {pe_deg_raw.max()-pe_deg_raw.min():.1f}°, 평균변화: {pe_diff_raw.mean():.2f}°/f")
print(f"  복원  range: {pe_deg_recon.max()-pe_deg_recon.min():.1f}°, 평균변화: {pe_diff_recon.mean():.2f}°/f")

# ── 시각화 ──
fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)

# 1. X 좌표 비교
axes[0].plot(frames, sh_vx, 'b-', alpha=0.3, label='어깨 X (raw)')
axes[0].plot(frames, sh_vx_recon, 'b-', lw=2, label='어깨 X (복원)')
axes[0].plot(frames, pe_vx, 'r-', alpha=0.3, label='골반 X (raw)')
axes[0].plot(frames, pe_vx_recon, 'r-', lw=2, label='골반 X (복원)')
axes[0].axhline(0, color='gray', lw=0.5, ls='--')
axes[0].set_ylabel('vec_X')
axes[0].set_title('1. X 좌표: raw vs 기하학적 복원')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. arctan2 각도 비교
axes[1].plot(frames, sh_deg_raw, 'b-', alpha=0.3, label='어깨 raw')
axes[1].plot(frames, sh_deg_recon, 'b-', lw=2, label='어깨 복원')
axes[1].plot(frames, pe_deg_raw, 'r-', alpha=0.3, label='골반 raw')
axes[1].plot(frames, pe_deg_recon, 'r-', lw=2, label='골반 복원')
axes[1].set_ylabel('각도 (°)')
axes[1].set_title('2. arctan2 각도: raw vs 복원')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. 프레임간 변화량 비교
axes[2].plot(frames[1:], sh_diff_raw, 'b-', alpha=0.3, label='어깨 raw')
axes[2].plot(frames[1:], sh_diff_recon, 'b-', lw=2, label='어깨 복원')
axes[2].plot(frames[1:], pe_diff_raw, 'r-', alpha=0.3, label='골반 raw')
axes[2].plot(frames[1:], pe_diff_recon, 'r-', lw=2, label='골반 복원')
axes[2].axhline(5, color='orange', lw=1, ls='--', label='5°')
axes[2].set_ylabel('|변화량| (°/frame)')
axes[2].set_xlabel('Frame')
axes[2].set_title('3. 프레임간 변화량: raw vs 복원')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/graphs/x_reconstruct_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/x_reconstruct_analysis.png")
plt.close()
