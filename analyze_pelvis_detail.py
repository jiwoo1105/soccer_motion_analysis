"""
골반 2D 폭 프레임별 상세 분석
- 매 프레임 좌/우 힙 image landmarks 좌표
- 2D 폭 변화 추이
- 실제 프레임 이미지 + 골반 좌표 시각화 (10프레임 간격)
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
h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 데이터 수집
data = []
sample_frames_img = {}  # 특정 프레임의 이미지 저장

# 10프레임 간격으로 이미지 캡처할 프레임 목록
sample_indices = list(range(0, total, 15))[:16]  # 최대 16장

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
        if results.pose_landmarks:
            il = results.pose_landmarks.landmark

            l_hip_x = il[23].x  # normalized [0,1]
            l_hip_y = il[23].y
            r_hip_x = il[24].x
            r_hip_y = il[24].y

            l_sh_x = il[11].x
            r_sh_x = il[12].x

            # 2D 폭 (px)
            pe_width = abs(r_hip_x - l_hip_x) * w_frame
            sh_width = abs(r_sh_x - l_sh_x) * w_frame

            # 부호 있는 폭 (R - L, 양수면 R이 오른쪽에)
            pe_signed = (r_hip_x - l_hip_x) * w_frame
            sh_signed = (r_sh_x - l_sh_x) * w_frame

            data.append({
                'frame': idx,
                'l_hip_x': l_hip_x * w_frame,
                'l_hip_y': l_hip_y * h_frame,
                'r_hip_x': r_hip_x * w_frame,
                'r_hip_y': r_hip_y * h_frame,
                'pe_width': pe_width,
                'pe_signed': pe_signed,
                'sh_width': sh_width,
                'sh_signed': sh_signed,
            })

            if idx in sample_indices:
                # 프레임에 골반/어깨 포인트 표시
                vis = frame.copy()
                # 골반 (빨간)
                lhx, lhy = int(l_hip_x * w_frame), int(l_hip_y * h_frame)
                rhx, rhy = int(r_hip_x * w_frame), int(r_hip_y * h_frame)
                cv2.circle(vis, (lhx, lhy), 8, (0, 0, 255), -1)
                cv2.circle(vis, (rhx, rhy), 8, (255, 0, 0), -1)
                cv2.line(vis, (lhx, lhy), (rhx, rhy), (0, 0, 255), 2)
                cv2.putText(vis, f"L_HIP", (lhx-30, lhy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.putText(vis, f"R_HIP", (rhx-30, rhy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                # 어깨 (초록)
                lsx = int(l_sh_x * w_frame)
                rsx = int(r_sh_x * w_frame)
                lsy = int(il[11].y * h_frame)
                rsy = int(il[12].y * h_frame)
                cv2.circle(vis, (lsx, lsy), 8, (0, 255, 0), -1)
                cv2.circle(vis, (rsx, rsy), 8, (0, 255, 0), -1)
                cv2.line(vis, (lsx, lsy), (rsx, rsy), (0, 255, 0), 2)
                # 폭 정보
                info = f"F{idx} | Pelvis:{pe_width:.0f}px({pe_signed:+.0f}) | Shoulder:{sh_width:.0f}px({sh_signed:+.0f})"
                cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                sample_frames_img[idx] = vis
        idx += 1
cap.release()

frames = np.array([d['frame'] for d in data])
pe_width = np.array([d['pe_width'] for d in data])
pe_signed = np.array([d['pe_signed'] for d in data])
sh_width = np.array([d['sh_width'] for d in data])
sh_signed = np.array([d['sh_signed'] for d in data])
l_hip_x = np.array([d['l_hip_x'] for d in data])
r_hip_x = np.array([d['r_hip_x'] for d in data])

print(f"총 {len(data)}프레임 분석")
print(f"\n=== 골반 2D 폭 (abs) ===")
print(f"  범위: {pe_width.min():.1f} ~ {pe_width.max():.1f} px")
print(f"  평균: {pe_width.mean():.1f} px")
print(f"  std:  {pe_width.std():.1f} px")

print(f"\n=== 골반 2D 폭 (signed: R-L) ===")
print(f"  범위: {pe_signed.min():.1f} ~ {pe_signed.max():.1f} px")
print(f"  양수(R>L): {(pe_signed > 0).sum()}프레임")
print(f"  음수(L>R): {(pe_signed < 0).sum()}프레임")

print(f"\n=== 어깨 2D 폭 (signed: R-L) ===")
print(f"  범위: {sh_signed.min():.1f} ~ {sh_signed.max():.1f} px")
print(f"  양수(R>L): {(sh_signed > 0).sum()}프레임")
print(f"  음수(L>R): {(sh_signed < 0).sum()}프레임")

# ── 그래프 1: 2D 폭 추이 ──
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

# 1. 부호 있는 폭
axes[0].plot(frames, sh_signed, 'g-', lw=1.5, label='어깨 (R-L) px')
axes[0].plot(frames, pe_signed, 'r-', lw=1.5, label='골반 (R-L) px')
axes[0].axhline(0, color='gray', lw=1, ls='--')
axes[0].set_ylabel('폭 (px, 부호 있음)')
axes[0].set_title('1. 부호 있는 폭 (R-L): 음수 = R이 왼쪽에 → 좌우 뒤집힘?')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. 절대값 폭
axes[1].plot(frames, sh_width, 'g-', lw=1.5, label='어깨 |R-L| px')
axes[1].plot(frames, pe_width, 'r-', lw=1.5, label='골반 |R-L| px')
axes[1].set_ylabel('폭 (px)')
axes[1].set_title('2. 절대값 폭: 실제 회전 = 폭 변화')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. L_HIP, R_HIP X 좌표 각각
axes[2].plot(frames, l_hip_x, 'r-', alpha=0.7, label='L_HIP x (px)')
axes[2].plot(frames, r_hip_x, 'b-', alpha=0.7, label='R_HIP x (px)')
axes[2].set_ylabel('X 좌표 (px)')
axes[2].set_xlabel('Frame')
axes[2].set_title('3. 좌/우 힙 X 좌표 개별 추적: 교차하면 좌우 뒤집힌 것')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('output/graphs', exist_ok=True)
plt.savefig('output/graphs/pelvis_detail_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/pelvis_detail_analysis.png")
plt.close()

# ── 그래프 2: 샘플 프레임 이미지 ──
n_samples = len(sample_frames_img)
if n_samples > 0:
    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i, (fidx, img) in enumerate(sorted(sample_frames_img.items())):
        if i >= len(axes):
            break
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Frame {fidx}', fontsize=10)
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('output/graphs/pelvis_sample_frames.png', dpi=120, bbox_inches='tight')
    print(f"샘플 프레임 저장: output/graphs/pelvis_sample_frames.png")
    plt.close()
