"""
어깨/골반 world_landmarks X 좌표 노이즈 패턴 분석
arctan2 방식 복원 + 스파이크 제거 가능성 검증
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

# Mediapipe로 전체 프레임 world_landmarks 추출
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"총 프레임: {total}")

sh_vx_raw = []  # 어깨 벡터 X (raw)
sh_vz_raw = []  # 어깨 벡터 Z (raw)
pe_vx_raw = []  # 골반 벡터 X (raw)
pe_vz_raw = []  # 골반 벡터 Z (raw)
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
        if results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark
            # 어깨: 12(R) - 11(L)
            sh_x = wl[12].x - wl[11].x
            sh_z = wl[12].z - wl[11].z
            # 골반: 24(R) - 23(L)
            pe_x = wl[24].x - wl[23].x
            pe_z = wl[24].z - wl[23].z

            sh_vx_raw.append(sh_x)
            sh_vz_raw.append(sh_z)
            pe_vx_raw.append(pe_x)
            pe_vz_raw.append(pe_z)
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

# arctan2 각도 계산 (raw)
sh_deg_raw = np.degrees(np.arctan2(sh_vz, sh_vx))
pe_deg_raw = np.degrees(np.arctan2(pe_vz, pe_vx))

# 프레임 간 각도 변화량
sh_diff = np.abs(np.diff(sh_deg_raw))
pe_diff = np.abs(np.diff(pe_deg_raw))

# ── 분석 결과 출력 ──
print("\n=== 어깨 arctan2 각도 변화 분석 ===")
print(f"  각도 범위: {sh_deg_raw.min():.1f}° ~ {sh_deg_raw.max():.1f}°")
print(f"  프레임간 변화량 평균: {sh_diff.mean():.2f}°")
print(f"  프레임간 변화량 최대: {sh_diff.max():.2f}°")
for thresh in [5, 10, 15, 20, 30]:
    count = np.sum(sh_diff > thresh)
    pct = count / len(sh_diff) * 100
    print(f"  > {thresh}° 점프 프레임: {count}개 ({pct:.1f}%)")

print("\n=== 골반 arctan2 각도 변화 분석 ===")
print(f"  각도 범위: {pe_deg_raw.min():.1f}° ~ {pe_deg_raw.max():.1f}°")
print(f"  프레임간 변화량 평균: {pe_diff.mean():.2f}°")
print(f"  프레임간 변화량 최대: {pe_diff.max():.2f}°")
for thresh in [5, 10, 15, 20, 30]:
    count = np.sum(pe_diff > thresh)
    pct = count / len(pe_diff) * 100
    print(f"  > {thresh}° 점프 프레임: {count}개 ({pct:.1f}%)")

# ── 시뮬레이션: 스파이크 제거 후 결과 ──
def remove_spikes_and_interpolate(angles, threshold=10):
    """threshold 이상 점프하는 프레임을 제거하고 선형 보간"""
    cleaned = angles.copy()
    bad_mask = np.zeros(len(angles), dtype=bool)

    for i in range(1, len(angles)):
        if abs(cleaned[i] - cleaned[i-1]) > threshold:
            bad_mask[i] = True

    # 연속 bad 구간 확인
    bad_indices = np.where(bad_mask)[0]
    good_indices = np.where(~bad_mask)[0]

    if len(good_indices) < 2:
        return cleaned, bad_mask

    # 선형 보간
    cleaned[bad_mask] = np.interp(
        np.where(bad_mask)[0], good_indices, cleaned[good_indices]
    )
    return cleaned, bad_mask

sh_cleaned_5, sh_bad_5 = remove_spikes_and_interpolate(sh_deg_raw, 5)
sh_cleaned_10, sh_bad_10 = remove_spikes_and_interpolate(sh_deg_raw, 10)
pe_cleaned_5, pe_bad_5 = remove_spikes_and_interpolate(pe_deg_raw, 5)
pe_cleaned_10, pe_bad_10 = remove_spikes_and_interpolate(pe_deg_raw, 10)

print(f"\n=== 스파이크 제거 시뮬레이션 ===")
print(f"어깨 5° 제거: {sh_bad_5.sum()}개 프레임 제거 → range {sh_cleaned_5.max()-sh_cleaned_5.min():.1f}°")
print(f"어깨 10° 제거: {sh_bad_10.sum()}개 프레임 제거 → range {sh_cleaned_10.max()-sh_cleaned_10.min():.1f}°")
print(f"골반 5° 제거: {pe_bad_5.sum()}개 프레임 제거 → range {pe_cleaned_5.max()-pe_cleaned_5.min():.1f}°")
print(f"골반 10° 제거: {pe_bad_10.sum()}개 프레임 제거 → range {pe_cleaned_10.max()-pe_cleaned_10.min():.1f}°")

# ── 시각화 ──
fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

# 1. X 좌표 raw
axes[0].plot(frames, sh_vx, 'b-', alpha=0.7, label='어깨 vec_X')
axes[0].plot(frames, pe_vx, 'r-', alpha=0.7, label='골반 vec_X')
axes[0].axhline(0, color='gray', lw=0.5, ls='--')
axes[0].set_ylabel('vec_X (world)')
axes[0].set_title('1. 어깨/골반 벡터 X 좌표 (raw) - 부호 반전 확인')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. arctan2 각도 raw
axes[1].plot(frames, sh_deg_raw, 'b-', alpha=0.7, label='어깨 arctan2')
axes[1].plot(frames, pe_deg_raw, 'r-', alpha=0.7, label='골반 arctan2')
axes[1].set_ylabel('각도 (°)')
axes[1].set_title('2. arctan2(Z, X) 각도 (raw) - 점프 확인')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. 프레임간 변화량
axes[2].bar(frames[1:], sh_diff, color='blue', alpha=0.5, label='어깨 |Δ|')
axes[2].bar(frames[1:], pe_diff, color='red', alpha=0.5, label='골반 |Δ|')
axes[2].axhline(5, color='orange', lw=1.5, ls='--', label='5° threshold')
axes[2].axhline(10, color='red', lw=1.5, ls='--', label='10° threshold')
axes[2].set_ylabel('|변화량| (°/frame)')
axes[2].set_title('3. 프레임 간 각도 변화량 - 스파이크 빈도 확인')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. 스파이크 제거 후 (10° threshold)
axes[3].plot(frames, sh_deg_raw, 'b-', alpha=0.2, label='어깨 raw')
axes[3].plot(frames, sh_cleaned_10, 'b-', lw=2, label='어깨 cleaned (10°)')
axes[3].plot(frames, pe_deg_raw, 'r-', alpha=0.2, label='골반 raw')
axes[3].plot(frames, pe_cleaned_10, 'r-', lw=2, label='골반 cleaned (10°)')
# bad frames 표시
sh_bad_idx = frames[sh_bad_10]
pe_bad_idx = frames[pe_bad_10]
axes[3].scatter(sh_bad_idx, sh_deg_raw[sh_bad_10], color='blue', s=20, zorder=5, marker='x')
axes[3].scatter(pe_bad_idx, pe_deg_raw[pe_bad_10], color='red', s=20, zorder=5, marker='x')
axes[3].set_ylabel('각도 (°)')
axes[3].set_xlabel('Frame')
axes[3].set_title('4. 스파이크 제거 + 보간 결과 (10° threshold) - X 표시 = 제거된 프레임')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('output/graphs', exist_ok=True)
plt.savefig('output/graphs/x_noise_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/x_noise_analysis.png")
plt.close()
