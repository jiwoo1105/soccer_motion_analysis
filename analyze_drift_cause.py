"""
Mediapipe 드리프트 발생 원인 프레임별 분석
- 연속 프레임의 X, Z 좌표 변화를 추적
- 실제 이미지 위에 어깨/골반 좌표 표시
- 드리프트 구간 vs 안정 구간 비교
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
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 전체 프레임 데이터 수집
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
                # world landmarks (3D)
                'sh_wx': wl[12].x - wl[11].x,  # 어깨 벡터 X
                'sh_wz': wl[12].z - wl[11].z,  # 어깨 벡터 Z
                'pe_wx': wl[24].x - wl[23].x,  # 골반 벡터 X
                'pe_wz': wl[24].z - wl[23].z,  # 골반 벡터 Z
                # 개별 좌표 (드리프트 원인 추적)
                'ls_x': wl[11].x, 'ls_z': wl[11].z,  # 왼쪽 어깨
                'rs_x': wl[12].x, 'rs_z': wl[12].z,  # 오른쪽 어깨
                'lh_x': wl[23].x, 'lh_z': wl[23].z,  # 왼쪽 골반
                'rh_x': wl[24].x, 'rh_z': wl[24].z,  # 오른쪽 골반
                # image landmarks (2D)
                'ls_ix': il[11].x * w_frame, 'rs_ix': il[12].x * w_frame,
                'lh_ix': il[23].x * w_frame, 'rh_ix': il[24].x * w_frame,
            })
        idx += 1
cap.release()

n = len(data)
frames = np.array([d['frame'] for d in data])

# 어깨 벡터 X, Z
sh_wx = np.array([d['sh_wx'] for d in data])
sh_wz = np.array([d['sh_wz'] for d in data])

# 개별 어깨 X 좌표
ls_x = np.array([d['ls_x'] for d in data])
rs_x = np.array([d['rs_x'] for d in data])
ls_z = np.array([d['ls_z'] for d in data])
rs_z = np.array([d['rs_z'] for d in data])

# image landmarks
ls_ix = np.array([d['ls_ix'] for d in data])
rs_ix = np.array([d['rs_ix'] for d in data])

# arctan2 각도
sh_deg = np.degrees(np.arctan2(sh_wz, sh_wx))
sh_unwrap = np.degrees(np.unwrap(np.radians(sh_deg)))

# 프레임간 X 변화량
ls_x_diff = np.diff(ls_x)
rs_x_diff = np.diff(rs_x)
ls_z_diff = np.diff(ls_z)
rs_z_diff = np.diff(rs_z)

# ── 시각화 ──
fig, axes = plt.subplots(5, 1, figsize=(18, 28), sharex=True)

# 1. 왼쪽/오른쪽 어깨의 world X 좌표 개별 추적
ax = axes[0]
ax.plot(frames, ls_x, 'b-', lw=1.2, label='L_SHOULDER world X')
ax.plot(frames, rs_x, 'r-', lw=1.2, label='R_SHOULDER world X')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.set_ylabel('world X 좌표')
ax.set_title('① 왼쪽/오른쪽 어깨 world X 좌표 — 두 선이 같이 흔들리면 전체가 드리프트')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 어깨 벡터 X (R-L) vs Z (R-L)
ax = axes[1]
ax.plot(frames, sh_wx, 'b-', lw=1.2, label='어깨 벡터 X (R-L)')
ax.plot(frames, sh_wz, 'r-', lw=1.2, label='어깨 벡터 Z (R-L)')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.set_ylabel('벡터 값')
ax.set_title('② 어깨 벡터 X vs Z — X는 불안정(0 근처 왔다갔다), Z는 안정적')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 프레임간 X 변화량 (드리프트의 실체)
ax = axes[2]
ax.plot(frames[1:], ls_x_diff, 'b-', alpha=0.5, lw=0.8, label='L_SHOULDER ΔX/frame')
ax.plot(frames[1:], rs_x_diff, 'r-', alpha=0.5, lw=0.8, label='R_SHOULDER ΔX/frame')
ax.axhline(0, color='gray', lw=0.5, ls='--')
# 이동평균으로 드리프트 방향 표시
window = 20
if len(ls_x_diff) > window:
    ls_drift = np.convolve(ls_x_diff, np.ones(window)/window, mode='valid')
    ax.plot(frames[window:], ls_drift, 'b-', lw=2.5, label=f'L ΔX 이동평균({window}f) = 드리프트 방향')
    rs_drift = np.convolve(rs_x_diff, np.ones(window)/window, mode='valid')
    ax.plot(frames[window:], rs_drift, 'r-', lw=2.5, label=f'R ΔX 이동평균({window}f)')
ax.set_ylabel('ΔX / frame')
ax.set_title('③ 프레임간 X 변화량 — 매 프레임 작지만 한쪽으로 치우침 = 드리프트')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. image landmarks X vs world X (2D는 안정, 3D는 불안정 비교)
ax = axes[3]
ax.plot(frames, ls_ix, 'b-', lw=1.2, label='L_SHOULDER image X (px)')
ax.plot(frames, rs_ix, 'r-', lw=1.2, label='R_SHOULDER image X (px)')
ax.set_ylabel('image X (px)')
ax.set_title('④ Image landmarks X (2D) — 부드럽고 안정적 (world X와 대조)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. arctan2 각도 + 드리프트 구간 표시
ax = axes[4]
ax.plot(frames, sh_unwrap, 'b-', lw=1.2, label='어깨 arctan2(Z,X) unwrap')
# 드리프트 구간 표시: 각도가 30프레임동안 같은 방향으로 변하는 구간
angle_diff = np.diff(sh_unwrap)
drift_dir = np.sign(angle_diff)
# 같은 방향으로 15프레임 이상 지속되는 구간 찾기
run_start = 0
for i in range(1, len(drift_dir)):
    if drift_dir[i] != drift_dir[run_start]:
        run_len = i - run_start
        if run_len >= 15:
            color = 'red' if drift_dir[run_start] > 0 else 'blue'
            ax.axvspan(frames[run_start], frames[i], alpha=0.15, color=color)
        run_start = i
ax.set_ylabel('각도 (°)')
ax.set_xlabel('Frame')
ax.set_title('⑤ arctan2 각도 — 빨간/파란 영역 = 15프레임 이상 같은 방향 지속 (= 드리프트 구간)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Mediapipe 드리프트 원인 분석 (인,인 3-1)', fontsize=15, fontweight='bold')
plt.tight_layout()
os.makedirs('output/graphs', exist_ok=True)
plt.savefig('output/graphs/drift_cause_analysis.png', dpi=150, bbox_inches='tight')
print(f"그래프 저장: output/graphs/drift_cause_analysis.png")
plt.close()

# ── 프레임별 이미지 + 좌표 시각화 (드리프트 구간 10프레임) ──
# 드리프트가 가장 큰 구간 찾기 (30프레임에서 각도 변화 최대)
window_check = 30
max_drift = 0
max_drift_start = 0
for i in range(len(sh_unwrap) - window_check):
    drift = abs(sh_unwrap[i + window_check] - sh_unwrap[i])
    if drift > max_drift:
        max_drift = drift
        max_drift_start = i

print(f"\n최대 드리프트 구간: frame {frames[max_drift_start]}~{frames[max_drift_start+window_check]}")
print(f"  각도 변화: {sh_unwrap[max_drift_start]:.1f}° → {sh_unwrap[max_drift_start+window_check]:.1f}° ({max_drift:.1f}°)")

# 해당 구간에서 5프레임 간격으로 이미지 + 좌표 출력
print(f"\n  프레임별 world X 좌표 (드리프트 구간):")
print(f"  {'Frame':>6} {'L_SH.x':>8} {'R_SH.x':>8} {'벡터X':>8} {'벡터Z':>8} {'arctan2':>8} {'Δ각도':>8}")
print(f"  {'-'*56}")
for i in range(max_drift_start, min(max_drift_start + window_check + 1, n)):
    d = data[i]
    angle = sh_unwrap[i]
    delta = sh_unwrap[i] - sh_unwrap[max_drift_start] if i > max_drift_start else 0
    print(f"  {d['frame']:>6} {d['ls_x']:>8.4f} {d['rs_x']:>8.4f} {d['sh_wx']:>8.4f} {d['sh_wz']:>8.4f} {angle:>8.1f}° {delta:>+7.1f}°")

# 프레임 이미지 캡처
cap = cv2.VideoCapture(VIDEO_PATH)
sample_indices = list(range(max_drift_start, min(max_drift_start + window_check + 1, n), 5))
sample_imgs = {}

for si in sample_indices:
    fn = data[si]['frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if ret:
        vis = frame.copy()
        d = data[si]
        # 어깨 표시
        lsx, lsy = int(d['ls_ix']), int(data[si].get('ls_ix', 0))
        # image landmarks에서 Y도 필요하므로 다시 계산
        cv2.putText(vis, f"F{fn} | vec_X={d['sh_wx']:.3f} | angle={sh_unwrap[si]:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        sample_imgs[fn] = vis
cap.release()

# 프레임 이미지 저장
if sample_imgs:
    cols = min(len(sample_imgs), 4)
    rows = (len(sample_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (fn, img) in enumerate(sorted(sample_imgs.items())):
        if i >= len(axes):
            break
        si = [j for j, d in enumerate(data) if d['frame'] == fn][0]
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = data[si]
        axes[i].set_title(f'F{fn} | X={d["sh_wx"]:.3f} | {sh_unwrap[si]:.1f}°', fontsize=9)
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'드리프트 구간 프레임 (F{data[max_drift_start]["frame"]}~F{data[max_drift_start+window_check]["frame"]})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/graphs/drift_cause_frames.png', dpi=120, bbox_inches='tight')
    print(f"프레임 이미지 저장: output/graphs/drift_cause_frames.png")
    plt.close()
