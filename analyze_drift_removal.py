"""
드리프트 제거 방법 비교 테스트
1. 고주파 통과 필터 (butterworth highpass)
2. 베이스라인 제거 (큰 윈도우 savgol 빼기)
3. 원본 (비교용)
모든 결과를 2D 폭과 비교
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

VIDEO_PATH = "input/in_in/인,인 3-1.MOV"


def extract_all(video_path):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh_vx, sh_vz, pe_vx, pe_vz = [], [], [], []
    sh_w2d, pe_w2d = [], []
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
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
                sh_w2d.append(abs(il[12].x - il[11].x) * w_frame)
                pe_w2d.append(abs(il[24].x - il[23].x) * w_frame)
    cap.release()
    return (np.array(sh_vx), np.array(sh_vz),
            np.array(pe_vx), np.array(pe_vz),
            np.array(sh_w2d), np.array(pe_w2d))


def base_process(vx, vz):
    """기본: unwrap + 10° 제거 + savgol41"""
    raw = np.degrees(np.arctan2(vz, vx))
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))
    cleaned = unwrapped.copy()
    for i in range(1, len(cleaned)):
        if abs(cleaned[i] - cleaned[i-1]) > 10:
            cleaned[i] = cleaned[i-1]
    n = len(cleaned)
    win = min(41, n if n % 2 == 1 else n - 1)
    smoothed = savgol_filter(cleaned, win, 2) if win >= 3 else cleaned
    return smoothed


def method_baseline_removal(signal, big_win=121):
    """방법1: 베이스라인 제거 - 큰 윈도우 savgol로 추세선 구해서 빼기"""
    n = len(signal)
    win = min(big_win, n if n % 2 == 1 else n - 1)
    if win >= 3:
        baseline = savgol_filter(signal, win, 2)
    else:
        baseline = signal
    detrended = signal - baseline
    return detrended, baseline


def method_highpass(signal, cutoff=0.3, fps=30):
    """방법2: 고주파 통과 필터 - cutoff Hz 이하 제거"""
    b, a = butter(2, cutoff / (fps / 2), btype='high')
    filtered = filtfilt(b, a, signal)
    return filtered


def get_peak_score(angles, distance=10, prominence=2):
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    extrema = sorted(list(peaks) + list(valleys))
    if len(extrema) < 2:
        return None, []
    diffs = [abs(angles[extrema[i+1]] - angles[extrema[i]]) for i in range(len(extrema)-1)]
    return float(np.mean(diffs)), extrema


print("데이터 추출 중...")
sh_vx, sh_vz, pe_vx, pe_vz, sh_w2d, pe_w2d = extract_all(VIDEO_PATH)
n = len(sh_vx)
frames = np.arange(n)

# 기본 처리
sh_base = base_process(sh_vx, sh_vz)
pe_base = base_process(pe_vx, pe_vz)

# 방법1: 베이스라인 제거 (여러 윈도우)
sh_detrend_81, sh_bl_81 = method_baseline_removal(sh_base, 81)
sh_detrend_121, sh_bl_121 = method_baseline_removal(sh_base, 121)
sh_detrend_161, sh_bl_161 = method_baseline_removal(sh_base, 161)

pe_detrend_81, pe_bl_81 = method_baseline_removal(pe_base, 81)
pe_detrend_121, pe_bl_121 = method_baseline_removal(pe_base, 121)
pe_detrend_161, pe_bl_161 = method_baseline_removal(pe_base, 161)

# 방법2: 고주파 통과 (여러 cutoff)
sh_hp_02 = method_highpass(sh_base, 0.2)
sh_hp_03 = method_highpass(sh_base, 0.3)
sh_hp_05 = method_highpass(sh_base, 0.5)

pe_hp_02 = method_highpass(pe_base, 0.2)
pe_hp_03 = method_highpass(pe_base, 0.3)
pe_hp_05 = method_highpass(pe_base, 0.5)

# 2D 폭도 detrend해서 비교 (움직임 패턴만 추출)
sh_w2d_smooth = savgol_filter(sh_w2d, min(41, n if n%2==1 else n-1), 2)
pe_w2d_smooth = savgol_filter(pe_w2d, min(41, n if n%2==1 else n-1), 2)
sh_w2d_detrend, _ = method_baseline_removal(sh_w2d_smooth, 121)
pe_w2d_detrend, _ = method_baseline_removal(pe_w2d_smooth, 121)

# 점수 비교
print(f"\n{'방법':<25} {'어깨 score':>10} {'어깨 range':>12} {'골반 score':>10} {'골반 range':>12}")
print("-" * 72)

methods = {
    '원본 (10°+sav41)': (sh_base, pe_base),
    '베이스라인제거 win81': (sh_detrend_81, pe_detrend_81),
    '베이스라인제거 win121': (sh_detrend_121, pe_detrend_121),
    '베이스라인제거 win161': (sh_detrend_161, pe_detrend_161),
    '고주파통과 0.2Hz': (sh_hp_02, pe_hp_02),
    '고주파통과 0.3Hz': (sh_hp_03, pe_hp_03),
    '고주파통과 0.5Hz': (sh_hp_05, pe_hp_05),
}

for name, (sh, pe) in methods.items():
    sh_sc, _ = get_peak_score(sh)
    pe_sc, _ = get_peak_score(pe)
    sh_r = sh.max() - sh.min()
    pe_r = pe.max() - pe.min()
    sh_str = f"{sh_sc:.1f}°" if sh_sc else "N/A"
    pe_str = f"{pe_sc:.1f}°" if pe_sc else "N/A"
    print(f"{name:<25} {sh_str:>10} {sh_r:>10.1f}° {pe_str:>10} {pe_r:>10.1f}°")

# ── 시각화 ──
fig, axes = plt.subplots(4, 1, figsize=(18, 20), sharex=True)

# 1. 어깨 원본 + 베이스라인
ax = axes[0]
ax.plot(frames, sh_base, 'b-', alpha=0.5, lw=1, label='원본 (10°+sav41)')
ax.plot(frames, sh_bl_121, 'g--', lw=2, label='베이스라인 (win121) = 드리프트')
ax.set_ylabel('각도 (°)')
ax.set_title('어깨: 원본에서 드리프트(초록 점선) 분리')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 어깨 드리프트 제거 결과
ax = axes[1]
ax.plot(frames, sh_detrend_121, 'b-', lw=1.5, label='베이스라인 제거 (win121)')
ax.plot(frames, sh_hp_03, 'c-', lw=1.5, label='고주파통과 (0.3Hz)')
# peak 표시
for sig, color in [(sh_detrend_121, 'blue'), (sh_hp_03, 'cyan')]:
    sc, ext = get_peak_score(sig)
    if ext:
        ax.scatter(frames[ext], sig[ext], color=color, s=40, zorder=5)
ax1b = ax.twinx()
ax1b.plot(frames, sh_w2d_detrend, 'r-', alpha=0.5, lw=1, label='2D폭 detrend (기준)')
ax1b.set_ylabel('2D 폭 변화', color='red')
sh_sc1, _ = get_peak_score(sh_detrend_121)
sh_sc2, _ = get_peak_score(sh_hp_03)
ax.set_ylabel('각도 변화 (°)')
ax.set_title(f'어깨 드리프트 제거 후: 베이스라인={sh_sc1:.1f}° / 고주파={sh_sc2:.1f}°')
lines = ax.get_lines() + ax1b.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
ax.grid(True, alpha=0.3)

# 3. 골반 원본 + 베이스라인
ax = axes[2]
ax.plot(frames, pe_base, 'r-', alpha=0.5, lw=1, label='원본 (10°+sav41)')
ax.plot(frames, pe_bl_121, 'g--', lw=2, label='베이스라인 (win121) = 드리프트')
ax.set_ylabel('각도 (°)')
ax.set_title('골반: 원본에서 드리프트(초록 점선) 분리')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 골반 드리프트 제거 결과
ax = axes[3]
ax.plot(frames, pe_detrend_121, 'r-', lw=1.5, label='베이스라인 제거 (win121)')
ax.plot(frames, pe_hp_03, 'm-', lw=1.5, label='고주파통과 (0.3Hz)')
for sig, color in [(pe_detrend_121, 'red'), (pe_hp_03, 'magenta')]:
    sc, ext = get_peak_score(sig)
    if ext:
        ax.scatter(frames[ext], sig[ext], color=color, s=40, zorder=5)
ax2b = ax.twinx()
ax2b.plot(frames, pe_w2d_detrend, 'orange', alpha=0.5, lw=1, label='2D폭 detrend (기준)')
ax2b.set_ylabel('2D 폭 변화', color='orange')
pe_sc1, _ = get_peak_score(pe_detrend_121)
pe_sc2, _ = get_peak_score(pe_hp_03)
ax.set_ylabel('각도 변화 (°)')
ax.set_xlabel('Frame')
ax.set_title(f'골반 드리프트 제거 후: 베이스라인={pe_sc1:.1f}° / 고주파={pe_sc2:.1f}°')
lines2 = ax.get_lines() + ax2b.get_lines()
ax.legend(lines2, [l.get_label() for l in lines2], loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle('드리프트 제거 비교 (인,인 3-1)', fontsize=15, fontweight='bold')
plt.tight_layout()
os.makedirs('output/graphs', exist_ok=True)
plt.savefig('output/graphs/drift_removal_compare.png', dpi=150, bbox_inches='tight')
print(f"\n그래프 저장: output/graphs/drift_removal_compare.png")
plt.close()
