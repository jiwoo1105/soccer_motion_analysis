# -*- coding: utf-8 -*-
"""
회전 지표 진단 스크립트 v2

v1 대비 변경:
  (a) 경계 왜곡 제거: 신호 양 끝 EDGE(20)프레임 내의 극점은 제외
      (드리프트 제거용 savgol 베이스라인이 경계에서 부정확 → 가짜 스윙 발생)
  (b) 지표 추가:
      - 리듬 규칙성: 극점 간 프레임 간격의 CV (작을수록 일정한 리듬으로 회전)
      - 회전 지속성: 초당 반주기 수 (30fps 기준, 높을수록 쉬지 않고 회전)
  (c) 품질 필터: 협응 r < 0.5 영상은 추적 불량 의심(예: 7-3) →
      Spearman을 전체 / 품질통과 두 버전으로 계산

실행: python3 diagnose_rotation_v2.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

from extract_rotation_score import extract_rotation_data, process_angle, VIDEOS

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'output/diagnosis_v2'
EDGE = 20       # 경계 제외 프레임 수
FPS = 30.0
QUALITY_R = 0.5  # 협응 r이 이 미만이면 추적 불량 의심


def parse_score(stem):
    if '기준' in stem:
        return 10
    m = re.search(r'(\d+)-\d+', stem)
    return int(m.group(1)) if m else None


def extrema_features(angles, distance=10, prominence=3, edge=EDGE):
    """경계 제외 극점 → (극점 인덱스, Δ리스트, 간격CV, 초당 반주기 수)"""
    n = len(angles)
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    ext = sorted(list(peaks) + list(valleys))
    ext = [e for e in ext if edge <= e < n - edge]   # (a) 경계 극점 제외

    diffs = [abs(angles[ext[i + 1]] - angles[ext[i]]) for i in range(len(ext) - 1)]
    gaps = np.diff(ext) if len(ext) >= 2 else np.array([])

    amp = float(np.mean(diffs)) if diffs else None
    amp_cv = float(np.std(diffs) / np.mean(diffs)) if len(diffs) >= 2 else None
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) >= 2 else None
    rate = len(diffs) / ((n - 2 * edge) / FPS) if n > 2 * edge and diffs else None

    return ext, diffs, amp, amp_cv, gap_cv, rate


def analyze_video(video_path):
    from pathlib import Path
    stem = Path(video_path).stem
    print(f"\n처리 중: {stem}")

    frames, sh_vx, sh_vz, pe_vx, pe_vz = extract_rotation_data(video_path)
    if len(frames) < 2 * EDGE + 20:
        print("  프레임 부족, 건너뜀")
        return None

    sh_angle, *_ = process_angle(sh_vx, sh_vz)
    pe_angle, *_ = process_angle(pe_vx, pe_vz)

    coord_r, _ = pearsonr(sh_angle, pe_angle)

    sh_ext, sh_diffs, sh_amp, sh_ampcv, sh_gapcv, sh_rate = extrema_features(sh_angle)
    pe_ext, pe_diffs, pe_amp, pe_ampcv, pe_gapcv, pe_rate = extrema_features(pe_angle)

    quality_ok = coord_r >= QUALITY_R

    # 그래프
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(frames)
    ax.axvspan(frames[0], frames[min(EDGE, n - 1)], color='gray', alpha=0.15)
    ax.axvspan(frames[max(0, n - 1 - EDGE)], frames[-1], color='gray', alpha=0.15,
               label='경계 제외 구간')
    ax.plot(frames, sh_angle, '-', color='#1f6fb5', lw=1.8, label='어깨')
    ax.plot(frames, pe_angle, '-', color='#c0392b', lw=1.8, label='골반')
    if sh_ext:
        ax.scatter(frames[sh_ext], sh_angle[sh_ext], color='#1f6fb5', s=45,
                   zorder=5, edgecolors='black', linewidths=0.5)
    if pe_ext:
        ax.scatter(frames[pe_ext], pe_angle[pe_ext], color='#c0392b', s=45,
                   zorder=5, edgecolors='black', linewidths=0.5)
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    score = parse_score(stem)
    flag = '' if quality_ok else '  [추적 불량 의심]'
    r2 = lambda v: None if v is None else round(v, 2)
    ax.set_title(f"{stem} (점수 {score}){flag} | 협응 r={coord_r:.2f} | "
                 f"골반: 진폭 {r2(pe_amp)}° 리듬CV {r2(pe_gapcv)} {r2(pe_rate)}회/초",
                 fontsize=12)
    ax.set_xlabel('Frame'); ax.set_ylabel('각도 변화 (°)')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/diag_{stem}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  협응 r={coord_r:.2f}{flag}")
    print(f"  어깨: 진폭 {r2(sh_amp)}° | 진폭CV {r2(sh_ampcv)} | 리듬CV {r2(sh_gapcv)} | {r2(sh_rate)}회/초")
    print(f"  골반: 진폭 {r2(pe_amp)}° | 진폭CV {r2(pe_ampcv)} | 리듬CV {r2(pe_gapcv)} | {r2(pe_rate)}회/초")

    return {
        'video': stem, 'score': score, 'coord_r': float(coord_r),
        'quality_ok': quality_ok,
        'sh_amp': sh_amp, 'pe_amp': pe_amp,
        'sh_amp_cv': sh_ampcv, 'pe_amp_cv': pe_ampcv,
        'sh_gap_cv': sh_gapcv, 'pe_gap_cv': pe_gapcv,
        'sh_rate': sh_rate, 'pe_rate': pe_rate,
    }


FEATURES = [
    ('sh_amp',    '어깨 진폭',        '높을수록'),
    ('pe_amp',    '골반 진폭',        '높을수록'),
    ('coord_r',   '어깨-골반 협응',    '높을수록'),
    ('sh_amp_cv', '어깨 진폭CV',      '낮을수록'),
    ('pe_amp_cv', '골반 진폭CV',      '낮을수록'),
    ('sh_gap_cv', '어깨 리듬CV',      '낮을수록'),
    ('pe_gap_cv', '골반 리듬CV',      '낮을수록'),
    ('sh_rate',   '어깨 반주기/초',    '높을수록'),
    ('pe_rate',   '골반 반주기/초',    '높을수록'),
]


def spearman_table(rows, label):
    scored = [r for r in rows if r['score'] is not None]
    if len(scored) < 5:
        return
    print(f"\n점수와의 Spearman 순위상관 — {label} (n={len(scored)}):")
    for key, name, better in FEATURES:
        vals = [(r[key], r['score']) for r in scored if r[key] is not None]
        if len(vals) >= 5:
            rho, p = spearmanr([v[0] for v in vals], [v[1] for v in vals])
            mark = ' **' if abs(rho) >= 0.6 else ''
            print(f"  {name:<12} rho={rho:+.2f} (p={p:.3f})  [{better}]{mark}")


def main():
    rows = [r for v in VIDEOS if os.path.exists(v) for r in [analyze_video(v)] if r]
    if not rows:
        print("결과 없음"); return
    rows.sort(key=lambda r: (r['score'] is None, r['score']))

    f = lambda v, d=1: (f"{v:.{d}f}" if v is not None else "N/A")
    print(f"\n{'='*104}")
    print(f"{'영상':<15} {'점수':>4} {'품질':>4} {'골반진폭':>8} {'골반리듬CV':>9} "
          f"{'골반회전/초':>9} {'어깨진폭':>8} {'어깨리듬CV':>9} {'협응r':>6}")
    print('-' * 104)
    for r in rows:
        print(f"{r['video']:<15} {r['score'] if r['score'] is not None else '?':>4} "
              f"{'OK' if r['quality_ok'] else '의심':>4} "
              f"{f(r['pe_amp']):>8} {f(r['pe_gap_cv'],2):>9} {f(r['pe_rate'],2):>9} "
              f"{f(r['sh_amp']):>8} {f(r['sh_gap_cv'],2):>9} {f(r['coord_r'],2):>6}")

    spearman_table(rows, '전체')
    good = [r for r in rows if r['quality_ok']]
    if len(good) < len(rows):
        excluded = [r['video'] for r in rows if not r['quality_ok']]
        print(f"\n제외(협응 r < {QUALITY_R}): {', '.join(excluded)}")
        spearman_table(good, '품질 통과만')

    import csv
    with open(f'{OUT_DIR}/diagnosis_v2_summary.csv', 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n그래프: {OUT_DIR}/diag_*.png | 요약: {OUT_DIR}/diagnosis_v2_summary.csv")


if __name__ == "__main__":
    main()
