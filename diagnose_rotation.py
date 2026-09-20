# -*- coding: utf-8 -*-
"""
회전 지표 진단 스크립트

목적: "회전 진폭(S_rot)이 파일명 점수와 안 맞는" 원인 진단.
가설: 진폭은 의도적 회전(숙련)과 몸 흔들림(미숙)을 구분 못 한다.
      숙련자일수록 (1) 어깨-골반이 한 덩어리로 같이 돌고(협응),
      (2) 반주기 회전 폭이 일정할 것(규칙성).

영상별 산출:
  - S_rot (어깨/골반, Hampel 파이프라인 그대로)
  - 어깨-골반 협응: 드리프트 제거 신호 간 Pearson 상관계수
  - 규칙성: 반주기 변화량 Δ_i의 변동계수(CV = std/mean, 작을수록 규칙적)
  - 어깨·골반 겹침 그래프 (output/diagnosis/*.png)

마지막에 파일명 점수(기준=10)와 각 후보 지표의 Spearman 순위상관 출력.

실행: python3 diagnose_rotation.py
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

OUT_DIR = 'output/diagnosis'


def parse_score(stem):
    """'인,인 3-1' → 3, '인,인 기준1' → 10"""
    if '기준' in stem:
        return 10
    m = re.search(r'(\d+)-\d+', stem)
    return int(m.group(1)) if m else None


def extrema_diffs(angles, distance=10, prominence=3):
    """극점 열 → 반주기 변화량 Δ_i 리스트"""
    peaks, _ = find_peaks(angles, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-angles, distance=distance, prominence=prominence)
    ext = sorted(list(peaks) + list(valleys))
    diffs = [abs(angles[ext[i + 1]] - angles[ext[i]]) for i in range(len(ext) - 1)]
    return ext, diffs


def analyze_video(video_path):
    from pathlib import Path
    stem = Path(video_path).stem
    print(f"\n처리 중: {stem}")

    frames, sh_vx, sh_vz, pe_vx, pe_vz = extract_rotation_data(video_path)
    if len(frames) < 20:
        print("  프레임 부족, 건너뜀")
        return None

    # Hampel 파이프라인 (extract_rotation_score.process_angle 그대로)
    sh_angle, sh_bad, *_ = process_angle(sh_vx, sh_vz)
    pe_angle, pe_bad, *_ = process_angle(pe_vx, pe_vz)

    # 1. 진폭 (기존 S_rot)
    sh_ext, sh_diffs = extrema_diffs(sh_angle)
    pe_ext, pe_diffs = extrema_diffs(pe_angle)
    sh_amp = float(np.mean(sh_diffs)) if len(sh_diffs) else None
    pe_amp = float(np.mean(pe_diffs)) if len(pe_diffs) else None

    # 2. 협응: 어깨-골반 신호 상관 (같은 프레임에서 추출되므로 정렬됨)
    coord_r, _ = pearsonr(sh_angle, pe_angle)

    # 3. 규칙성: Δ_i 변동계수 (작을수록 규칙적)
    sh_cv = float(np.std(sh_diffs) / np.mean(sh_diffs)) if len(sh_diffs) >= 2 else None
    pe_cv = float(np.std(pe_diffs) / np.mean(pe_diffs)) if len(pe_diffs) >= 2 else None

    # 그래프: 어깨·골반 겹침 + 극점
    fig, ax = plt.subplots(figsize=(14, 5))
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
    ax.set_title(f"{stem}  (점수 {score})  |  협응 r={coord_r:.2f}  "
                 f"어깨 CV={sh_cv if sh_cv is None else round(sh_cv,2)}  "
                 f"골반 CV={pe_cv if pe_cv is None else round(pe_cv,2)}",
                 fontsize=13)
    ax.set_xlabel('Frame'); ax.set_ylabel('각도 변화 (°)')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(f"{OUT_DIR}/diag_{stem}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  협응 r={coord_r:.2f} | 어깨: 진폭 {sh_amp and round(sh_amp,1)}°, CV {sh_cv and round(sh_cv,2)}"
          f" | 골반: 진폭 {pe_amp and round(pe_amp,1)}°, CV {pe_cv and round(pe_cv,2)}")

    return {
        'video': stem, 'score': score,
        'sh_amp': sh_amp, 'pe_amp': pe_amp,
        'coord_r': float(coord_r),
        'sh_cv': sh_cv, 'pe_cv': pe_cv,
    }


def main():
    rows = [r for v in VIDEOS if os.path.exists(v) for r in [analyze_video(v)] if r]
    if not rows:
        print("결과 없음"); return

    rows.sort(key=lambda r: (r['score'] is None, r['score']))

    print(f"\n{'='*92}")
    print(f"{'영상':<16} {'점수':>4} {'어깨진폭':>8} {'골반진폭':>8} {'협응r':>7} {'어깨CV':>7} {'골반CV':>7}")
    print('-' * 92)
    f = lambda v, d=1: (f"{v:.{d}f}" if v is not None else "N/A")
    for r in rows:
        print(f"{r['video']:<16} {r['score'] if r['score'] is not None else '?':>4} "
              f"{f(r['sh_amp']):>8} {f(r['pe_amp']):>8} {f(r['coord_r'],2):>7} "
              f"{f(r['sh_cv'],2):>7} {f(r['pe_cv'],2):>7}")

    # 점수와의 Spearman 순위상관 (지표 후보 비교)
    scored = [r for r in rows if r['score'] is not None]
    if len(scored) >= 5:
        scores = [r['score'] for r in scored]
        print(f"\n점수와의 Spearman 순위상관 (n={len(scored)}):")
        for key, label, better in [
            ('sh_amp', '어깨 진폭', '높을수록?'),
            ('pe_amp', '골반 진폭', '높을수록?'),
            ('coord_r', '어깨-골반 협응', '높을수록'),
            ('sh_cv', '어깨 규칙성(CV)', '낮을수록'),
            ('pe_cv', '골반 규칙성(CV)', '낮을수록'),
        ]:
            vals = [(r[key], r['score']) for r in scored if r[key] is not None]
            if len(vals) >= 5:
                rho, p = spearmanr([v[0] for v in vals], [v[1] for v in vals])
                print(f"  {label:<14} rho={rho:+.2f} (p={p:.3f})  [{better} 좋다는 가설]")

    import csv
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f'{OUT_DIR}/diagnosis_summary.csv', 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n그래프: {OUT_DIR}/diag_*.png | 요약: {OUT_DIR}/diagnosis_summary.csv")


if __name__ == "__main__":
    main()
