# -*- coding: utf-8 -*-
"""Phase 3 검증 — 점수가 사람이 매긴 등급을 재현하는가

출력:
  1. 영상별 원시값 / 지표 점수 / 그룹 점수 / 총점 / 터치 품질
  2. 각 지표 점수와 점수대의 Pearson·Spearman 상관
  3. 하(3,5) / 중(6,7) / 상(8,9) 3그룹 평균과 단조성 여부

**상관이 낮게 나와도 파라미터를 튜닝하지 않는다.** 있는 그대로 보고하고
어떤 영상이 어긋나는지 짚는다. n이 작아 과적합이 너무 쉽다.

실행:
    python3.9 validate_scoring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scoring import raw_metrics, calibration
from scoring.calibration import METRICS, METRIC_LABELS
from scoring.score_mapper import score_video, GROUPS

TIER_GROUPS = (('하', (3, 5)), ('중', (6, 7)), ('상', (8, 9)))


def f(v, d=1):
    return f'{v:.{d}f}' if v is not None else 'N/A'


def table(results):
    print(f"\n{'='*104}")
    print('영상별 결과 (점수대 오름차순)')
    print(f"{'영상':<13}{'등급':>4}"
          f"{'헤드업':>14}{'상체각도':>14}{'어깨':>13}{'골반':>13}"
          f"{'어깨·골반':>9}{'총점':>7}{'터치':>7}")
    print('-' * 104)
    for r in sorted(results, key=lambda r: (r['tier'] is None, r['tier'] or 0)):
        s, g, raw = r['scores'], r['group_scores'], r['raw']
        cell = lambda m, d=1: f"{f(s[m])} ({f(raw[m], d)})"
        print(f"{r['video']:<13}{r['tier'] if r['tier'] else '기준':>4}"
              f"{cell('headup'):>14}{cell('trunk'):>14}"
              f"{cell('shoulder'):>13}{cell('pelvis'):>13}"
              f"{f(g['rotation']):>9}{f(r['total']):>7}"
              f"{str(r['touch_valid'])+'/'+str(r['touch_total']):>7}")


def correlations(results):
    """지표 점수 ↔ 점수대 상관. 기준영상은 등급이 없으므로 제외한다."""
    scored = [r for r in results if r['tier'] is not None]
    print(f"\n{'='*104}")
    print('점수 ↔ 사람이 매긴 등급 상관 (기준영상 제외)')
    print(f"{'지표':<12}{'n':>4}{'Pearson':>10}{'Spearman':>11}{'p':>9}  판정")
    print('-' * 104)

    out = {}
    for key, label, members in GROUPS:
        pairs = [(r['group_scores'][key], r['tier']) for r in scored
                 if r['group_scores'].get(key) is not None]
        if len(pairs) < 4:
            print(f"{label:<12}{len(pairs):>4}  표본 부족")
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        pr, _ = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        verdict = '유효' if abs(sr) >= 0.6 else ('약함' if abs(sr) >= 0.3 else '무관')
        print(f"{label:<12}{len(pairs):>4}{pr:>+10.2f}{sr:>+11.2f}{sp:>9.3f}  {verdict}")
        out[key] = abs(sr)

    # 구성 지표도 따로 — 어깨·골반 각각이 등급을 재는지 확인
    print()
    for m in METRICS:
        pairs = [(r['scores'][m], r['tier']) for r in scored
                 if r['scores'].get(m) is not None]
        if len(pairs) < 4:
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        pr, _ = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        print(f"  └ {METRIC_LABELS[m]:<10}{len(pairs):>4}{pr:>+10.2f}{sr:>+11.2f}{sp:>9.3f}")

    return out


def monotonicity(results):
    scored = [r for r in results if r['tier'] is not None]
    print(f"\n{'='*104}")
    print('등급 그룹별 평균 — 하(3,5) / 중(6,7) / 상(8,9)')
    print(f"{'지표':<12}" + ''.join(f'{name:>10}' for name, _ in TIER_GROUPS)
          + '   단조성')
    print('-' * 104)

    for key, label, _ in GROUPS:
        means = []
        for _, tiers in TIER_GROUPS:
            vals = [r['group_scores'][key] for r in scored
                    if r['tier'] in tiers and r['group_scores'].get(key) is not None]
            means.append(float(np.mean(vals)) if vals else None)
        cells = ''.join(f'{f(m):>10}' for m in means)
        ok = (all(m is not None for m in means)
              and means[0] <= means[1] <= means[2])
        print(f"{label:<12}{cells}   {'✅ 만족' if ok else '❌ 불만족'}")

    means = []
    for _, tiers in TIER_GROUPS:
        vals = [r['total'] for r in scored
                if r['tier'] in tiers and r['total'] is not None]
        means.append(float(np.mean(vals)) if vals else None)
    ok = all(m is not None for m in means) and means[0] <= means[1] <= means[2]
    print(f"{'총점':<12}" + ''.join(f'{f(m):>10}' for m in means)
          + f"   {'✅ 만족' if ok else '❌ 불만족'}")


def order_agreement(results):
    """순서일치 — 아무 두 영상을 골랐을 때 점수 순서가 등급 순서와 맞는 비율"""
    scored = [r for r in results if r['tier'] is not None]
    print(f"\n{'='*104}")
    print('순서일치 (등급이 다른 모든 쌍 중 점수 순서가 맞는 비율)')
    for key, label, _ in GROUPS:
        pairs = [r for r in scored if r['group_scores'].get(key) is not None]
        hit = tot = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a, b = pairs[i], pairs[j]
                if a['tier'] == b['tier']:
                    continue
                tot += 1
                if ((a['tier'] - b['tier']) *
                        (a['group_scores'][key] - b['group_scores'][key])) > 0:
                    hit += 1
        if tot:
            print(f"  {label:<12}{hit}/{tot} = {hit/tot:.0%}")


def main():
    calib = calibration.load()
    results = [score_video(r, calib) for r in raw_metrics.load_all()]

    print(f"기준영상 앵커: {', '.join(calib['reference_videos'])}")
    print(f"{'지표':<12}{'opt':>10}{'TAU(2σ)':>10}{'n':>5}{'앵커 산출 영상 수':>16}")
    for m in METRICS:
        c = calib['metrics'][m]
        if c['opt'] is None:
            continue
        print(f"{METRIC_LABELS[m]:<12}{c['opt']:>10.2f}{c['tau']:>10.2f}"
              f"{c['n_all']:>5}{c['n_reference']:>16}")

    table(results)
    correlations(results)
    monotonicity(results)
    order_agreement(results)


if __name__ == '__main__':
    main()
