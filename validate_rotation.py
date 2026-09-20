# -*- coding: utf-8 -*-
"""회전 후보 지표를 독립 표본(인,아웃)으로 검증한다

in_in 10편에서는 후보를 83개 던져도 순열검정을 통과하지 못했다 (best |rho|=0.79,
귀무 95분위 0.84, p=0.12). n=10에서 best-of-K의 우연 상한이 너무 높아서다.
그래서 in_in에서 뽑은 후보를 **미리 정하고** in_out에서 그대로 재현되는지만 본다.
후보를 미리 고정하면 다중비교가 사라져 단일 검정이 된다.

실행:
    python3.9 validate_rotation.py
"""
import os
import sys
import json
import glob
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

from scoring import rotation_signal as rs

CACHE = Path('output/scoring_cache')
W, FPS = 8, 30

# in_in에서 상위로 올라온 후보. **여기서 더 늘리지 마라** — 늘리는 순간
# 다중비교가 되살아나 검증의 의미가 사라진다.
CANDIDATES = ['tw_rng', 'T_w_sh', 'T_w_ratio', 'T_sh_rng', 'T_pe_rng']


def nfc(s):
    """macOS는 파일명을 NFD(자모 분리)로 돌려준다. 그대로 '기준' in stem 을 하면
    항상 False가 나온다. 문자열 비교 전에 NFC로 합쳐야 한다."""
    import unicodedata
    return unicodedata.normalize('NFC', s)


def tier_of(stem):
    """파일명에서 점수대. 기준영상은 None"""
    stem = nfc(stem)
    if '기준' in stem:
        return None
    m = re.search(r'(\d+)(?:-\d+)?\s*$', stem)
    return int(m.group(1)) if m else None


def touches(stem):
    from scoring.extraction_cache import load
    from scoring.touch_filter import filter_touches
    from extract_depth_metrics import detect_touches_by_ball_direction
    pose_frames, ball_tracks = load(stem)
    t = detect_touches_by_ball_direction(pose_frames, ball_tracks, max_foot_dist=200)
    valid, _ = filter_touches(t, pose_frames, ball_tracks)
    return [x.frame_number for x in valid]


def smooth(wl, li, ri):
    a = rs.z_axis_angle(wl, li, ri)
    bad, _ = rs.detect_spikes_hampel(a)
    c = a.copy()
    g = np.where(~bad)[0]
    if bad.any():
        c[bad] = np.interp(np.where(bad)[0], g, c[g])
    return savgol_filter(c, min(41, len(c) | 1), 2)


def features(stem, frames):
    z = np.load(CACHE / f'{stem}_pose.npz')
    wl, lm = z['world_landmarks'], z['landmarks']
    fw, fh = float(z['frame_width']), float(z['frame_height'])
    n = len(wl)

    sa, sp = smooth(wl, 11, 12), smooth(wl, 23, 24)
    tw = sa - sp
    ms = np.stack([(lm[:, 11, 0] + lm[:, 12, 0]) / 2 * fw,
                   (lm[:, 11, 1] + lm[:, 12, 1]) / 2 * fh], 1)
    mh = np.stack([(lm[:, 23, 0] + lm[:, 24, 0]) / 2 * fw,
                   (lm[:, 23, 1] + lm[:, 24, 1]) / 2 * fh], 1)
    torso = np.maximum(np.linalg.norm(ms - mh, axis=1), 1e-6)
    ws = savgol_filter(np.hypot((lm[:, 12, 0] - lm[:, 11, 0]) * fw,
                                (lm[:, 12, 1] - lm[:, 11, 1]) * fh) / torso, 21, 2)
    wp = savgol_filter(np.hypot((lm[:, 24, 0] - lm[:, 23, 0]) * fw,
                                (lm[:, 24, 1] - lm[:, 23, 1]) * fh) / torso, 21, 2)

    F = {'tw_rng': float(np.percentile(tw, 95) - np.percentile(tw, 5)),
         'n_touch': len(frames)}
    if frames:
        idx = np.unique(np.concatenate(
            [np.arange(max(0, f - W), min(n, f + W + 1)) for f in frames]))
        F['T_w_sh'] = float(ws[idx].mean())
        F['T_w_ratio'] = float(np.median(ws[idx] / np.maximum(wp[idx], 1e-6)))
        F['T_sh_rng'] = float(np.mean([sa[max(0, f - W):f + W + 1].ptp() for f in frames]))
        F['T_pe_rng'] = float(np.mean([sp[max(0, f - W):f + W + 1].ptp() for f in frames]))
    else:
        for k in ('T_w_sh', 'T_w_ratio', 'T_sh_rng', 'T_pe_rng'):
            F[k] = np.nan
    return F


# 좌우 랜드마크가 스왑돼 복구 불가. rotation_measurements.EXCLUDED와 같은 이유다.
EXCLUDED = ('인,인 7-3',)


def collect(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        stem = Path(path).stem
        if nfc(stem) in EXCLUDED:
            print(f'  [제외] {nfc(stem)} — 좌우 랜드마크 스왑')
            continue
        if not (CACHE / f'{stem}_pose.npz').exists():
            print(f'  [캐시 없음] {stem}')
            continue
        try:
            out[stem] = features(stem, touches(stem))
            out[stem]['tier'] = tier_of(stem)
        except Exception as e:
            print(f'  [{stem}] 오류: {e}')
    return out


def report(name, data):
    labeled = {k: v for k, v in data.items() if v['tier'] is not None}
    tiers = [v['tier'] for v in labeled.values()]
    print(f'\n=== {name}  (라벨 {len(labeled)}편 / 전체 {len(data)}편) ===')
    print(f"{'영상':<16}{'등급':>4}{'터치':>5}" + ''.join(f'{c:>11}' for c in CANDIDATES))
    for k, v in sorted(data.items(), key=lambda x: (x[1]['tier'] or 99)):
        t = v['tier'] if v['tier'] is not None else '기준'
        print(f"{k[-12:]:<16}{str(t):>4}{v['n_touch']:>5}"
              + ''.join(f"{v[c]:>11.3f}" if np.isfinite(v[c]) else f"{'N/A':>11}"
                        for c in CANDIDATES))
    res = {}
    for c in CANDIDATES:
        vals = [v[c] for v in labeled.values()]
        ok = [i for i, x in enumerate(vals) if np.isfinite(x)]
        if len(ok) < 5:
            res[c] = (np.nan, np.nan, 0)
            continue
        rho, p = spearmanr([tiers[i] for i in ok], [vals[i] for i in ok])
        res[c] = (rho, p, len(ok))
    print(f"\n{'지표':<12}{'rho':>8}{'p':>8}{'n':>4}")
    for c in CANDIDATES:
        rho, p, n = res[c]
        mark = ' ***' if (np.isfinite(p) and p < 0.05) else ''
        print(f'  {c:<12}{rho:>+8.2f}{p:>8.3f}{n:>4}{mark}')
    return res


def main():
    print('인,인 (탐색 표본) 과 인,아웃 (검증 표본) 을 각각 계산한다')
    # 한글이 든 glob 패턴은 NFD/NFC 불일치로 매칭에 실패한다. ASCII 패턴만 쓴다.
    a = collect('input/in_in/*.MOV')
    b = collect('input/in_out/*.MOV')
    ra = report('인,인', a)
    rb = report('인,아웃', b)

    print('\n' + '=' * 56)
    print('검증 결과 — in_in에서 나온 후보가 in_out에서도 같은 방향인가')
    print(f"{'지표':<12}{'in_in rho':>11}{'in_out rho':>12}{'in_out p':>10}   판정")
    print('-' * 56)
    for c in CANDIDATES:
        r1 = ra.get(c, (np.nan,))[0]
        r2, p2, n2 = rb.get(c, (np.nan, np.nan, 0))
        if not np.isfinite(r2):
            verdict = '검증 불가'
        elif r1 * r2 <= 0:
            verdict = '✗ 부호 반전 — 기각'
        elif p2 < 0.05:
            verdict = '✓ 재현됨'
        else:
            verdict = '△ 방향만 일치 (유의하지 않음)'
        print(f'  {c:<12}{r1:>+11.2f}{r2:>+12.2f}{p2:>10.3f}   {verdict}')

    with open('output/rotation_validation.json', 'w') as f:
        json.dump({'in_in': {k: {kk: (None if not np.isfinite(vv) else vv)
                                 if isinstance(vv, float) else vv
                                 for kk, vv in v.items()} for k, v in a.items()},
                   'in_out': {k: {kk: (None if not np.isfinite(vv) else vv)
                                  if isinstance(vv, float) else vv
                                  for kk, vv in v.items()} for k, v in b.items()}},
                  f, ensure_ascii=False, indent=1)
    print('\n저장: output/rotation_validation.json')


if __name__ == '__main__':
    main()
