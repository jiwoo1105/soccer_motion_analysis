# -*- coding: utf-8 -*-
"""앵커(opt)와 허용편차(TAU) 산출 → calibration.json

opt  = 기준1·기준2의 해당 지표 평균 — 기준영상이 정답 템플릿이므로 최적값으로 삼는다.
TAU  = 2 × std(전체 영상 원시값)

TAU를 2σ로 고정하는 이유: 상관계수가 올라갈 때까지 τ를 돌리는 것은 n=11에서
과적합이다. 2σ는 "이 동작 분포에서 극단값"이라는 통계적 의미가 있고 라벨을
전혀 보지 않고 계산된다.

실행:
    python3.9 -m scoring.calibration
"""
import json
from pathlib import Path

import numpy as np

METRICS = ('headup', 'trunk', 'ratio')
METRIC_LABELS = {'headup': '헤드업', 'trunk': '상체각도', 'ratio': '어깨-골반'}

CALIB_PATH = Path('output/calibration.json')
REFERENCE_KEYWORD = '기준'


def build(results):
    """원시값 리스트 → 캘리브레이션 dict"""
    refs = [r for r in results if REFERENCE_KEYWORD in r['video']]
    if not refs:
        raise ValueError('기준영상이 없다. opt를 산출할 수 없다.')

    calib = {'reference_videos': [r['video'] for r in refs], 'metrics': {}}

    for m in METRICS:
        ref_vals = [r[m] for r in refs if r.get(m) is not None]
        all_vals = [r[m] for r in results if r.get(m) is not None]

        if not ref_vals or len(all_vals) < 2:
            calib['metrics'][m] = {'opt': None, 'tau': None,
                                   'note': '값이 부족해 산출 불가'}
            continue

        opt = float(np.mean(ref_vals))
        tau = float(2 * np.std(all_vals, ddof=1))

        calib['metrics'][m] = {
            'opt': opt,
            'tau': tau,
            'n_reference': len(ref_vals),
            'n_all': len(all_vals),
            'std': float(np.std(all_vals, ddof=1)),
        }

    return calib


def save(calib, path=CALIB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    return path


def load(path=CALIB_PATH):
    with open(path) as f:
        return json.load(f)


def main():
    from scoring.raw_metrics import load_all
    results = load_all()
    calib = build(results)
    save(calib)

    print(f"기준영상: {', '.join(calib['reference_videos'])}\n")
    print(f"{'지표':<10}{'opt':>10}{'TAU(2σ)':>12}{'σ':>10}")
    print('-' * 42)
    for m in METRICS:
        c = calib['metrics'][m]
        if c['opt'] is None:
            print(f"{METRIC_LABELS[m]:<10}{'N/A':>10}")
            continue
        print(f"{METRIC_LABELS[m]:<10}{c['opt']:>10.3f}{c['tau']:>12.3f}{c['std']:>10.3f}")

    # 근거 문서가 어깨-골반 TAU 초기값으로 0.40을 제시했다. 채택본은 2σ이고,
    # 0.40은 참고값으로만 병기한다. 둘 중 상관이 높은 쪽을 고르는 것은 튜닝이다.
    r = calib['metrics']['ratio']
    if r['opt'] is not None:
        print(f"\n참고 — 어깨-골반 TAU: 채택 2σ={r['tau']:.3f} / 문서 제시값 0.40")

    print(f'\n저장: {CALIB_PATH}')


if __name__ == '__main__':
    main()
