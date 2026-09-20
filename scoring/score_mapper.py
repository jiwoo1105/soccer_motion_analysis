# -*- coding: utf-8 -*-
"""Legacy calibration/rotation scoring: 원시값 → 10점 변환.

기존 가중치(상체 0.5 / 헤드업 0.3 / 회전 0.2)를 보존한 재현용 경로다.
현재 동일 비중 평가는 evaluate.py와 scoring.current_metrics를 사용한다.

세 지표 모두 같은 식을 쓴다:

    score = clamp(0, 10, (TAU - |x - opt|) / TAU * 10)

기준영상 값(opt)이 최적이고 거기서 멀어질수록 감점한다. 이 형태를 쓰면
"값이 클수록 좋은가 작을수록 좋은가"라는 방향성 논쟁이 사라진다.
(head_pose_analyzer.py의 주석은 '작을수록 좋음', 프로젝트 메모리는 '클수록 좋음'으로
서로 충돌하고 있었는데, 앵커 방식에서는 그 충돌이 성립하지 않는다.)

가중치는 Phase 4에서 `|Spearman ρ|` 순위대로 0.5 / 0.3 / 0.2 세 고정값을 배정한다.
라벨에 맞춰 연속값을 fitting하지 않는다 — n=11에서 3개 가중치 학습은 과적합이다.
"""
import numpy as np

from scoring.calibration import METRICS, METRIC_LABELS, load

# 채점 단위는 지표가 아니라 **그룹**이다.
#
# 어깨와 골반은 교차상관 0.93으로 사실상 같은 신호다. 네 지표를 동등 가중하면
# 총점의 절반을 회전이 차지해 같은 정보가 두 번 계산된다. 그래서 두 지표를
# 한 그룹으로 묶어 평균을 낸 뒤 아래의 기존 0.5 / 0.3 / 0.2 가중치를 적용한다.
#
# 앵커(opt)는 어깨 27.19° / 골반 24.51°로 서로 달라서 그대로 분리해 둔다.
# 원시값을 먼저 평균하면 두 앵커가 섞여 의미가 사라진다.
# 점수를 낸 뒤 평균해야 각자의 기준에서 잰 편차가 보존된다.
GROUPS = (
    ('headup', '헤드업', ('headup',)),
    ('trunk', '상체각도', ('trunk',)),
    ('rotation', '어깨-골반', ('shoulder', 'pelvis')),
)

# Phase 3 검증(validate_scoring.py, n=10) 결과의 |Spearman ρ| 순위로 배정했다.
# 0.5 / 0.3 / 0.2 세 고정값만 쓴다 — 연속값 fitting은 n=10에서 과적합이다.
#
#   상체각도   ρ = +0.70 (p=0.023), 순서일치 76%, 단조성 만족   → 0.5
#   헤드업     ρ = +0.26 (p=0.572), 순서일치 42%                → 0.3
#   어깨-골반  ρ = −0.05 (p=0.893), 순서일치 49%                → 0.2
#
# 등급을 실제로 재는 건 상체각도 하나뿐이다. 헤드업과 어깨-골반은 무작위 수준이라
# 총점에 남겨두되 기여를 최소화했다. 숫자를 맞추려고 손으로 조정하지 마라.
DEFAULT_WEIGHTS = {
    'trunk': 0.5,
    'headup': 0.3,
    'rotation': 0.2,
}


def map_score(x, opt, tau):
    """원시값 → 0~10점. 계산 불가면 None"""
    if x is None or opt is None or tau is None or tau <= 0:
        return None
    return max(0.0, min(10.0, (tau - abs(x - opt)) / tau * 10))


def score_video(raw, calib=None, weights=None):
    """원시값 dict → 지표별 점수 + 그룹 점수 + 총점

    산출 불가한 그룹이 있으면 남은 그룹의 가중치를 재정규화하고
    partial 플래그를 세운다.
    """
    calib = calib or load()
    weights = weights or DEFAULT_WEIGHTS

    scores = {}
    for m in METRICS:
        c = calib['metrics'].get(m, {})
        scores[m] = map_score(raw.get(m), c.get('opt'), c.get('tau'))

    group_scores = {}
    for key, _, members in GROUPS:
        vals = [scores[m] for m in members if scores.get(m) is not None]
        group_scores[key] = float(np.mean(vals)) if vals else None

    available = {k: w for k, w in weights.items() if group_scores.get(k) is not None}
    total_weight = sum(available.values())

    if total_weight > 0:
        total = sum(group_scores[k] * w for k, w in available.items()) / total_weight
    else:
        total = None

    return {
        'video': raw.get('video'),
        'tier': raw.get('tier'),
        'scores': scores,
        'group_scores': group_scores,
        'total': total,
        'partial': len(available) < len(GROUPS),
        'missing': [k for k, _, _ in GROUPS if group_scores.get(k) is None],
        'touch_quality': raw.get('touch_quality'),
        'touch_valid': raw.get('touch_valid'),
        'touch_total': raw.get('touch_total'),
        'raw': {m: raw.get(m) for m in METRICS},
    }


def format_report(result, calib=None):
    """CLI 출력용 문자열"""
    calib = calib or load()
    s, g, raw = result['scores'], result['group_scores'], result['raw']
    f = lambda v, d=1: (f'{v:.{d}f}' if v is not None else 'N/A')

    lines = [result['video'], '─' * 52]
    for key, label, members in GROUPS:
        extra = ''
        if len(members) == 1:
            m = members[0]
            opt = calib['metrics'].get(m, {}).get('opt')
            extra = f"(측정 {f(raw[m],2)}, 기준 {f(opt,2)})"
        else:
            extra = '(' + ', '.join(
                f"{METRIC_LABELS[m]} {f(s[m])}" for m in members) + ')'
        if key in ('headup', 'rotation'):
            extra += f"  유효터치 {result['touch_valid']}/{result['touch_total']}"
        lines.append(f"{label:<9}{f(g[key]):>6} / 10   {extra}")
    lines.append('─' * 52)
    lines.append(f"{'총점':<9}{f(result['total']):>6} / 10"
                 + ('   ⚠ 부분산출' if result['partial'] else ''))
    return '\n'.join(lines)
