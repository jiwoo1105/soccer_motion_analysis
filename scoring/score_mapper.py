# -*- coding: utf-8 -*-
"""원시값 → 10점 변환

세 지표 모두 같은 식을 쓴다:

    score = clamp(0, 10, (TAU - |x - opt|) / TAU * 10)

기준영상 값(opt)이 최적이고 거기서 멀어질수록 감점한다. 이 형태를 쓰면
"값이 클수록 좋은가 작을수록 좋은가"라는 방향성 논쟁이 사라진다.
(head_pose_analyzer.py의 주석은 '작을수록 좋음', 프로젝트 메모리는 '클수록 좋음'으로
서로 충돌하고 있었는데, 앵커 방식에서는 그 충돌이 성립하지 않는다.)

가중치는 Phase 4에서 `|Spearman ρ|` 순위대로 0.5 / 0.3 / 0.2 세 고정값을 배정한다.
라벨에 맞춰 연속값을 fitting하지 않는다 — n=11에서 3개 가중치 학습은 과적합이다.
"""
from scoring.calibration import METRICS, METRIC_LABELS, load

# Phase 4에서 확정. 확정 전에는 동등 가중으로 동작한다.
DEFAULT_WEIGHTS = {m: 1.0 / len(METRICS) for m in METRICS}


def map_score(x, opt, tau):
    """원시값 → 0~10점. 계산 불가면 None"""
    if x is None or opt is None or tau is None or tau <= 0:
        return None
    return max(0.0, min(10.0, (tau - abs(x - opt)) / tau * 10))


def score_video(raw, calib=None, weights=None):
    """원시값 dict → 지표별 점수 + 총점

    헤드업이 N/A면 남은 지표의 가중치를 재정규화하고 partial 플래그를 세운다.
    """
    calib = calib or load()
    weights = weights or DEFAULT_WEIGHTS

    scores = {}
    for m in METRICS:
        c = calib['metrics'].get(m, {})
        scores[m] = map_score(raw.get(m), c.get('opt'), c.get('tau'))

    available = {m: w for m, w in weights.items() if scores.get(m) is not None}
    total_weight = sum(available.values())

    if total_weight > 0:
        total = sum(scores[m] * w for m, w in available.items()) / total_weight
    else:
        total = None

    return {
        'video': raw.get('video'),
        'tier': raw.get('tier'),
        'scores': scores,
        'total': total,
        'partial': len(available) < len(METRICS),
        'missing': [m for m in METRICS if scores.get(m) is None],
        'touch_quality': raw.get('touch_quality'),
        'touch_valid': raw.get('touch_valid'),
        'touch_total': raw.get('touch_total'),
        'raw': {m: raw.get(m) for m in METRICS},
    }


def format_report(result, calib=None):
    """CLI 출력용 문자열"""
    calib = calib or load()
    s, raw = result['scores'], result['raw']
    f = lambda v, d=1: (f'{v:.{d}f}' if v is not None else 'N/A')

    lines = [result['video'], '─' * 46]
    for m in METRICS:
        opt = calib['metrics'].get(m, {}).get('opt')
        detail = f"(측정 {f(raw[m],2)}, 기준 {f(opt,2)})"
        extra = ''
        if m == 'headup':
            extra = f"  유효터치 {result['touch_valid']}/{result['touch_total']}"
        lines.append(f"{METRIC_LABELS[m]:<9}{f(s[m]):>6} / 10   {detail}{extra}")
    lines.append('─' * 46)
    lines.append(f"{'총점':<9}{f(result['total']):>6} / 10"
                 + ('   ⚠ 부분산출' if result['partial'] else ''))
    lines.append('⚠ 어깨-골반 비율은 3점대 판별에만 검증됨 (5~9점대 구분 불가)')
    return '\n'.join(lines)
