# -*- coding: utf-8 -*-
"""어깨·골반 회전 진폭 — 확정 측정값 (z축)

이 표가 회전 지표의 **단일 진실 공급원**이다. 재추출하지 마라.

측정 방법 (검증 완료):
    신호원   : world landmark z축 — atan2(vz, vx)
    파이프라인: unwrap → Hampel(half_win=5, τ=10°) → 선형보간 → savgol(41,2)
              → 드리프트 제거(savgol 81,2) → find_peaks(distance=10, prominence=3)
              → 연속 극값 차의 평균
    측정 구간: 영상 전체

제외: 7-3 — 최대 각속도 168~179°/frame으로 좌우 랜드마크가 스왑됐다. 복구 불가.

────────────────────────────────────────────────────────────────
2D 폭 방식(arccos(w/w_max) + 터치 중앙값 차)은 보류됐다. 되돌리지 마라.
  ① 13편 중 5편이 유효 터치 부족으로 N/A (6-1, 7-3, 8-2, 9-2, 기준1).
     기준1이 N/A라 앵커가 기준2 단일 영상에 걸린다.
  ② 8-1이 어깨 0.3° / 골반 2.2° — 생리학적으로 불가능한 값. 측정 실패다.
  ③ 5-1이 z축 9.4°(최저) → 2D 44.9°(최대급)로 역전.
     arccos(w/w_max)는 w ≈ w_max(정면)에서 도함수가 발산해 노이즈가 증폭된다.
z축은 12편 전부 값이 나오고 Hampel 4방식 비교 검증을 마쳤다.
────────────────────────────────────────────────────────────────
"""

# {영상 stem: (어깨 진폭°, 골반 진폭°)}
ROTATION_AMPLITUDE = {
    '인,인 3-1': (26.2, 21.7),
    '인,인 3-2': (19.7, 19.6),
    '인,인 5-1': (9.4, 11.1),
    '인,인 6-1': (16.3, 23.1),
    '인,인 7-1': (22.7, 30.3),
    '인,인 7-2': (21.3, 28.8),
    '인,인 8-1': (15.1, 20.3),
    '인,인 8-2': (11.1, 17.0),
    '인,인 9-1': (22.4, 33.6),
    '인,인 9-2': (16.6, 22.5),
    '인,인 기준1': (26.2, 29.3),
    '인,인 기준2': (22.4, 31.5),
}

# 좌우 랜드마크 스왑으로 복구 불가. 회전뿐 아니라 모든 지표에서 제외한다.
EXCLUDED = ('인,인 7-3',)


def shoulder(stem):
    v = ROTATION_AMPLITUDE.get(stem)
    return v[0] if v else None


def pelvis(stem):
    v = ROTATION_AMPLITUDE.get(stem)
    return v[1] if v else None


def verify(tolerance=1.0):
    """표의 값이 캐시된 포즈에서 실제로 재현되는지 확인한다.

    표를 손으로 옮겨 적은 값이라 조용히 어긋날 수 있다. 재추출 없이 캐시된
    world_landmarks에 z축 파이프라인을 다시 돌려 대조한다. 몇 초면 끝난다.

    실행: python3.9 -m scoring.rotation_measurements

    Returns:
        (전부 통과했는가, [(영상, 부위, 표값, 재현값, 차이) …])
    """
    import numpy as np
    from pathlib import Path
    from scoring import rotation_signal as rs
    from scoring.extraction_cache import load, is_cached, VIDEOS

    rows = []
    for v in VIDEOS:
        stem = Path(v).stem
        if stem not in ROTATION_AMPLITUDE or not is_cached(stem):
            continue
        wl = np.array([p.world_landmarks for p in load(stem)[0]])
        for part, (a, b), expected in (
            ('어깨', (11, 12), ROTATION_AMPLITUDE[stem][0]),
            ('골반', (23, 24), ROTATION_AMPLITUDE[stem][1]),
        ):
            got, _, _, _ = rs.measure(wl, a, b)
            diff = abs(got - expected) if got is not None else None
            rows.append((stem, part, expected, got, diff))

    bad = [r for r in rows if r[4] is None or r[4] > tolerance]
    return not bad, rows


def main():
    ok, rows = verify()
    print(f"{'영상':<13}{'부위':>5}{'표':>8}{'재현':>8}{'차이':>8}")
    print('-' * 44)
    for stem, part, exp, got, diff in rows:
        g = lambda v: (f'{v:.1f}' if v is not None else 'N/A')
        print(f"{stem:<13}{part:>5}{exp:>8.1f}{g(got):>8}{g(diff):>8}")

    diffs = [r[4] for r in rows if r[4] is not None]
    if diffs:
        print(f"\n평균 절대차 {sum(diffs)/len(diffs):.2f}° / 최대 {max(diffs):.2f}° (n={len(rows)})")
    print('통과' if ok else '⚠ 표와 캐시가 어긋난다 — 표를 다시 확인하라')


if __name__ == '__main__':
    main()
