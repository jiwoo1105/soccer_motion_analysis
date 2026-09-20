# -*- coding: utf-8 -*-
"""인,아웃 16편을 캐시로 추출 — 회전 지표 검증용 독립 표본

in_in 10편(기준 제외)에서는 후보를 아무리 던져도 순열검정을 통과하지 못했다.
표본이 n=10이라 best-of-K의 귀무 상한(|rho| 0.84)이 너무 높기 때문이다.
독립 표본이 있으면 in_in에서 찾은 후보를 '검증'할 수 있다.

실행: python3.9 extract_in_out.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pathlib import Path
from scoring.extraction_cache import build, is_cached

VIDEOS = sorted(glob.glob('input/in_out/*.MOV'))

if __name__ == '__main__':
    print(f'대상 {len(VIDEOS)}편')
    ok = 0
    for v in VIDEOS:
        stem = Path(v).stem
        if is_cached(stem):
            print(f'  [캐시됨] {stem}', flush=True)
            ok += 1
            continue
        try:
            ok += bool(build(v))
        except Exception as e:
            print(f'  [{stem}] 오류: {e}', flush=True)
    print(f'\n완료 {ok}/{len(VIDEOS)}')
