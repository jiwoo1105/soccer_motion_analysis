# -*- coding: utf-8 -*-
"""3영상 파노라마 비교 그림

레퍼런스 / 높은 점수 / 낮은 점수 영상을 한 장에 놓고, 각 행에 프레임을 가로로
나열해 세 지표를 눈으로 비교한다. 오른쪽에 지표별 점수와 총점을 붙인다.

프레임은 **영상 전체 등간격**으로 뽑는다. 점수는 터치 윈도우 기준으로 산출되므로
둘의 구간이 일치하지 않는다 — 캡션에 명시한다.

실행:
    python3.9 make_panorama.py
    python3.9 make_panorama.py --videos "인,인 기준1" "인,인 7-2" "인,인 3-1"
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scoring.extraction_cache import load, is_cached, VIDEOS
from scoring.overlay import annotate
from scoring import raw_metrics, calibration
from scoring.score_mapper import score_video
from scoring.calibration import METRICS, METRIC_LABELS

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

N_FRAMES = 6
CROP_ASPECT = 3 / 4      # 가로/세로 — 사람이 서 있으므로 세로로 긴 크롭
STRIP_HEIGHT = 420
OUT_PATH = 'output/figures/panorama_comparison.png'


def video_path_for(stem):
    for v in VIDEOS:
        if Path(v).stem == stem:
            return v
    raise FileNotFoundError(f'영상 경로를 찾을 수 없다: {stem}')


def person_crop(landmarks, w, h, margin=0.35):
    """랜드마크를 감싸는 크롭 영역 (가로/세로 = CROP_ASPECT 고정)

    1920x1080 원본을 그대로 6장 늘어놓으면 사람이 너무 작아 자세가 안 보인다.
    """
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box_h = (y1 - y0) * (1 + margin)
    box_w = box_h * CROP_ASPECT
    if box_w < (x1 - x0) * (1 + margin):
        box_w = (x1 - x0) * (1 + margin)
        box_h = box_w / CROP_ASPECT

    x0 = int(max(0, cx - box_w / 2))
    x1 = int(min(w, cx + box_w / 2))
    y0 = int(max(0, cy - box_h / 2))
    y1 = int(min(h, cy + box_h / 2))
    return x0, y0, x1, y1


def build_strip(stem, n_frames=N_FRAMES, height=STRIP_HEIGHT):
    """영상 전체 등간격 n장을 오버레이해서 가로로 이어붙인 이미지"""
    pose_frames, _ = load(stem)
    frame_map = {pf.frame_number: pf for pf in pose_frames}
    available = sorted(frame_map)

    # 양 끝을 피해 내부를 등간격으로 — 첫/마지막 프레임은 자세가 준비 안 된 경우가 많다
    picks = [available[int(len(available) * (i + 0.5) / n_frames)]
             for i in range(n_frames)]

    cap = cv2.VideoCapture(video_path_for(stem))
    tiles = []
    for fn in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        pf = frame_map[fn]
        img, _, _ = annotate(frame, pf)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = person_crop(pf.landmarks, w, h)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = height / crop.shape[0]
        tiles.append(cv2.resize(crop, (int(crop.shape[1] * scale), height)))
    cap.release()

    if not tiles:
        return None

    # 흰 구분선을 끼워 프레임 경계를 명확히
    sep = np.full((height, 4, 3), 255, dtype=np.uint8)
    parts = []
    for i, t in enumerate(tiles):
        if i:
            parts.append(sep)
        parts.append(t)
    strip = np.hstack(parts)
    return cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)


def deviation_color(result, calib):
    """앵커 대비 편차를 행 테두리 색으로. 기준영상은 회색."""
    if result['tier'] is None:
        return '#888888'
    total = result['total']
    if total is None:
        return '#888888'
    # 10점 → 초록, 0점 → 빨강
    t = max(0.0, min(1.0, total / 10))
    return matplotlib.colors.to_hex((1 - t * 0.85, 0.25 + t * 0.55, 0.25))


def make(stems, out_path=OUT_PATH):
    calib = calibration.load()
    results = {s: score_video(raw_metrics.compute(s), calib) for s in stems}
    strips = {s: build_strip(s) for s in stems}

    rows = len(stems)
    fig = plt.figure(figsize=(20, 3.9 * rows))
    gs = fig.add_gridspec(rows, 2, width_ratios=[5.2, 1], hspace=0.22, wspace=0.03)

    role = ['레퍼런스', '높은 점수', '낮은 점수']

    for i, s in enumerate(stems):
        r = results[s]
        color = deviation_color(r, calib)

        ax = fig.add_subplot(gs[i, 0])
        if strips[s] is not None:
            ax.imshow(strips[s])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)
        tier = '기준' if r['tier'] is None else f"{r['tier']}점대"
        ax.set_title(f"{s}  —  {role[i] if i < len(role) else ''} ({tier})",
                     fontsize=15, fontweight='bold', loc='left', pad=8)

        # 점수 패널 — 좌표는 전부 axes 비율(0~1). 데이터 좌표를 섞으면
        # 텍스트가 축 밖으로 나가 그림 전체가 늘어난다.
        axp = fig.add_subplot(gs[i, 1])
        axp.set_xlim(0, 1)
        axp.set_ylim(0, 1)
        axp.axis('off')
        T = axp.transAxes
        f = lambda v, d=1: (f'{v:.{d}f}' if v is not None else 'N/A')

        for j, m in enumerate(METRICS):
            y = 0.94 - j * 0.20
            axp.text(0.04, y, METRIC_LABELS[m], fontsize=13, va='top', transform=T)
            axp.text(0.96, y, f(r['scores'][m]), fontsize=16, va='top',
                     ha='right', fontweight='bold', transform=T)
            axp.text(0.04, y - 0.085, f"측정 {f(r['raw'][m], 2)}", fontsize=9,
                     va='top', color='#666666', transform=T)

        axp.plot([0.04, 0.96], [0.33, 0.33], color='#333333', lw=1.2,
                 transform=T, clip_on=False)
        axp.text(0.04, 0.27, '총점', fontsize=15, va='top', fontweight='bold',
                 transform=T)
        axp.text(0.96, 0.29, f(r['total']), fontsize=24, va='top', ha='right',
                 fontweight='bold', color=color, transform=T)

        note = f"유효터치 {r['touch_valid']}/{r['touch_total']}"
        if r['partial']:
            note += '  ⚠ 부분산출'
        axp.text(0.04, 0.10, note, fontsize=9, va='top', color='#666666',
                 transform=T)

    fig.suptitle('드리블 3지표 비교 — 레퍼런스 / 높은 점수 / 낮은 점수',
                 fontsize=19, fontweight='bold', y=0.985)
    fig.text(0.5, 0.012,
             '노란 화살표 = 헤드업(어깨중앙→눈중앙)   파란 호 = 상체각도(무릎-엉덩이-어깨)   '
             '빨강/초록 선 = 어깨선/골반선\n'
             '프레임은 영상 전체 등간격, 점수는 터치 윈도우 기준 산출',
             ha='center', fontsize=11, color='#444444')

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'저장: {out_path}')

    f1 = lambda v: (f'{v:.1f}' if v is not None else 'N/A')
    for s in stems:
        r = results[s]
        parts = ', '.join(f"{METRIC_LABELS[m]} {f1(r['scores'][m])}" for m in METRICS)
        print(f"{s}: {parts} | 총점 {f1(r['total'])}")


def auto_select():
    """레퍼런스 / 최고 점수대 / 최저 점수대 자동 선택 (터치 품질 우선)"""
    results = raw_metrics.load_all()
    scored = [r for r in results if r['tier'] is not None]
    refs = [r for r in results if r['tier'] is None]
    if not scored or not refs:
        raise ValueError('점수대 영상 또는 기준영상이 부족하다')

    key = lambda r: (r['touch_quality'], r['touch_valid'])
    ref = max(refs, key=key)
    high = max([r for r in scored if r['tier'] >= 7], key=key, default=None)
    low = max([r for r in scored if r['tier'] <= 5], key=key, default=None)
    if high is None or low is None:
        raise ValueError('높은/낮은 점수대 영상이 부족하다')
    return [ref['video'], high['video'], low['video']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='*', help='stem 3개 (레퍼런스, 높은점수, 낮은점수)')
    ap.add_argument('--out', default=OUT_PATH)
    args = ap.parse_args()

    stems = args.videos if args.videos else auto_select()
    missing = [s for s in stems if not is_cached(s)]
    if missing:
        print(f"캐시 없음: {', '.join(missing)}")
        return
    make(stems, args.out)


if __name__ == '__main__':
    main()
