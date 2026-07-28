# -*- coding: utf-8 -*-
"""프레임 위에 세 지표를 그린다

숫자를 읽지 않아도 차이가 보이게 하는 것이 목적이다.

    헤드업     노란 화살표  어깨중앙 → 눈중앙
    상체각도   파란 호      무릎-엉덩이-어깨 (엉덩이가 꼭짓점)
    어깨-골반  빨강/초록 선 어깨선 / 골반선
"""
import cv2
import numpy as np

from utils.math_utils import angle_with_vertical, calculate_angle

# BGR
YELLOW = (0, 220, 255)
BLUE = (255, 160, 60)
RED = (60, 60, 255)
GREEN = (80, 220, 80)
WHITE = (255, 255, 255)

L_EYE, R_EYE = 2, 5
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26


def _px(landmarks, idx, w, h):
    return int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)


def _mid(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def draw_headup(img, pf, thickness=3):
    """어깨중앙 → 눈중앙 화살표. 각도는 world_landmarks로 계산한다."""
    h, w = img.shape[:2]
    sh_c = _mid(_px(pf.landmarks, L_SH, w, h), _px(pf.landmarks, R_SH, w, h))
    eye_c = _mid(_px(pf.landmarks, L_EYE, w, h), _px(pf.landmarks, R_EYE, w, h))

    wl = pf.world_landmarks
    vec = (wl[L_EYE] + wl[R_EYE]) / 2 - (wl[L_SH] + wl[R_SH]) / 2
    angle = angle_with_vertical(vec)

    # 화살표가 너무 짧아 안 보이므로 3배로 늘린다
    tip = (sh_c[0] + (eye_c[0] - sh_c[0]) * 3, sh_c[1] + (eye_c[1] - sh_c[1]) * 3)
    cv2.arrowedLine(img, sh_c, tip, YELLOW, thickness, tipLength=0.3)
    return angle


def draw_trunk(img, pf, thickness=3):
    """무릎-엉덩이-어깨 각. 좌우 중 visibility가 높은 쪽을 그린다."""
    h, w = img.shape[:2]
    left_vis = pf.visibility[L_KNEE] + pf.visibility[L_HIP] + pf.visibility[L_SH]
    right_vis = pf.visibility[R_KNEE] + pf.visibility[R_HIP] + pf.visibility[R_SH]
    k, hip, s = ((L_KNEE, L_HIP, L_SH) if left_vis >= right_vis
                 else (R_KNEE, R_HIP, R_SH))

    p_k = _px(pf.landmarks, k, w, h)
    p_h = _px(pf.landmarks, hip, w, h)
    p_s = _px(pf.landmarks, s, w, h)

    cv2.line(img, p_k, p_h, BLUE, thickness)
    cv2.line(img, p_h, p_s, BLUE, thickness)

    # 꼭짓점에 호를 그려 각도를 눈에 보이게
    a1 = np.degrees(np.arctan2(p_k[1] - p_h[1], p_k[0] - p_h[0]))
    a2 = np.degrees(np.arctan2(p_s[1] - p_h[1], p_s[0] - p_h[0]))
    if a2 < a1:
        a1, a2 = a2, a1
    if a2 - a1 > 180:
        a1, a2 = a2, a1 + 360
    r = max(18, int(0.035 * h))
    cv2.ellipse(img, p_h, (r, r), 0, a1, a2, BLUE, thickness)

    wl = pf.world_landmarks
    return calculate_angle(wl[k], wl[hip], wl[s])


def draw_rotation(img, pf, thickness=4):
    """어깨선(빨강) / 골반선(초록). 두 선이 나란하면 몸통 분리가 없다는 뜻이다."""
    h, w = img.shape[:2]
    cv2.line(img, _px(pf.landmarks, L_SH, w, h), _px(pf.landmarks, R_SH, w, h),
             RED, thickness)
    cv2.line(img, _px(pf.landmarks, L_HIP, w, h), _px(pf.landmarks, R_HIP, w, h),
             GREEN, thickness)


def annotate(frame, pf, show_values=True):
    """세 오버레이를 모두 그린 사본을 반환"""
    img = frame.copy()
    headup = draw_headup(img, pf)
    trunk = draw_trunk(img, pf)
    draw_rotation(img, pf)

    if show_values:
        h, w = img.shape[:2]
        scale = w / 900
        lines = [(f'head {headup:.0f}', YELLOW), (f'trunk {trunk:.0f}', BLUE)]
        for i, (text, color) in enumerate(lines):
            y = int(38 * scale) + i * int(34 * scale)
            cv2.putText(img, text, (int(14 * scale), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0),
                        max(4, int(5 * scale)), cv2.LINE_AA)
            cv2.putText(img, text, (int(14 * scale), y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color,
                        max(1, int(2 * scale)), cv2.LINE_AA)

    return img, headup, trunk
