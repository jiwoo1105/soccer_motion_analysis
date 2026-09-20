"""
어깨/골반 회전 각도 trace 영상 생성
매 프레임마다 skeleton + 어깨/골반 XZ 방향각을 영상에 오버레이
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path
from core.pose_extractor import PoseExtractor
from visualization.skeleton_drawer import SkeletonDrawer
from scipy.signal import savgol_filter
import config


def compute_angles_and_velocity(pose_frames):
    """각 프레임의 어깨/골반 XZ 각도 + 변화율 계산 (savgol X 안정화)"""
    n = len(pose_frames)

    sh_vx = np.zeros(n)
    sh_vz = np.zeros(n)
    pe_vx = np.zeros(n)
    pe_vz = np.zeros(n)

    for i, pf in enumerate(pose_frames):
        wl = pf.world_landmarks
        sh_vec = wl[12] - wl[11]
        sh_vx[i], sh_vz[i] = sh_vec[0], sh_vec[2]
        pe_vec = wl[24] - wl[23]
        pe_vx[i], pe_vz[i] = pe_vec[0], pe_vec[2]

    # X만 savgol smoothing
    win = min(21, n if n % 2 == 1 else n - 1)
    if win >= 3:
        sh_vx = savgol_filter(sh_vx, window_length=win, polyorder=2)
        pe_vx = savgol_filter(pe_vx, window_length=win, polyorder=2)

    sh_deg = np.degrees(np.arctan2(sh_vz, sh_vx))
    pe_deg = np.degrees(np.arctan2(pe_vz, pe_vx))

    # 변화율: 전 프레임 대비 변화량 (10°/f 클리핑)
    sh_vel = np.zeros(n)
    pe_vel = np.zeros(n)
    sh_vel[1:] = np.clip(np.abs(np.diff(sh_deg)), 0, 10)
    pe_vel[1:] = np.clip(np.abs(np.diff(pe_deg)), 0, 10)

    return sh_deg, pe_deg, sh_vel, pe_vel


def draw_rotation_info(frame, sh_vel, pe_vel, sh_mean, pe_mean, frame_idx, landmarks):
    """프레임에 변화율 정보 오버레이"""
    h, w, _ = frame.shape
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    def to_px(idx):
        return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

    # 어깨 라인 위에 변화율
    pt_lsh, pt_rsh = to_px(L_SH), to_px(R_SH)
    sh_cx = (pt_lsh[0] + pt_rsh[0]) // 2
    sh_cy = (pt_lsh[1] + pt_rsh[1]) // 2
    sh_text = f"S: {sh_vel:.1f}/f"
    sh_color = (0, 255, 255)

    # 골반 라인 아래에 변화율
    pt_lhip, pt_rhip = to_px(L_HIP), to_px(R_HIP)
    pe_cx = (pt_lhip[0] + pt_rhip[0]) // 2
    pe_cy = (pt_lhip[1] + pt_rhip[1]) // 2
    pe_text = f"P: {pe_vel:.1f}/f"
    pe_color = (255, 150, 0)

    for text, cx, cy, color, y_off in [
        (sh_text, sh_cx, sh_cy, sh_color, -25),
        (pe_text, pe_cx, pe_cy, pe_color, 30),
    ]:
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        tx = cx - tw // 2
        ty = cy + y_off
        cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(output, text, (tx, ty), font, 0.7, color, 2, cv2.LINE_AA)

    # 좌상단 패널
    lines = [
        (f"Frame: {frame_idx}", (220, 220, 220)),
        (f"S velocity: {sh_vel:.1f} deg/f  (avg {sh_mean:.2f})", sh_color),
        (f"P velocity: {pe_vel:.1f} deg/f  (avg {pe_mean:.2f})", pe_color),
    ]
    line_h = 24
    pad = 8
    max_w = max(cv2.getTextSize(t, font, 0.55, 1)[0][0] for t, _ in lines)
    box_h = len(lines) * line_h + pad * 2

    overlay = output.copy()
    cv2.rectangle(overlay, (10 - pad, 10 - pad), (10 + max_w + pad, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    for i, (text, color) in enumerate(lines):
        ty = 10 + pad + (i + 1) * line_h - 4
        cv2.putText(output, text, (10, ty), font, 0.55, color, 1, cv2.LINE_AA)

    return output


def make_trace_video(video_path, output_path=None):
    stem = Path(video_path).stem

    if output_path is None:
        out_dir = Path('output/videos')
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f'rotation_trace_{stem}.mp4')

    # 포즈 추출
    print(f"포즈 추출 중: {video_path}")
    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        detect_ball=False,
    )
    pose_frames = extractor.extract_from_video(video_path)
    if len(pose_frames) == 0:
        print("포즈 추출 실패")
        return

    # 각도 + 변화율 계산
    sh_deg, pe_deg, sh_vel, pe_vel = compute_angles_and_velocity(pose_frames)
    pose_dict = {pf.frame_number: (pf, i) for i, pf in enumerate(pose_frames)}

    sh_mean = float(np.mean(sh_vel[1:]))
    pe_mean = float(np.mean(pe_vel[1:]))
    print(f"어깨 평균 변화율: {sh_mean:.2f}°/frame")
    print(f"골반 평균 변화율: {pe_mean:.2f}°/frame")

    # 비디오 생성
    drawer = SkeletonDrawer(color=(0, 255, 0))
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"trace 영상 생성 중... → {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in pose_dict:
            pf, arr_idx = pose_dict[frame_idx]

            # 스켈레톤
            frame = drawer.draw_skeleton(frame, pf.landmarks)

            # 어깨/골반 방향 화살표
            frame = drawer.draw_body_direction_vectors(frame, pf.landmarks)

            # 변화율 정보 오버레이
            frame = draw_rotation_info(
                frame, sh_vel[arr_idx], pe_vel[arr_idx],
                sh_mean, pe_mean,
                frame_idx, pf.landmarks
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "input/in_in/인,인 3-1.MOV"
    make_trace_video(video)
