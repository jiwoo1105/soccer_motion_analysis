"""
Legacy SAM2 기반 터치 감지 + 평가지표 추출 보조 모듈.

터치 감지: SAM2 공 2D 궤적의 방향 전환 + MediaPipe 발목 근접도.
파일명은 과거 실험 이름이며, 현재 구현은 Depth Anything V2를 사용하지 않는다.
기존 캐시/시각화 경로의 호환성을 위해 유지한다. 현재 평가는 evaluate.py를 사용한다.
평가지표:
  1. 헤드업: 터치 시점 ±8프레임 head angle range
  2. 어깨 회전: 터치→터치 구간 어깨 방향 변화량
  3. 골반 회전: 터치→터치 구간 골반 방향 변화량
"""

import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from core.pose_extractor import PoseExtractor
from analysis.head_pose_analyzer import HeadPoseAnalyzer
from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
from visualization.skeleton_drawer import SkeletonDrawer
from analysis.ball_motion_analyzer import BallMotionData, TouchEvent
from utils.math_utils import angle_with_vertical
import config

VIDEOS = [
    "input/in_in/인,인 3-1.MOV",
    "input/in_in/인,인 3-2.MOV",
    "input/in_in/인,인 5-1.MOV",
    "input/in_in/인,인 6-1.MOV",
    "input/in_in/인,인 7-1.MOV",
    "input/in_in/인,인 7-2.MOV",
    "input/in_in/인,인 7-3.MOV",
    "input/in_in/인,인 8-1.MOV",
    "input/in_in/인,인 8-2.MOV",
    "input/in_in/인,인 9-1.MOV",
    "input/in_in/인,인 9-2.MOV",
    "input/in_in/인,인 기준1.MOV",
    "input/in_in/인,인 기준2.MOV",
]


def run_sam2(video_path, pose_frames):
    """SAM2 공 추적 → pose_frames에 ball_position 주입

    발목 좌표는 **전 프레임**을 넘긴다. 예전엔 첫 20프레임만 넘겼는데, 그러면
    추적 도중 공-발 거리 게이트(`sam2_ball_tracker.MAX_ANKLE_DIST`)가 작동할 수
    없어서 SAM2가 배경 공으로 갈아타도 잡아내지 못했다.
    """
    import tempfile
    try:
        ankle_positions = {}
        for pf in pose_frames:
            fw, fh = pf.frame_width, pf.frame_height
            lm = pf.landmarks
            ankle_positions[pf.frame_number] = [
                [float(lm[27][0] * fw), float(lm[27][1] * fh)],
                [float(lm[28][0] * fw), float(lm[28][1] * fh)],
            ]
        # 프레임 수가 많아 -c 인자에 그대로 싣기엔 크다. 파일로 넘긴다.
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fp:
            json.dump(ankle_positions, fp)
            ankle_path = fp.name

        sam2_script = f"""
import sys, json
sys.path.insert(0, '{os.path.abspath(".")}')
from sam2_ball_tracker import run_sam2_tracking
with open({ankle_path!r}) as f:
    ankle_raw = json.load(f)
ankle_positions = {{int(k): v for k, v in ankle_raw.items()}}
tracks = run_sam2_tracking({video_path!r}, ankle_positions=ankle_positions)
result = {{str(k): list(v) for k, v in tracks.items()}}
print(json.dumps(result))
"""
        result = subprocess.run(['python3.11', '-c', sam2_script],
                                capture_output=True, text=True, timeout=2400)
        os.unlink(ankle_path)
        if result.returncode == 0:
            json_line = [l for l in result.stdout.strip().split('\n') if l.startswith('{')][-1]
            tracks = {int(k): tuple(v) for k, v in json.loads(json_line).items()}
            frame_map = {pf.frame_number: pf for pf in pose_frames}
            for frame_idx, (cx, cy, r) in tracks.items():
                if frame_idx in frame_map:
                    pf = frame_map[frame_idx]
                    pf.ball_position = (cx, cy)
                    r = max(8, r)
                    pf.ball_bbox = (cx - r, cy - r, cx + r, cy + r)
            return tracks
    except Exception as e:
        print(f"    SAM2 스킵: {e}")
    return {}


def detect_touches_by_ball_direction(pose_frames, ball_tracks,
                                      max_foot_dist=200, min_frames_between=8):
    """공의 이동 방향 전환 + 발 근접 = 터치 감지

    한 방향 이동 중에는 터치가 1개만 존재한다고 가정.
    공 X 궤적이 극대/극소인 지점(방향 전환) 중
    발과 가까운 프레임만 터치로 인정.

    Args:
        pose_frames: PoseFrame 리스트
        ball_tracks: {frame_idx: (cx, cy, r)} SAM2 추적 결과
        max_foot_dist: 터치 인정 최대 공-발 2D 거리 (px)
        min_frames_between: 터치 간 최소 프레임 간격

    Returns:
        List[TouchEvent]
    """
    from scipy.signal import savgol_filter, find_peaks

    frame_map = {pf.frame_number: pf for pf in pose_frames}
    sorted_frames = sorted(ball_tracks.keys())

    if len(sorted_frames) < 10:
        return []

    # 공 X 좌표 시계열
    frames_arr = np.array(sorted_frames)
    ball_x = np.array([ball_tracks[f][0] for f in sorted_frames])

    # smoothing으로 노이즈 제거
    win = min(11, len(ball_x) if len(ball_x) % 2 == 1 else len(ball_x) - 1)
    if win >= 3:
        ball_x_smooth = savgol_filter(ball_x, window_length=win, polyorder=2)
    else:
        ball_x_smooth = ball_x

    # 극대(→에서 ←로) + 극소(←에서 →로) = 방향 전환 지점
    # 양 끝에 패딩 추가해서 시작/끝 극값도 감지
    padded = np.concatenate([[ball_x_smooth[0]], ball_x_smooth, [ball_x_smooth[-1]]])
    peak_idx, _ = find_peaks(padded, distance=min_frames_between, prominence=10)
    trough_idx, _ = find_peaks(-padded, distance=min_frames_between, prominence=10)
    # 패딩 인덱스 보정 (-1)
    peak_idx = peak_idx - 1
    trough_idx = trough_idx - 1
    # 범위 내로 클리핑
    peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < len(ball_x_smooth))]
    trough_idx = trough_idx[(trough_idx >= 0) & (trough_idx < len(ball_x_smooth))]

    # 합쳐서 시간순 정렬
    turn_indices = sorted(set(list(peak_idx) + list(trough_idx)))

    # TouchEvent 생성 (발 근접 조건 필터링)
    touches = []
    for ti in turn_indices:
        fn = int(frames_arr[ti])
        if fn not in frame_map:
            continue

        pf = frame_map[fn]
        cx, cy, r = ball_tracks[fn]
        lm = pf.landmarks
        wl = pf.world_landmarks
        fw, fh = pf.frame_width, pf.frame_height

        # 가장 가까운 발 판별 (2D 거리)
        l_ax = lm[27][0] * fw
        l_ay = lm[27][1] * fh
        r_ax = lm[28][0] * fw
        r_ay = lm[28][1] * fh
        l_dist = np.sqrt((cx - l_ax) ** 2 + (cy - l_ay) ** 2)
        r_dist = np.sqrt((cx - r_ax) ** 2 + (cy - r_ay) ** 2)

        min_dist = min(l_dist, r_dist)
        if min_dist > max_foot_dist:
            continue  # 발에서 멀면 터치 아님

        if l_dist < r_dist:
            foot, ankle_idx = 'left', 27
            ax, ay = l_ax, l_ay
        else:
            foot, ankle_idx = 'right', 28
            ax, ay = r_ax, r_ay

        touch = TouchEvent(
            frame_number=fn,
            timestamp=fn / 30.0,
            ball_2d=(cx, cy),
            ball_3d=(cx, cy, 0),
            touching_foot=foot,
            ankle_2d=(int(ax), int(ay)),
            ankle_z=float(wl[ankle_idx][2]),
            distance=float(min_dist),
            shoulder_direction=tuple(wl[12] - wl[11]),
            pelvis_direction=tuple(wl[24] - wl[23]),
        )
        touches.append(touch)

    return touches


def compute_rotation_between_touches(touches: List[TouchEvent], pose_frames, window=8):
    """터치 ±8프레임 구간의 2D 폭 median 비교로 회전량 계산

    T1 ±8프레임의 어깨/골반 폭 median vs T2 ±8프레임의 median.
    |차이| = 터치 구간의 회전량 (px).
    image landmarks만 사용 (world_landmarks X 노이즈 회피).
    """
    from scipy.signal import savgol_filter

    if len(touches) < 2:
        return [], []

    frame_map = {pf.frame_number: pf for pf in pose_frames}
    all_frames = sorted(frame_map.keys())
    n = len(all_frames)
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}

    sh_width = np.zeros(n)
    pe_width = np.zeros(n)
    for i, fn in enumerate(all_frames):
        pf = frame_map[fn]
        lm = pf.landmarks
        fw = pf.frame_width
        sh_width[i] = abs(lm[12][0] - lm[11][0]) * fw
        pe_width[i] = abs(lm[24][0] - lm[23][0]) * fw

    # smoothing
    win = min(11, n if n % 2 == 1 else n - 1)
    if win >= 3:
        sh_width = savgol_filter(sh_width, window_length=win, polyorder=2)
        pe_width = savgol_filter(pe_width, window_length=win, polyorder=2)

    shoulder_deltas = []
    pelvis_deltas = []

    for i in range(len(touches) - 1):
        f1 = touches[i].frame_number
        f2 = touches[i + 1].frame_number
        i1 = frame_to_idx.get(f1)
        i2 = frame_to_idx.get(f2)
        if i1 is None or i2 is None:
            continue

        r1_start = max(0, i1 - window)
        r1_end = min(n, i1 + window + 1)
        r2_start = max(0, i2 - window)
        r2_end = min(n, i2 + window + 1)

        sh_delta = abs(float(np.median(sh_width[r1_start:r1_end])) - float(np.median(sh_width[r2_start:r2_end])))
        shoulder_deltas.append(sh_delta)

        pe_delta = abs(float(np.median(pe_width[r1_start:r1_end])) - float(np.median(pe_width[r2_start:r2_end])))
        pelvis_deltas.append(pe_delta)

    return shoulder_deltas, pelvis_deltas


def make_touch_trace_video(video_path, pose_frames, ball_tracks, touches, output_path):
    """터치 시점 표시 trace 영상 생성"""
    drawer = SkeletonDrawer(color=(0, 255, 0))
    pose_dict = {pf.frame_number: pf for pf in pose_frames}
    touch_set = {t.frame_number: t for t in touches}

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    touch_list = sorted(touch_set.keys())
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in pose_dict:
            pf = pose_dict[frame_idx]
            frame = drawer.draw_skeleton(frame, pf.landmarks)
            frame = drawer.draw_body_direction_vectors(frame, pf.landmarks)

        # 공 표시
        if frame_idx in ball_tracks:
            cx, cy, r = ball_tracks[frame_idx]
            cv2.circle(frame, (int(cx), int(cy)), max(8, int(r)), (160, 160, 160), 2)

        # 터치 프레임 강조
        if frame_idx in touch_set:
            t = touch_set[frame_idx]
            ti = touch_list.index(frame_idx) + 1
            # 공 주위 빨간 원
            cv2.circle(frame, (int(t.ball_2d[0]), int(t.ball_2d[1])), 30, (0, 0, 255), 3)
            # TOUCH 텍스트
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"TOUCH #{ti}"
            (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
            tx = (width - tw) // 2
            cv2.rectangle(frame, (tx - 10, 30), (tx + tw + 10, 80), (255, 255, 255), -1)
            cv2.putText(frame, text, (tx, 70), font, 1.2, (0, 0, 255), 3)

        # 좌상단 정보
        font = cv2.FONT_HERSHEY_SIMPLEX
        info = f"Frame: {frame_idx}  Touches: {len(touches)}"
        cv2.putText(frame, info, (10, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  trace 영상 저장: {output_path}")


def process(video_path, save_trace=True):
    stem = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"처리 중: {stem}")

    # 1. 포즈 추출
    print("  1. 포즈 추출...")
    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        detect_ball=False,
    )
    pose_frames = extractor.extract_from_video(video_path)
    if len(pose_frames) == 0:
        return None

    # 2. SAM2 공 추적
    print("  2. SAM2 공 추적...")
    ball_tracks = run_sam2(video_path, pose_frames)
    if not ball_tracks:
        print("    공 추적 실패")
        return None
    print(f"    공 감지: {len(ball_tracks)} 프레임")

    # 3. 공 방향 전환 + 발 근접 터치 감지
    print("  3. 터치 감지...")
    touches = detect_touches_by_ball_direction(pose_frames, ball_tracks)
    print(f"    터치 감지: {len(touches)}개")
    for t in touches:
        print(f"      Frame {t.frame_number}: {t.touching_foot} foot, dist={t.distance:.1f}px")

    if len(touches) < 2:
        print("    터치 부족 (최소 2개 필요)")
        return None

    # 4. trace 영상 생성
    if save_trace:
        trace_path = f"output/videos/touch_trace_{stem}.mp4"
        make_touch_trace_video(video_path, pose_frames, ball_tracks, touches, trace_path)

    # 5. 터치 간 어깨/골반 변화율
    shoulder_deltas, pelvis_deltas = compute_rotation_between_touches(touches, pose_frames)
    avg_sh = float(np.mean(shoulder_deltas)) if shoulder_deltas else None
    avg_pe = float(np.mean(pelvis_deltas)) if pelvis_deltas else None
    rotation_avg = (avg_sh + avg_pe) / 2 if avg_sh and avg_pe else None

    # 6. 헤드업 (터치 ±8프레임)
    touch_frames = [t.frame_number for t in touches]
    ball_data = BallMotionData(
        frame_numbers=np.array([pf.frame_number for pf in pose_frames]),
        positions=np.array([[0, 0]] * len(pose_frames)),
        touch_frames=touch_frames,
        touch_count=len(touches),
        touch_events=touches,
    )
    head_data = HeadPoseAnalyzer(min_visibility_threshold=0.5).analyze(pose_frames, ball_data)
    headup = head_data.touch_window_mean_range if head_data else None

    return {
        'video': stem,
        'headup': headup,
        'shoulder': avg_sh,
        'pelvis': avg_pe,
        'rotation_avg': rotation_avg,
        'touch_count': len(touches),
        'sh_deltas': shoulder_deltas,
        'pe_deltas': pelvis_deltas,
    }


def main():
    header = f"{'영상':<20} {'헤드업':>8} {'어깨(px)':>8} {'골반(px)':>8} {'터치':>4}"
    print(header)
    print("-" * 50)

    results = []
    for v in VIDEOS:
        r = process(v, save_trace=False)
        if r:
            results.append(r)
            hu = f"{r['headup']:.1f}°" if r['headup'] else "N/A"
            sh = f"{r['shoulder']:.1f}" if r['shoulder'] else "N/A"
            pe = f"{r['pelvis']:.1f}" if r['pelvis'] else "N/A"
            print(f"  → {r['video']:<20} {hu:>8} {sh:>8} {pe:>8} {r['touch_count']:>4}")

    print("\n" + "=" * 50)
    print(f"\n{header}")
    print("-" * 50)
    for r in results:
        hu = f"{r['headup']:.1f}°" if r['headup'] else "N/A"
        sh = f"{r['shoulder']:.1f}" if r['shoulder'] else "N/A"
        pe = f"{r['pelvis']:.1f}" if r['pelvis'] else "N/A"
        print(f"{r['video']:<20} {hu:>8} {sh:>8} {pe:>8} {r['touch_count']:>4}")


if __name__ == "__main__":
    main()
