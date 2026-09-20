# Legacy visualization entry point; use evaluate.py for the current metrics and equal-weight total.
# main.py
"""
축구 드리블 분석 시스템
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pose_extractor import PoseExtractor
from analysis.ball_motion_analyzer import BallMotionAnalyzer
from analysis.head_pose_analyzer import HeadPoseAnalyzer
from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
from visualization.skeleton_drawer import SkeletonDrawer
from visualization.ball_motion_plotter import BallMotionPlotter
from visualization.head_pose_plotter import HeadPosePlotter
from visualization.trunk_pose_plotter import TrunkPosePlotter
from analysis.dribble_cycle_analyzer import analyze_dribble_cycles
from visualization.dribble_cycle_plotter import plot_dribble_cycles
from visualization.pose_3d_plotter import plot_3d_pose
import cv2
import numpy as np
import config


def main():
    print("\n" + "="*70)
    print("축구 드리블 분석 시스템")
    print("="*70)

    # 비디오 경로 설정
    video_path = "input/in_in/인,인 3-1.MOV"

    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"\n 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        print(f"   input/ 폴더에 soccer1.mp4 파일을 넣어주세요.")
        return

    print(f"\n📹 비디오 파일: {video_path}")

    # 1. 비디오 정보 출력
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"   - 총 프레임: {total_frames}")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - 재생 시간: {duration:.1f}초")

    # 2. 포즈 추출
    print(f"\n{'='*70}")
    print("1단계: 포즈 추출 중...")
    print(f"{'='*70}")

    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        detect_ball=False,  # SAM2가 공 추적 전담 → YOLO 공 탐지 불필요
    )

    pose_frames = extractor.extract_from_video(video_path)

    if len(pose_frames) == 0:
        print("\n 오류: 영상에서 포즈를 감지할 수 없습니다.")
        return

    print(f"\n포즈 추출 완료: {len(pose_frames)}개 프레임")

    # 2-1. SAM 2 공 추적 (YOLO 결과를 SAM 2로 덮어씀)
    print(f"\n{'='*70}")
    print("1.5단계: SAM 2 공 추적 중...")
    print(f"{'='*70}\n")
    try:
        import subprocess, json

        # 처음 20프레임의 발목 픽셀 좌표 추출 (초기 공 탐지용)
        ankle_positions_for_sam2 = {}
        for pf in pose_frames:
            if pf.frame_number >= 20:
                break
            fw, fh = pf.frame_width, pf.frame_height
            lm = pf.landmarks
            ankle_positions_for_sam2[pf.frame_number] = [
                [float(lm[27][0] * fw), float(lm[27][1] * fh)],
                [float(lm[28][0] * fw), float(lm[28][1] * fh)],
            ]
        ankle_json = json.dumps(ankle_positions_for_sam2)

        sam2_script = """
import sys, json
sys.path.insert(0, '{project_dir}')
from sam2_ball_tracker import run_sam2_tracking
ankle_raw = json.loads('{ankle_json}')
ankle_positions = {{int(k): v for k, v in ankle_raw.items()}}
tracks = run_sam2_tracking('{video_path}', ankle_positions=ankle_positions)
result = {{str(k): list(v) for k, v in tracks.items()}}
print(json.dumps(result))
""".format(
            project_dir=os.path.abspath('.'),
            video_path=video_path,
            ankle_json=ankle_json.replace("'", "\\'"),
        )

        result = subprocess.run(
            ['python3.11', '-c', sam2_script],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            # JSON 마지막 줄만 파싱 (progress bar 출력 제외)
            json_line = [l for l in result.stdout.strip().split('\n') if l.startswith('{')][-1]
            tracks_raw = json.loads(json_line)
            tracks = {int(k): tuple(v) for k, v in tracks_raw.items()}

            # pose_frames에 주입
            frame_map = {pf.frame_number: pf for pf in pose_frames}
            injected = 0
            for frame_idx, (cx, cy, r) in tracks.items():
                if frame_idx in frame_map:
                    pf = frame_map[frame_idx]
                    pf.ball_position = (cx, cy)
                    r = max(8, r)
                    pf.ball_bbox = (cx - r, cy - r, cx + r, cy + r)
                    injected += 1
            print(f"SAM 2 결과 주입 완료: {injected}프레임")
        else:
            print(f"SAM 2 실패, YOLO 결과 유지:\n{result.stderr[-300:]}")
    except Exception as e:
        print(f"SAM 2 스킵 ({e}), YOLO 결과 유지")

    # 3. 공 움직임 분석
    print(f"\n{'='*70}")
    print("2단계: 공 움직임 분석 중...")
    print(f"{'='*70}\n")

    ball_analyzer = BallMotionAnalyzer(
        min_distance_between_touches=config.BALL_MOTION_CONFIG['min_distance_between_touches'],
        peak_prominence=config.BALL_MOTION_CONFIG['peak_prominence'],
        max_touch_distance=config.BALL_MOTION_CONFIG['max_touch_distance'],
    )
    ball_motion_data = ball_analyzer.analyze(pose_frames)

    if ball_motion_data:
        print(ball_motion_data)
    else:
        print("Warning: 공 움직임 분석 실패 (공이 충분히 탐지되지 않음)")

    # 3. 머리 자세 분석
    print(f"\n{'='*70}")
    print("3단계: 머리 자세 분석 중...")
    print(f"{'='*70}\n")

    head_analyzer = HeadPoseAnalyzer(min_visibility_threshold=0.5)
    head_pose_data = head_analyzer.analyze(pose_frames, ball_motion_data)

    if head_pose_data:
        print(head_pose_data)
    else:
        print("Warning: 머리 자세 분석 실패 (랜드마크 신뢰도 부족)")

    # 상체 자세 분석
    print(f"\n{'='*70}")
    print("3-1단계: 상체 자세 분석 중...")
    print(f"{'='*70}\n")

    trunk_analyzer = TrunkPoseAnalyzer(min_visibility_threshold=0.5)
    trunk_pose_data = trunk_analyzer.analyze(pose_frames)

    if trunk_pose_data:
        print(trunk_pose_data)
    else:
        print("Warning: 상체 자세 분석 실패 (랜드마크 신뢰도 부족)")

    stem = Path(video_path).stem
    graphs_dir = Path('output/graphs')
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # 4. 공 움직임 그래프 생성
    if ball_motion_data:
        print(f"\n{'='*70}")
        print("4단계: 공 움직임 그래프 생성 중...")
        print(f"{'='*70}\n")

        plotter = BallMotionPlotter()
        plotter.plot_motion(ball_motion_data,
                            save_path=str(graphs_dir / f'ball_motion {stem}.png'))

        if ball_motion_data.touch_directions:
            plotter.plot_3d_direction(ball_motion_data,
                                      save_path=str(graphs_dir / f'ball_3d_direction {stem}.png'))

        plotter.plot_trajectory_2d(ball_motion_data,
                                   save_path=str(graphs_dir / f'ball_trajectory {stem}.png'))

        if ball_motion_data.touch_directions:
            plotter.plot_coordination(ball_motion_data,
                                      save_path=str(graphs_dir / f'coordination {stem}.png'))

    # 4-0. 드리블 사이클 방향 분석
    if ball_motion_data and ball_motion_data.all_ball_positions:
        print(f"\n{'='*70}")
        print("4-0단계: 드리블 사이클 방향 분석 중...")
        print(f"{'='*70}\n")

        _fps = fps

        cycle_data = analyze_dribble_cycles(ball_motion_data.all_ball_positions, fps=_fps)
        if cycle_data:
            print(f"  peak {len(cycle_data.peak_frames)}개 / trough {len(cycle_data.trough_frames)}개")
            print(f"  NEAR↑(카메라쪽): {cycle_data.near_count}회 / FAR↓(반대쪽): {cycle_data.far_count}회")
            for i, hc in enumerate(cycle_data.half_cycles):
                print(f"    [{i+1}] Frame {hc.start_frame}~{hc.end_frame}  {hc.direction}  Δr={hc.r_change:+.1f}px")
            plot_dribble_cycles(cycle_data, fps=_fps,
                                save_path=str(graphs_dir / f'dribble_cycles_{stem}.png'))
        else:
            print("  사이클 분석 실패 (데이터 부족)")

    # 4-1. 머리 각도 그래프 생성
    if head_pose_data:
        print(f"\n{'='*70}")
        print("4-1단계: 머리 각도 그래프 생성 중...")
        print(f"{'='*70}\n")

        head_plotter = HeadPosePlotter()
        head_plotter.plot_head_angle(head_pose_data,
                                     save_path=str(graphs_dir / f'head_angle {stem}.png'))

        if head_pose_data.touch_window_data:
            head_plotter.plot_touch_window_rate(
                head_pose_data,
                save_path=str(graphs_dir / f'head_touch_window_rate {stem}.png')
            )

    # 4-1-1. 3D 포즈 시각화 (62번 프레임)
    if pose_frames:
        target_frame_idx = 62
        frame_map = {pf.frame_number: pf for pf in pose_frames}
        target_frame = frame_map.get(target_frame_idx, pose_frames[len(pose_frames) // 2])
        plot_3d_pose(
            world_landmarks=target_frame.world_landmarks,
            frame_index=target_frame.frame_number,
            video_stem=stem,
            save_path=str(graphs_dir / f'3d_pose_{stem}.png')
        )

    # 4-2. 상체 각도 그래프 생성
    if trunk_pose_data:
        print(f"\n{'='*70}")
        print("4-2단계: 상체 각도 그래프 생성 중...")
        print(f"{'='*70}\n")

        trunk_plotter = TrunkPosePlotter()
        trunk_plotter.plot_trunk_angle(trunk_pose_data,
                                       save_path=str(graphs_dir / f'trunk_angle {stem}.png'))

    # 5. 스켈레톤 비디오 생성 (공 위치 및 터치 표시 포함)
    print(f"\n{'='*70}")
    print("5단계: 스켈레톤 비디오 생성 중...")
    print(f"{'='*70}\n")

    create_skeleton_video(video_path, pose_frames, ball_motion_data,
                          head_data=head_pose_data,
                          trunk_data=trunk_pose_data)

    print()


def create_skeleton_video(video_path: str, pose_frames, ball_motion_data=None,
                          touch_frames_override=None, output_filename=None,
                          head_data=None, trunk_data=None,
                          touch_display_window=15):
    """
    스켈레톤 비디오 생성 (공 위치 및 터치 표시 포함)

    Args:
        video_path: 원본 비디오 경로
        pose_frames: 포즈 프레임 리스트
        ball_motion_data: BallMotionData 객체
        touch_frames_override: 터치 프레임 목록 직접 지정 (멀티뷰 확정 터치용)
        output_filename: 출력 파일명 (None이면 입력 파일명 기반 자동 생성)
        head_data: HeadPoseData (영상에 헤드각도 오버레이용)
        trunk_data: TrunkPoseData (영상에 상체각도 오버레이용)
        touch_display_window: 터치 전후 몇 프레임 동안 TOUCH 표시 유지
    """
    videos_dir = Path('output/videos')
    videos_dir.mkdir(parents=True, exist_ok=True)

    drawer = SkeletonDrawer(color=(0, 255, 0))

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_filename is None:
        stem = Path(video_path).stem
        output_filename = f'skeleton_output {stem}.mp4'
    output_path = str(videos_dir / output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"스켈레톤 그리는 중... → {output_path}")

    # ── 사전 준비 ─────────────────────────────────────────────────────

    pose_dict = {pf.frame_number: pf for pf in pose_frames}

    # 최종 터치 프레임 목록 (override 우선)
    if touch_frames_override is not None:
        final_touch_frames = sorted(touch_frames_override)
    elif ball_motion_data is not None:
        final_touch_frames = sorted(ball_motion_data.touch_frames)
    else:
        final_touch_frames = []

    # 전체 프레임 공 위치 dict (SAM2 감지 + 선형 보간 통합) → (x, y, r)
    all_ball_positions = {}
    interp_frames = set()   # 선형 보간된 프레임 번호 (SAM2 미감지)
    if ball_motion_data is not None:
        if ball_motion_data.all_ball_positions:
            all_ball_positions = ball_motion_data.all_ball_positions
        if ball_motion_data.interp_positions:
            interp_frames = set(ball_motion_data.interp_positions.keys())

    # 터치 표시: 정확한 터치 프레임에만 "TOUCH #N" 표시
    touch_window = {}
    for i, tf in enumerate(final_touch_frames):
        label = f"TOUCH #{i + 1}"
        touch_window[tf] = (i + 1, label)

    # 정확한 터치 프레임 → 터치한 발('left'/'right') 매핑
    exact_touch_foot = {}
    if ball_motion_data is not None:
        for touch in ball_motion_data.touch_events:
            exact_touch_foot[touch.frame_number] = touch.touching_foot
    elif touch_frames_override is not None:
        for tf in touch_frames_override:
            exact_touch_foot[tf] = None  # 발 정보 없으면 원 미표시

    exact_touch_set = set(final_touch_frames)

    # MediaPipe 랜드마크 인덱스 (발끝, 뒤꿈치)
    _HEEL  = {  'left': 29, 'right': 30}
    _TOE   = {  'left': 31, 'right': 32}

    # 프레임 → 각도 딕셔너리 (헤드, 상체)
    head_angle_map = {}
    head_mean_range = None
    if head_data is not None:
        for fn, ang in zip(head_data.frame_numbers, head_data.head_angles):
            head_angle_map[int(fn)] = float(ang)
        head_mean_range = head_data.touch_window_mean_range

    trunk_angle_map = {}
    trunk_mean_angle = None
    if trunk_data is not None:
        for fn, ang in zip(trunk_data.frame_numbers, trunk_data.trunk_angles):
            trunk_angle_map[int(fn)] = float(ang)
        trunk_mean_angle = float(trunk_data.mean_angle)

    # 상체 회전 점수 (어깨·골반 XZ 평면 회전각 평균)
    rotation_score = None
    if ball_motion_data is not None and ball_motion_data.touch_directions:
        dirs = ball_motion_data.touch_directions
        avg_sh = sum(d.shoulder_rotation_angle for d in dirs) / len(dirs)
        avg_pe = sum(d.pelvis_rotation_angle   for d in dirs) / len(dirs)
        rotation_score = (avg_sh + avg_pe) / 2

    # ── 프레임 루프 ──────────────────────────────────────────────────

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ball_position = None
        ball_bbox     = None

        if frame_idx in pose_dict:
            pf        = pose_dict[frame_idx]
            ball_position = pf.ball_position
            ball_bbox     = pf.ball_bbox

            # 1. 스켈레톤
            frame = drawer.draw_skeleton(frame, pf.landmarks)

            # 1-1. 헤드업 측정 기준 벡터 (어깨중앙 → 눈중앙, 빨간 막대)
            frame = drawer.draw_head_vector(frame, pf.landmarks)

            # 1-2. 어깨·골반 방향벡터 + 향하는 방향 화살표 (협응성 분석 기준)
            frame = drawer.draw_body_direction_vectors(frame, pf.landmarks, pf.world_landmarks)

            # 1-3. 상체 각도 연결선 (엉덩이→어깨, 엉덩이→무릎, 흰색)
            frame = drawer.draw_trunk_angle(
                frame, pf.landmarks,
                trunk_angle=trunk_angle_map.get(frame_idx)
            )

            # 2. 공 위치 표시
            if frame_idx in all_ball_positions:
                bx, by, br = all_ball_positions[frame_idx]
                bradius = max(8, int(br))
                if frame_idx in interp_frames:
                    # RTS 보간 프레임 → 주황 원 (SAM2 미감지, RTS 예측 위치)
                    frame = drawer.draw_ball_interpolated(frame, (bx, by), radius=bradius)
                else:
                    # SAM2 감지 + RTS 스무딩 프레임 → 회색 원
                    cx, cy = int(bx), int(by)
                    cv2.circle(frame, (cx, cy), bradius, (160, 160, 160), 2)
                    cv2.circle(frame, (cx, cy), 3,       (160, 160, 160), -1)

        # 3. 터치 윈도우 오버레이
        if frame_idx in touch_window:
            _, label = touch_window[frame_idx]
            # 정확한 터치 프레임에서만 터치한 발 중앙에 빨간 원 표시
            foot_pos = None
            if frame_idx in exact_touch_set:
                foot = exact_touch_foot.get(frame_idx)
                if foot is not None and frame_idx in pose_dict:
                    lm = pose_dict[frame_idx].landmarks
                    heel_idx = _HEEL[foot]
                    toe_idx  = _TOE[foot]
                    fx = (lm[heel_idx][0] + lm[toe_idx][0]) / 2 * width
                    fy = (lm[heel_idx][1] + lm[toe_idx][1]) / 2 * height
                    foot_pos = (fx, fy)
            frame = drawer.draw_touch_highlight(frame, foot_pos, text=label)

        # 4. 정보 패널 (좌상단)
        frame = _draw_info_panel(
            frame, frame_idx, total,
            head_angle_map.get(frame_idx),
            trunk_angle_map.get(frame_idx),
            len(final_touch_frames),
            head_mean_range=head_mean_range,
            trunk_mean=trunk_mean_angle,
            rotation_score=rotation_score,
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f" 스켈레톤 비디오 저장: {output_path}")


def _draw_info_panel(frame: np.ndarray, frame_idx: int, total_frames: int,
                     head_angle, trunk_angle, touch_count: int,
                     head_mean_range: float = None, trunk_mean: float = None,
                     rotation_score: float = None) -> np.ndarray:
    """
    좌상단에 현재 프레임 정보 패널 오버레이

    표시 항목:
      Frame: 현재/전체
      Head:  현재 헤드각도 (있을 때만)
      Trunk: 현재 상체각도 (있을 때만)
      Touches: 총 터치 횟수
    """
    output = frame.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 6
    line_h     = 22
    x0, y0     = 10, 10

    lines = [
        (f"Frame: {frame_idx} / {total_frames}", (220, 220, 220)),
        (f"Touches: {touch_count}",              (100, 255, 100)),
    ]
    if head_angle is not None:
        head_extra = f"  range:{head_mean_range:.1f}" if head_mean_range is not None else ""
        lines.append((f"Head:  {head_angle:+.1f}deg{head_extra}", (255, 200, 100)))
    if trunk_angle is not None:
        trunk_avg_str = f"  avg:{trunk_mean:.1f}" if trunk_mean is not None else ""
        lines.append((f"Trunk: {trunk_angle:.1f}deg{trunk_avg_str}", (100, 200, 255)))
    if rotation_score is not None:
        lines.append((f"Rotation: {rotation_score:.1f}deg", (180, 255, 180)))

    # 배경 박스 크기 계산
    max_w = max(cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t, _ in lines)
    box_h = len(lines) * line_h + pad * 2

    # 반투명 배경 (어두운 직사각형)
    overlay = output.copy()
    cv2.rectangle(overlay,
                  (x0 - pad, y0 - pad),
                  (x0 + max_w + pad, y0 + box_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    # 텍스트 그리기
    for i, (text, color) in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(output, text, (x0, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    return output




if __name__ == "__main__":
    main()
