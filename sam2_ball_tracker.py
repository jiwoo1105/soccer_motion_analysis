"""
SAM 2 기반 공 추적 스크립트

동작:
1. YOLO로 첫 프레임에서 발과 가장 가까운 공 위치 자동 탐지
2. 해당 위치를 SAM 2 프롬프트로 사용
3. 전체 영상에 걸쳐 SAM 2가 공을 추적
4. 결과를 칼만 필터 없이 raw 위치로 저장 → 영상 출력
"""

import cv2
import numpy as np
import torch
import os
import tempfile
from pathlib import Path
from ultralytics import YOLO


def extract_frames(video_path: str, output_dir: str) -> int:
    """영상을 프레임 이미지로 추출 (SAM 2 video predictor용)"""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(output_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    print(f"  총 {frame_idx}프레임 추출 완료")
    return frame_idx


def find_initial_ball(video_path: str, yolo_model_path: str = "yolov8n.pt",
                      confidence: float = 0.3,
                      ankle_positions: dict = None) -> tuple:
    """
    YOLO로 초기 공 위치 탐지

    처음 20프레임에서 발목에 가장 가까운 공들 중 반지름이 가장 큰 공 선택.
    score = radius / (ankle_dist + 1)  → 높을수록 좋음
    ankle_positions 없으면 화면 중앙 하단 기준 사용.

    Returns: (frame_idx, cx, cy, radius)
    """
    from ultralytics import YOLO as YOLOModel
    model = YOLOModel(yolo_model_path)
    cap = cv2.VideoCapture(video_path)

    BALL_CLASS = 32

    best_frame = 0
    best_cx, best_cy, best_r = None, None, None
    best_score = float('inf')

    frame_idx = 0
    while frame_idx < 20:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        results = model(frame, conf=confidence, verbose=False)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != BALL_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                radius = ((x2 - x1) + (y2 - y1)) / 4

                if cy < h * 0.5:
                    continue

                if ankle_positions and frame_idx in ankle_positions:
                    ankles = ankle_positions[frame_idx]
                    dist = min(
                        np.sqrt((cx - ax)**2 + (cy - ay)**2)
                        for ax, ay in ankles
                    )
                else:
                    dist = np.sqrt((cx - w / 2)**2 + (cy - h * 0.8)**2)

                if dist < best_score:
                    best_score = dist
                    best_frame = frame_idx
                    best_cx, best_cy, best_r = cx, cy, radius

        frame_idx += 1

    cap.release()

    if best_cx is None:
        raise ValueError("초기 공 위치를 찾을 수 없습니다.")

    print(f"  초기 공: 프레임={best_frame}, 위치=({best_cx:.0f}, {best_cy:.0f}), 반지름={best_r:.0f}")
    return best_frame, best_cx, best_cy, best_r


def track_with_sam2(video_path: str, init_frame: int, init_cx: float, init_cy: float,
                    checkpoint: str = "sam2_checkpoints/sam2.1_hiera_small.pt") -> dict:
    """
    SAM 2로 공 추적
    Returns: {frame_idx: (cx, cy, radius)}
    """
    from sam2.build_sam import build_sam2_video_predictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  SAM 2 디바이스: {device}")

    # 프레임 추출
    with tempfile.TemporaryDirectory() as tmpdir:
        print("  프레임 추출 중...")
        total_frames = extract_frames(video_path, tmpdir)

        print("  SAM 2 모델 로드 중...")
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device=device
        )

        print("  SAM 2 추적 시작...")
        with torch.inference_mode():
            inference_state = predictor.init_state(video_path=tmpdir)

            # 초기 포인트 프롬프트 (공 중앙)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_frame,
                obj_id=1,
                points=np.array([[init_cx, init_cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),  # 1 = foreground
            )

            # 전체 영상 전파
            results = {}
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                for obj_id, mask in zip(obj_ids, masks):
                    if obj_id != 1:
                        continue
                    mask_np = mask[0].cpu().numpy() > 0.0
                    if mask_np.sum() == 0:
                        continue

                    # 마스크에서 중심 + 반지름 계산
                    ys, xs = np.where(mask_np)
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    area = mask_np.sum()
                    radius = float(np.sqrt(area / np.pi))
                    results[frame_idx] = (cx, cy, radius)

    print(f"  추적된 프레임 수: {len(results)} / {total_frames}")
    return results


def render_video(video_path: str, ball_tracks: dict, output_path: str):
    """추적 결과를 영상에 오버레이"""
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in ball_tracks:
            cx, cy, r = ball_tracks[frame_idx]
            cx, cy, r = int(cx), int(cy), max(8, int(r))
            cv2.circle(frame, (cx, cy), r, (0, 165, 255), 2)   # 주황 원
            cv2.circle(frame, (cx, cy), 3, (0, 165, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  저장: {output_path}")


def run_sam2_tracking(video_path: str,
                      checkpoint: str = "sam2_checkpoints/sam2.1_hiera_small.pt",
                      yolo_model: str = "yolov8n.pt",
                      ankle_positions: dict = None) -> dict:
    """
    SAM 2 추적 실행 후 {frame_idx: (cx, cy, radius)} 반환
    main.py에서 호출해서 pose_frames에 주입하는 용도

    Args:
        ankle_positions: {frame_idx: [[ax, ay], [ax, ay]]} MediaPipe 2D 발목 좌표 (픽셀).
                         None이면 화면 중앙 고정값으로 fallback.
    """
    print("  SAM 2: 초기 공 위치 탐지 중...")
    init_frame, init_cx, init_cy, _ = find_initial_ball(
        video_path, yolo_model, ankle_positions=ankle_positions
    )

    print("  SAM 2: 공 추적 중...")
    tracks = track_with_sam2(video_path, init_frame, init_cx, init_cy, checkpoint)

    return tracks


def inject_sam2_to_pose_frames(pose_frames: list, tracks: dict) -> list:
    """
    SAM 2 추적 결과를 pose_frames의 ball_position/ball_bbox에 주입
    """
    frame_map = {pf.frame_number: pf for pf in pose_frames}

    injected = 0
    for frame_idx, (cx, cy, r) in tracks.items():
        if frame_idx in frame_map:
            pf = frame_map[frame_idx]
            pf.ball_position = (cx, cy)
            r = max(8, r)
            pf.ball_bbox = (cx - r, cy - r, cx + r, cy + r)
            injected += 1

    print(f"  SAM 2 결과 주입: {injected}프레임")
    return pose_frames


if __name__ == "__main__":
    VIDEO_PATH = "input/in_in/인,인 5-3 rts.MOV"
    OUTPUT_PATH = "output/sam2_ball_track.mp4"
    CHECKPOINT = "sam2_checkpoints/sam2.1_hiera_small.pt"

    os.makedirs("output", exist_ok=True)

    print("1. 초기 공 위치 탐지 중...")
    init_frame, init_cx, init_cy, init_r = find_initial_ball(VIDEO_PATH)

    print("\n2. SAM 2 추적 중...")
    tracks = track_with_sam2(VIDEO_PATH, init_frame, init_cx, init_cy, CHECKPOINT)

    print("\n3. 영상 렌더링 중...")
    render_video(VIDEO_PATH, tracks, OUTPUT_PATH)

    print("\n완료!")
