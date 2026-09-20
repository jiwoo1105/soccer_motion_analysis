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


# 추적된 공이 발에서 이만큼 넘게 떨어지면 "실제 공이 아니다"로 본다.
#
# 배경에 놓인 축구공들 때문에 SAM2가 엉뚱한 공에 고착되는 사고가 반복됐다
# (6-1·8-2는 초기화부터, 9-2·8-1·기준1은 63~83프레임쯤 갈아탐).
# 진단 결과 정상 추적 5편의 공-발 거리 중앙값은 61~110px, 실패 5편은 143~547px으로
# 250px에서 깔끔하게 갈렸다. 그래서 이 값을 게이트로 쓴다.
#
# 오른쪽 영역을 잘라내는 방법도 검토했지만 기각했다 — 기준1은 실제 공이 화면
# 오른쪽 끝(x≈1914)까지 가므로 크롭하면 진짜 공을 잘라낸다.
MAX_ANKLE_DIST = 250.0

# 게이트를 연속 몇 프레임 넘겨야 "놓쳤다"로 볼지. 한두 프레임 튀는 건 무시한다.
LOST_RUN = 5

BALL_CLASS = 32


def _ankle_dist(frame_idx: int, cx: float, cy: float, ankle_positions: dict):
    """공-발 거리(px). 해당 프레임 발목 좌표가 없으면 None"""
    if not ankle_positions:
        return None
    ankles = ankle_positions.get(frame_idx)
    if not ankles:
        return None
    return min(np.sqrt((cx - ax) ** 2 + (cy - ay) ** 2) for ax, ay in ankles)


def _scan_for_ball(model, frames_dir: str, total_frames: int, ankle_positions: dict,
                   start: int, end: int, confidence: float,
                   max_ankle_dist: float) -> tuple:
    """[start, end) 구간에서 발 근처 공을 찾는다.

    발목 좌표가 있으면 `max_ankle_dist` 안쪽 후보만 받아들이고, 그중 가장 가까운
    것을 고른다. 발목 좌표가 없으면(fallback) 화면 중앙 하단 기준 최근접.

    Returns: (frame_idx, cx, cy, radius) 또는 None
    """
    best = None
    best_dist = float('inf')

    for frame_idx in range(max(0, start), min(end, total_frames)):
        img = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]

        for r in model(img, conf=confidence, verbose=False):
            for box in r.boxes:
                if int(box.cls[0]) != BALL_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cy < h * 0.5:            # 화면 위쪽 절반은 공이 있을 수 없다
                    continue
                radius = ((x2 - x1) + (y2 - y1)) / 4

                dist = _ankle_dist(frame_idx, cx, cy, ankle_positions)
                if dist is None:
                    dist = np.sqrt((cx - w / 2) ** 2 + (cy - h * 0.8) ** 2)
                elif dist > max_ankle_dist:
                    continue            # 배경 공 — 후보에서 제외

                if dist < best_dist:
                    best_dist = dist
                    best = (frame_idx, cx, cy, radius)

        # 발 바로 옆(게이트의 절반)에 있는 공을 찾았으면 더 볼 것 없다
        if best is not None and best_dist < max_ankle_dist / 2:
            break

    return best


def find_initial_ball(video_path: str, yolo_model_path: str = "yolov8n.pt",
                      confidence: float = 0.3,
                      ankle_positions: dict = None,
                      search_frames: int = 20,
                      max_ankle_dist: float = MAX_ANKLE_DIST) -> tuple:
    """YOLO로 초기 공 위치 탐지 (발에서 max_ankle_dist 이내인 공만)

    `search_frames` 안에서 못 찾으면 구간을 두 배씩 넓혀 가며 다시 찾는다.
    처음 20프레임만 보던 시절엔 실제 공이 YOLO에 안 잡히면 배경 공을 집었다.

    Returns: (frame_idx, cx, cy, radius)
    """
    from ultralytics import YOLO as YOLOModel
    model = YOLOModel(yolo_model_path)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 6
    cap.release()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 스캔 구간만 프레임으로 떨어뜨린다 (전체 추출은 track_with_sam2가 한다)
        cap = cv2.VideoCapture(video_path)
        limit = min(total, max(search_frames * 4, 120))
        for i in range(limit):
            ret, frame = cap.read()
            if not ret:
                limit = i
                break
            cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), frame)
        cap.release()

        window = search_frames
        found = None
        while window <= limit and found is None:
            found = _scan_for_ball(model, tmpdir, limit, ankle_positions,
                                   0, window, confidence, max_ankle_dist)
            if found is None:
                window *= 2
        if found is None:      # 게이트를 만족하는 공이 끝내 없으면 게이트 없이 재시도
            found = _scan_for_ball(model, tmpdir, limit, ankle_positions,
                                   0, limit, confidence, float('inf'))

    if found is None:
        raise ValueError("초기 공 위치를 찾을 수 없습니다.")

    f, cx, cy, r = found
    d = _ankle_dist(f, cx, cy, ankle_positions)
    print(f"  초기 공: 프레임={f}, 위치=({cx:.0f}, {cy:.0f}), 반지름={r:.0f}"
          + (f", 발까지 {d:.0f}px" if d is not None else ""))
    return f, cx, cy, r


def track_with_sam2(video_path: str, init_frame: int, init_cx: float, init_cy: float,
                    checkpoint: str = "sam2_checkpoints/sam2.1_hiera_small.pt",
                    ankle_positions: dict = None,
                    max_ankle_dist: float = MAX_ANKLE_DIST,
                    max_reinit: int = 4,
                    yolo_model_path: str = "yolov8n.pt") -> dict:
    """SAM 2로 공 추적. 배경 공으로 갈아타면 끊고 발 근처에서 다시 잡는다.

    한 번 전파해서 끝내지 않고, 공-발 거리가 `max_ankle_dist`를 `LOST_RUN` 프레임
    연속 넘기면 거기서 구간을 끊는다. 그 뒤 프레임에서 발 근처 공을 YOLO로 다시 찾아
    프롬프트를 새로 주고 이어서 전파한다. `ankle_positions`가 없으면 게이트를 끄고
    예전처럼 한 번만 전파한다.

    Returns: {frame_idx: (cx, cy, radius)}
    """
    from sam2.build_sam import build_sam2_video_predictor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  SAM 2 디바이스: {device}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  프레임 추출 중...")
        total_frames = extract_frames(video_path, tmpdir)

        print("  SAM 2 모델 로드 중...")
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device=device
        )

        yolo = None
        results = {}
        frame, cx, cy = init_frame, init_cx, init_cy

        print("  SAM 2 추적 시작...")
        with torch.inference_mode():
            inference_state = predictor.init_state(video_path=tmpdir)

            for attempt in range(max_reinit + 1):
                predictor.reset_state(inference_state)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame,
                    obj_id=1,
                    points=np.array([[cx, cy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),   # 1 = foreground
                )

                segment = {}
                bad_run = 0
                lost_at = None

                for frame_idx, obj_ids, masks in predictor.propagate_in_video(
                        inference_state, start_frame_idx=frame):
                    for obj_id, mask in zip(obj_ids, masks):
                        if obj_id != 1:
                            continue
                        mask_np = mask[0].cpu().numpy() > 0.0
                        if mask_np.sum() == 0:
                            continue

                        ys, xs = np.where(mask_np)
                        mx, my = float(xs.mean()), float(ys.mean())
                        radius = float(np.sqrt(mask_np.sum() / np.pi))

                        d = _ankle_dist(frame_idx, mx, my, ankle_positions)
                        if d is not None and d > max_ankle_dist:
                            bad_run += 1
                            if bad_run >= LOST_RUN:
                                lost_at = frame_idx - bad_run + 1
                        else:
                            bad_run = 0
                        segment[frame_idx] = (mx, my, radius)

                    if lost_at is not None:
                        break

                # 놓치기 시작한 지점부터는 배경 공이므로 버린다
                if lost_at is not None:
                    segment = {k: v for k, v in segment.items() if k < lost_at}
                results.update(segment)

                if lost_at is None:
                    break
                print(f"    {lost_at}프레임에서 공을 놓침 (발에서 {max_ankle_dist:.0f}px 초과)"
                      f" — 재탐색")

                if yolo is None:
                    from ultralytics import YOLO as YOLOModel
                    yolo = YOLOModel(yolo_model_path)
                found = _scan_for_ball(yolo, tmpdir, total_frames, ankle_positions,
                                       lost_at + 1, total_frames, 0.3, max_ankle_dist)
                if found is None:
                    print("    이후 구간에서 발 근처 공을 못 찾음 — 추적 종료")
                    break
                frame, cx, cy, _ = found
                print(f"    {frame}프레임에서 재시작 ({cx:.0f}, {cy:.0f})")

    covered = len(results) / total_frames * 100 if total_frames else 0
    print(f"  추적된 프레임 수: {len(results)} / {total_frames} ({covered:.0f}%)")
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
    tracks = track_with_sam2(video_path, init_frame, init_cx, init_cy, checkpoint,
                             ankle_positions=ankle_positions,
                             yolo_model_path=yolo_model)

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
