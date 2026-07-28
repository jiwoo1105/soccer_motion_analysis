# -*- coding: utf-8 -*-
"""비싼 추출 결과(포즈 + 공 추적)를 디스크에 캐시

SAM2 공 추적이 영상 1편당 약 10분 걸린다. 터치 감지 파라미터를 한 번 바꿀 때마다
10분씩 다시 돌릴 수는 없으므로, **추출과 지표 계산을 분리한다.**

    추출 (비싼 부분, 1회)  →  npz + json 캐시  →  지표 계산 (싼 부분, 반복 가능)

실행:
    python3.9 -m scoring.extraction_cache            # in_in 13편 추출
    python3.9 -m scoring.extraction_cache --force
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CACHE_DIR = Path('output/scoring_cache')

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


def paths(stem):
    return (CACHE_DIR / f'{stem}_pose.npz', CACHE_DIR / f'{stem}_ball.json')


def is_cached(stem):
    p, b = paths(stem)
    return p.exists() and b.exists()


def build(video_path, force=False):
    """영상 → 포즈 + 공 추적 캐시. 이미 있으면 건너뛴다."""
    from pathlib import Path as P
    stem = P(video_path).stem
    if is_cached(stem) and not force:
        print(f'  [캐시됨] {stem}')
        return True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'\n[{stem}] 추출 시작', flush=True)

    import config
    from core.pose_extractor import PoseExtractor

    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        detect_ball=False,
    )
    pose_frames = extractor.extract_from_video(video_path)
    if len(pose_frames) < 20:
        print(f'  프레임 부족: {len(pose_frames)}', flush=True)
        return False
    print(f'  포즈 {len(pose_frames)}프레임', flush=True)

    from extract_depth_metrics import run_sam2
    ball_tracks = run_sam2(video_path, pose_frames)
    print(f'  공 추적 {len(ball_tracks)}프레임', flush=True)

    pose_path, ball_path = paths(stem)
    np.savez_compressed(
        pose_path,
        frame_numbers=np.array([pf.frame_number for pf in pose_frames]),
        timestamps=np.array([pf.timestamp for pf in pose_frames]),
        landmarks=np.array([pf.landmarks for pf in pose_frames]),
        world_landmarks=np.array([pf.world_landmarks for pf in pose_frames]),
        visibility=np.array([pf.visibility for pf in pose_frames]),
        frame_width=np.array(pose_frames[0].frame_width),
        frame_height=np.array(pose_frames[0].frame_height),
    )
    with open(ball_path, 'w') as f:
        json.dump({str(k): list(v) for k, v in ball_tracks.items()}, f)

    return True


def load(stem):
    """캐시 → (pose_frames, ball_tracks)

    PoseFrame을 되살려서 기존 analyzer들을 그대로 쓸 수 있게 한다.
    """
    from core.pose_extractor import PoseFrame

    pose_path, ball_path = paths(stem)
    d = np.load(pose_path)
    fw, fh = int(d['frame_width']), int(d['frame_height'])

    pose_frames = [
        PoseFrame(
            frame_number=int(fn),
            timestamp=float(ts),
            landmarks=lm,
            world_landmarks=wl,
            visibility=vis,
            frame_width=fw,
            frame_height=fh,
        )
        for fn, ts, lm, wl, vis in zip(
            d['frame_numbers'], d['timestamps'],
            d['landmarks'], d['world_landmarks'], d['visibility'])
    ]

    with open(ball_path) as f:
        ball_tracks = {int(k): tuple(v) for k, v in json.load(f).items()}

    return pose_frames, ball_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--videos', nargs='*')
    args = ap.parse_args()

    targets = args.videos if args.videos else VIDEOS
    for v in targets:
        if not os.path.exists(v):
            print(f'  파일 없음: {v}', flush=True)
            continue
        try:
            build(v, force=args.force)
        except Exception as e:
            print(f'  실패: {v} — {e}', flush=True)

    print('\n캐시 상태:', flush=True)
    for v in targets:
        stem = Path(v).stem
        print(f"  {'O' if is_cached(stem) else 'X'} {stem}", flush=True)


if __name__ == '__main__':
    main()
