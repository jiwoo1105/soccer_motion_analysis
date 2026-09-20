# -*- coding: utf-8 -*-
"""Legacy SAM2 공 재추적 도구 (현재 ROI 평가 경로와 별개).

`scoring.extraction_cache.build(force=True)`는 MediaPipe 포즈까지 다시 뽑아서
느리고, 포즈가 바뀌면 지금까지의 회전 분석과 대조가 안 된다. 그래서 SAM2만
다시 돌리고 `*_ball.json`만 교체한다. 포즈와 캐시 메타데이터는 유지한다.
매 실행마다 고유한 백업을 만들고, 같은 디렉터리의 임시 파일을 원자적으로 교체한다.
인자가 없으면 도움말만 출력한다. 실제 재추적은 영상 이름 또는 --all이 필요하다.

실행:
    python3.9 retrack_balls.py 6-1 8-2 9-2 8-1 기준1
    python3.9 retrack_balls.py --all
"""
import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

CACHE = Path('output/scoring_cache')
BACKUP = CACHE / 'backup_before_retrack'
VIDEO_DIR = Path('input/in_in')

ALL = ['3-1', '3-2', '5-1', '6-1', '7-1', '7-2', '8-1', '8-2',
       '9-1', '9-2', '기준1', '기준2']


def source_frame_count(video_path):
    """포즈가 없는 프레임까지 포함한 원본 영상 프레임 수."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    if not np.isfinite(count) or count <= 0 or count != int(count):
        raise ValueError(f'원본 영상 프레임 수를 읽을 수 없습니다: {video_path}')
    return int(count)


def ball_foot_stats(stem, source_frames):
    """원본 프레임 기준 추적률과, 포즈가 있는 프레임의 공-발 거리."""
    if not isinstance(source_frames, int) or isinstance(source_frames, bool) or source_frames <= 0:
        raise ValueError('source_frames must be a positive integer')
    with np.load(CACHE / f'{stem}_pose.npz', allow_pickle=False) as d:
        lm = d['landmarks']
        frame_rows = {int(frame): i for i, frame in enumerate(d['frame_numbers'])}
        fw, fh = float(d['frame_width']), float(d['frame_height'])
    with (CACHE / f'{stem}_ball.json').open(encoding='utf-8') as f:
        tracks = json.load(f)
    dists = []
    tracked_frames = 0
    for k, (cx, cy, _r) in tracks.items():
        frame = int(k)
        if not 0 <= frame < source_frames or not np.isfinite([cx, cy, _r]).all():
            continue
        tracked_frames += 1
        i = frame_rows.get(frame)
        if i is None:
            continue
        distances = np.array([
            np.hypot(cx - lm[i, 27, 0] * fw, cy - lm[i, 27, 1] * fh),
            np.hypot(cx - lm[i, 28, 0] * fw, cy - lm[i, 28, 1] * fh)])
        finite = distances[np.isfinite(distances)]
        if len(finite):
            dists.append(float(finite.min()))
    return tracked_frames / source_frames * 100, (float(np.median(dists)) if dists else float('nan'))


def replace_ball_cache(path, tracks):
    """고유 백업 후 atomic replace; 실패하면 기존 캐시는 그대로 남는다."""
    # Serialize before creating a backup or touching the existing cache.
    payload = json.dumps({str(k): list(v) for k, v in tracks.items()}, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        BACKUP.mkdir(parents=True, exist_ok=True)
        fd, backup_name = tempfile.mkstemp(dir=BACKUP, prefix=f'{path.stem}.', suffix='.json')
        os.close(fd)
        backup = Path(backup_name)
        try:
            shutil.copy2(path, backup)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
        # Keep mode/timestamps and supported filesystem metadata as well as the
        # unchanged flat JSON schema. The pose NPZ is never rewritten.
        shutil.copystat(path, temporary)
        os.replace(temporary, path)
        return backup
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def retrack(short):
    stem = f'인,인 {short}'
    video = VIDEO_DIR / f'{stem}.MOV'
    if not video.exists():
        print(f'[{short}] 영상 없음: {video}')
        return False

    from scoring.extraction_cache import load
    from extract_depth_metrics import run_sam2

    source_frames = source_frame_count(video)
    before = ball_foot_stats(stem, source_frames)
    pose_frames, _ = load(stem)

    print(f'\n[{short}] 재추적 시작 — 포즈 {len(pose_frames)}프레임', flush=True)
    tracks = run_sam2(str(video), pose_frames)
    if not tracks:
        print(f'[{short}] 추적 실패 — 기존 파일 유지')
        return False

    backup = replace_ball_cache(CACHE / f'{stem}_ball.json', tracks)

    after = ball_foot_stats(stem, source_frames)
    print(f'[{short}] 원본 {source_frames}프레임 기준 추적률 {before[0]:.0f}% → {after[0]:.0f}%   '
          f'공-발거리 {before[1]:.0f}px → {after[1]:.0f}px', flush=True)
    print(f'[{short}] 백업: {backup}', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('clips', nargs='*', metavar='CLIP', help='인,인 영상 이름의 접미사 (예: 3-1 기준1)')
    parser.add_argument('--all', action='store_true', help='목록의 모든 영상을 명시적으로 재추적')
    args = parser.parse_args()
    if args.all and args.clips:
        parser.error('영상 이름과 --all 중 하나만 지정하세요')
    if not args.all and not args.clips:
        parser.print_help()
        return 0
    targets = ALL if args.all else list(dict.fromkeys(args.clips))
    print(f'대상: {targets}')
    ok = 0
    for t in targets:
        try:
            ok += bool(retrack(t))
        except Exception as e:
            print(f'[{t}] 오류: {e}', flush=True)
    print(f'\n완료 {ok}/{len(targets)}   백업: {BACKUP}')
    return 0 if ok == len(targets) else 1


if __name__ == '__main__':
    sys.exit(main())
