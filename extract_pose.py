#!/usr/bin/env python3
"""Create pose-only NPZ caches with the existing MediaPipe PoseExtractor.

Uses config.MEDIAPIPE_CONFIG and detect_ball=False; no SAM2 tracking is run.
Requires a MediaPipe installation exposing mp.solutions.pose and its local
model assets. Library imports are deferred until after argument validation.

Coordinate extraction does not establish correct person identity or tracking.
Manual tracking review is required before using the caches for evaluation.
The adjacent *_pose_metadata.json records this outstanding review explicitly.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from redetect_balls import (
    _index_files, _key, discover_videos, save_json, sha256, source_path,
)


def pose_arrays(pose_frames):
    """Convert PoseFrame rows into the existing NPZ cache schema."""
    frames = list(pose_frames)
    if not frames:
        raise ValueError('No pose frames extracted; refusing to create an empty cache')
    arrays = {
        'frame_numbers': np.asarray([frame.frame_number for frame in frames]),
        'timestamps': np.asarray([frame.timestamp for frame in frames]),
        'landmarks': np.asarray([frame.landmarks for frame in frames]),
        'world_landmarks': np.asarray([frame.world_landmarks for frame in frames]),
        'visibility': np.asarray([frame.visibility for frame in frames]),
        'frame_width': np.asarray(frames[0].frame_width),
        'frame_height': np.asarray(frames[0].frame_height),
    }
    count = len(frames)
    shapes = {'frame_numbers': (count,), 'timestamps': (count,),
              'landmarks': (count, 33, 3), 'world_landmarks': (count, 33, 3),
              'visibility': (count, 33), 'frame_width': (), 'frame_height': ()}
    for key, value in arrays.items():
        if (value.shape != shapes[key] or value.dtype.kind not in 'iuf'
                or not np.isfinite(value).all()):
            raise ValueError(f'Invalid/nonfinite extracted {key}; expected numeric shape {shapes[key]}')
    numbers = arrays['frame_numbers']
    if ((numbers < 0).any() or (numbers != np.floor(numbers)).any()
            or (numbers > np.iinfo(np.int64).max).any()
            or any(int(right) <= int(left) for left, right in zip(numbers, numbers[1:]))):
        raise ValueError('Extracted frame numbers must be unique, increasing, nonnegative integers')
    arrays['frame_numbers'] = numbers.astype(np.int64)
    for key in ['frame_width', 'frame_height']:
        dimension = float(arrays[key])
        if dimension <= 0 or dimension != int(dimension):
            raise ValueError(f'Invalid extracted {key}: {dimension}')
        if any(getattr(frame, key) != dimension for frame in frames):
            raise ValueError(f'Inconsistent extracted {key} across pose frames')
        arrays[key] = np.asarray(int(dimension))
    return arrays


def save_pose_cache(path, arrays, *, overwrite=False):
    """Atomically publish an NPZ, refusing existing files unless explicitly allowed."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, default=Path('input/in_in'))
    parser.add_argument('--cache-dir', type=Path, default=Path('output/scoring_cache'))
    parser.add_argument('--manifest', type=Path, default=Path('experiments/2026-09-20/manifest.json'))
    parser.add_argument('--overwrite', action='store_true', help='Replace existing pose caches and metadata')
    args = parser.parse_args(argv)
    try:
        clips = discover_videos(args.video_dir, args.manifest)
        existing = _index_files(args.cache_dir, cache=True) if args.cache_dir.exists() else {}
        for clip in clips:
            clip['pose'] = existing.get(_key(clip['clip']), args.cache_dir / f'{clip["clip"]}_pose.npz')
            clip['metadata'] = clip['pose'].with_name(clip['pose'].stem + '_metadata.json')
            if not args.overwrite:
                for path in [clip['pose'], clip['metadata']]:
                    if path.exists() or path.is_symlink():
                        raise FileExistsError(f'Pose result exists: {path}; use --overwrite to replace it')

        import mediapipe as mp
        import cv2
        from config import MEDIAPIPE_CONFIG

        if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
            raise RuntimeError(f'MediaPipe {mp.__version__} does not expose mp.solutions.pose, '
                               'which the existing PoseExtractor requires. Use a compatible environment.')
        parameters = {key: MEDIAPIPE_CONFIG[key] for key in [
            'model_complexity', 'min_detection_confidence', 'min_tracking_confidence',
        ]}
        # The legacy Pose constructor otherwise downloads a missing lite/heavy
        # model into the installed package. Require explicit setup beforehand.
        variant = {0: 'lite', 1: 'full', 2: 'heavy'}[parameters['model_complexity']]
        pose_model = Path(mp.__file__).resolve().parent / 'modules' / 'pose_landmark' / f'pose_landmark_{variant}.tflite'
        if not pose_model.is_file():
            raise FileNotFoundError(f'MediaPipe model asset does not exist: {pose_model}. '
                                    'Set up the model asset first; automatic download is disabled.')
        from core.pose_extractor import PoseExtractor

        metadata = {
            'parameters': {**parameters, 'detect_ball': False},
            'versions': {'python': sys.version, 'numpy': np.__version__,
                         'mediapipe': mp.__version__, 'opencv': cv2.__version__},
            'pose_model_sha256': sha256(pose_model),
            'manifest': source_path(args.manifest), 'manifest_sha256': sha256(args.manifest),
            'manual_tracking_review_required': True, 'tracking_review_status': 'not_reviewed',
            'review_note': 'Coordinate extraction does not validate person identity or tracking. '
                           'Manual tracking review is required before evaluation.',
        }
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        for index, clip in enumerate(clips, start=1):
            started = time.perf_counter()
            # A fresh tracker per video preserves the existing extraction workflow.
            extractor = PoseExtractor(**parameters, detect_ball=False)
            try:
                frames = extractor.extract_from_video(str(clip['video']))
            finally:
                del extractor  # Existing PoseExtractor destructor closes its MediaPipe graph.
            arrays = pose_arrays(frames)
            save_pose_cache(clip['pose'], arrays, overwrite=args.overwrite)
            report = {
                **metadata, 'clip': clip['clip'], 'video': source_path(clip['video']),
                'pose_cache': source_path(clip['pose']), 'pose_sha256': sha256(clip['pose']),
                'pose_frames': len(frames), 'wall_s': time.perf_counter() - started,
            }
            save_json(clip['metadata'], report, overwrite=args.overwrite)
            print(json.dumps({'stage': 'pose_complete', 'clip': clip['clip'],
                              'completed': index, 'total': len(clips), 'pose_frames': len(frames),
                              'pose_cache': report['pose_cache'], 'mediapipe': mp.__version__,
                              'manual_tracking_review_required': True}, ensure_ascii=False), flush=True)
        print('Manual tracking review is required before evaluation, even when coordinates were extracted.',
              file=sys.stderr)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f'extract_pose: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
