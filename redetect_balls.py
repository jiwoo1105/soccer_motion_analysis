#!/usr/bin/env python3
"""Reproduce the latest ankle-following YOLO ball candidates from pose caches.

Only explicit local .pt weights are accepted; obtain the weights before running.
All retained manifest entries (including references) are processed. Output is
raw post-NMS class-32 candidates, without tracking or geometric selection.
OpenCV, Torch and Ultralytics are imported only after CLI input validation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unicodedata

import numpy as np

from scoring.roi_ball import PARAMETERS, remap_detections, roi_for


def nfc(value):
    return unicodedata.normalize('NFC', value)


def _key(value):
    return nfc(value).casefold()


def source_path(path):
    """Use cwd-relative paths when possible, otherwise an absolute resolved path."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _index_files(directory, *, cache=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Not a directory: {directory}')
    index = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if cache:
            if not path.name.lower().endswith('_pose.npz'):
                continue
            stem = path.name[:-len('_pose.npz')]
        else:
            if path.suffix.lower() not in {'.mov', '.mp4'}:
                continue
            stem = path.stem
        key = _key(stem)
        if key in index:
            raise ValueError(f'Duplicate NFC/case-insensitive stem {nfc(stem)!r}: '
                             f'{index[key]} and {path}')
        index[key] = path
    return index


def discover_videos(video_dir, manifest):
    """Match non-excluded manifest names to immediate video directory files."""
    with Path(manifest).open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get('clips'), list):
        raise ValueError(f'Manifest must contain a clips list: {manifest}')
    names = set()
    retained = []
    for entry in data['clips']:
        if not isinstance(entry, dict) or not isinstance(entry.get('name'), str):
            raise ValueError('Each manifest clip must have a string name')
        name = nfc(entry['name'])
        if not name.strip() or name in {'.', '..'} or '/' in name or '\\' in name or '\0' in name:
            raise ValueError(f'Invalid manifest clip name: {name!r}')
        if _key(name) in names:
            raise ValueError(f'Duplicate NFC/case-insensitive manifest clip: {name}')
        names.add(_key(name))
        if entry.get('exclude_reason') is None:
            retained.append(name)
    if not retained:
        raise ValueError('Manifest has no non-excluded clips')
    videos = _index_files(video_dir)
    clips = []
    for name in retained:
        if _key(name) not in videos:
            raise FileNotFoundError(f'Missing video for {name!r} in {video_dir} (.mov/.mp4)')
        clips.append({'clip': name, 'video': videos[_key(name)]})
    return clips


def discover_clips(video_dir, cache_dir, manifest):
    """Match retained videos to pose caches, rejecting ambiguous normalized names."""
    clips = discover_videos(video_dir, manifest)
    caches = _index_files(cache_dir, cache=True)
    for clip in clips:
        name = clip['clip']
        if _key(name) not in caches:
            raise FileNotFoundError(f'Missing pose cache for {name!r} in {cache_dir} (*_pose.npz)')
        clip['pose'] = caches[_key(name)]
    return clips


def load_pose_cache(path):
    """Load the existing NPZ schema, validating alignment without imputing rows.

    Individual nonfinite pose rows remain present so roi_for can record them as
    invalid_roi. Duplicate or malformed frame indices and mismatched arrays fail.
    World landmarks and old ball tracks are intentionally not used.
    """
    required = {'frame_numbers', 'timestamps', 'landmarks', 'visibility',
                'frame_width', 'frame_height'}
    with np.load(path, allow_pickle=False) as cache:
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f'Pose cache {path} is missing keys: {sorted(missing)}')
        arrays = {key: cache[key].copy() for key in required}
    numbers = arrays['frame_numbers']
    if (numbers.ndim != 1 or numbers.dtype.kind not in 'iuf'
            or not np.isfinite(numbers).all() or (numbers < 0).any()
            or (numbers != np.floor(numbers)).any()):
        raise ValueError(f'Invalid frame_numbers in pose cache: {path}')
    numbers = [int(number) for number in numbers]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f'Duplicate frame_numbers in pose cache: {path}')
    count = len(numbers)
    expected = {'timestamps': (count,), 'landmarks': (count, 33, 3),
                'visibility': (count, 33), 'frame_width': (), 'frame_height': ()}
    for key, shape in expected.items():
        if arrays[key].shape != shape or arrays[key].dtype.kind not in 'iuf':
            raise ValueError(f'Invalid {key} in pose cache {path}: expected numeric shape {shape}, '
                             f'got {arrays[key].shape}')
    dimensions = {}
    for key in ['frame_width', 'frame_height']:
        value = float(arrays[key])
        if not np.isfinite(value) or value <= 0 or value != int(value):
            raise ValueError(f'Invalid {key} in pose cache: {path}')
        dimensions[key] = int(value)
    return {**dimensions, 'map': {
        fn: (lm, vis, float(ts)) for fn, lm, vis, ts in zip(
            numbers, arrays['landmarks'], arrays['visibility'], arrays['timestamps'])
    }}


def save_json(path, value, *, overwrite=False):
    """Publish a complete JSON file atomically; default publication is exclusive."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking a fully written sibling file both avoids partial results
            # and refuses an existing destination, including a racing writer.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _infer(model, device, records, crops, torch):
    if device == 'mps':
        torch.mps.synchronize()
    start = time.perf_counter()
    results = model.predict(
        crops, device=device, imgsz=PARAMETERS['imgsz'], conf=PARAMETERS['conf'],
        iou=PARAMETERS['iou'], classes=[PARAMETERS['class_id']], rect=PARAMETERS['rect'],
        half=PARAMETERS['half'], verbose=False, save=False, save_txt=False, save_crop=False,
        batch=len(crops),
    )
    if device == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    if len(results) != len(records):
        raise RuntimeError(f'Detector returned {len(results)} results for {len(records)} crops')
    for record, result in zip(records, results):
        record['detections'] = remap_detections(
            result.boxes.xyxy.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
            result.boxes.cls.detach().cpu().numpy(), record['roi_xyxy'],
        )
        record['yolo_speed_ms'] = result.speed
        record['status'] = 'observed'
    return elapsed


def process_clip(clip, pose, model, *, cv2, torch, device, batch_size):
    """Decode all source frames and return legacy candidate fields (no disk writes)."""
    start = time.perf_counter()
    capture = cv2.VideoCapture(str(clip['video']))
    rows, pending, crops = [], [], []
    inference_s = 0.0
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {clip["video"]}')
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (not np.isfinite(reported_count) or reported_count <= 0
                or reported_count != int(reported_count) or not np.isfinite(fps) or fps <= 0):
            raise RuntimeError(f'Invalid frame count/FPS metadata for {clip["video"]}: '
                               f'frames={reported_count}, fps={fps}')
        source_count = int(reported_count)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            fn = len(rows)
            if frame is None or frame.size == 0:
                raise RuntimeError(f'Decode error for {clip["clip"]}: empty frame {fn}')
            if frame.shape[:2] != (pose['frame_height'], pose['frame_width']):
                raise RuntimeError(f'Video/pose dimensions differ for {clip["clip"]} at frame {fn}: '
                                   f'{frame.shape[:2]} vs '
                                   f'{(pose["frame_height"], pose["frame_width"])}')
            lm, vis, ts = pose['map'].get(fn, (None, None, 0.0))
            record = roi_for(lm, vis, fn, frame.shape, ts)
            record['source_frame_over_avg_fps_s'] = fn / fps
            rows.append(record)
            if record['status'] == 'pending':
                x1, y1, x2, y2 = record['roi_xyxy']
                crops.append(frame[y1:y2, x1:x2].copy())
                pending.append(record)
            if len(crops) == batch_size:
                inference_s += _infer(model, device, pending, crops, torch)
                pending, crops = [], []
        if len(rows) != source_count:
            raise RuntimeError(f'Decode error for {clip["clip"]}: decoded {len(rows)} frames; '
                               f'metadata reports {source_count}. No clip result was written.')
        if crops:
            inference_s += _infer(model, device, pending, crops, torch)
    finally:
        capture.release()
    counts = {
        'source_frames': len(rows),
        'inferred_frames': sum(row['status'] == 'observed' for row in rows),
        'missing_pose_frames': sum(row['status'] == 'missing_pose' for row in rows),
        'invalid_roi_frames': sum(row['status'] == 'invalid_roi' for row in rows),
        'frames_with_detections': sum(bool(row['detections']) for row in rows),
        'total_candidates': sum(len(row['detections']) for row in rows),
        'inference_s': inference_s, 'wall_s': time.perf_counter() - start,
    }
    return {
        'clip': clip['clip'], 'drill': nfc(clip['video'].parent.name),
        'video': source_path(clip['video']), 'pose_cache': source_path(clip['pose']),
        'source_average_fps': fps, 'source_frame_count_metadata': source_count,
        'decode_status': 'complete', 'counts': counts, 'frames': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, default=Path('input/in_in'))
    parser.add_argument('--cache-dir', type=Path, default=Path('output/scoring_cache'))
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Required manifest file (clips with non-null exclude_reason are skipped)')
    parser.add_argument('--output-dir', type=Path, default=Path('output/current/roi_candidates'))
    parser.add_argument('--model', type=Path, default=Path('yolo11s.pt'),
                        help='Existing local .pt weights; never downloaded automatically')
    parser.add_argument('--device', default='mps', help='Ultralytics device, e.g. mps, cpu, or 0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing candidate results')
    args = parser.parse_args(argv)
    try:
        if args.batch_size <= 0:
            raise ValueError('--batch-size must be positive')
        if not args.model.is_file():
            raise FileNotFoundError(f'Model weights do not exist: {args.model}. '
                                    'Set up local weights first; automatic download is disabled.')
        if args.model.suffix.lower() != '.pt':
            raise ValueError('--model must be an existing local .pt weights file')
        clips = discover_clips(args.video_dir, args.cache_dir, args.manifest)
        destinations = [args.output_dir / f'{clip["clip"]}_candidates.json' for clip in clips]
        summary_path = args.output_dir / 'batch_summary.json'
        if not args.overwrite:
            for path in [*destinations, summary_path]:
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f'Result exists: {path}; use --overwrite to replace it')
        inputs = []
        for clip in clips:
            pose_hash = sha256(clip['pose'])
            pose = load_pose_cache(clip['pose'])
            if sha256(clip['pose']) != pose_hash:
                raise RuntimeError(f'Pose cache changed while loading: {clip["pose"]}')
            inputs.append((clip, pose, pose_hash))
        model_hash = sha256(args.model)
        # These must be set before Ultralytics imports its configuration/checks.
        os.environ['YOLO_OFFLINE'] = 'true'
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO

        if args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable; select --device cpu or an available GPU')
        model = YOLO(str(args.model.resolve()))
        if sha256(args.model) != model_hash:
            raise RuntimeError(f'Model weights changed while loading: {args.model}')
        metadata = {
            'parameters': {**PARAMETERS, 'model': source_path(args.model), 'batch_size': args.batch_size},
            'device': args.device, 'model_sha256': model_hash,
            'manifest': source_path(args.manifest), 'manifest_sha256': sha256(args.manifest),
            'versions': {'python': sys.version, 'numpy': np.__version__,
                         'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                         'opencv': cv2.__version__},
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        report = {**metadata, 'stage': 'manifest_clips', 'clips': []}
        for (clip, pose, pose_hash), destination in zip(inputs, destinations):
            result = process_clip(clip, pose, model, cv2=cv2, torch=torch,
                                  device=args.device, batch_size=args.batch_size)
            if sha256(clip['pose']) != pose_hash:
                raise RuntimeError(f'Pose cache changed during inference: {clip["pose"]}')
            result = {**metadata, **result, 'pose_sha256': pose_hash}
            save_json(destination, result, overwrite=args.overwrite)
            report['clips'].append({'clip': clip['clip'], 'drill': result['drill'],
                                    'file': destination.name, **result['counts']})
            print(json.dumps({'stage': 'clip_complete', 'completed': len(report['clips']),
                              'total': len(clips), 'clip': clip['clip'], **result['counts']},
                             ensure_ascii=False), flush=True)
        report.update({
            'completed_all_manifest_clips': True,
            'total_source_frames': sum(clip['source_frames'] for clip in report['clips']),
            'total_inferred_frames': sum(clip['inferred_frames'] for clip in report['clips']),
            'wall_s': time.perf_counter() - started,
        })
        save_json(summary_path, report, overwrite=args.overwrite)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, EOFError) as error:
        print(f'redetect_balls: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
