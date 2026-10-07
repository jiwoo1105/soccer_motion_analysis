# -*- coding: utf-8 -*-
"""Original-video inference and explicit, source-bound cache replay.

No inference library is imported until an original video is decoded/inferred.
The cache always keeps the original frame indices, including missing poses.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                   indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_cache(bundle, video, roi):
    """Require exactly the source and extraction ROI used to create this bundle."""
    bundle = Path(bundle)
    meta = json.loads((bundle/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('video_sha256') != sha256(video):
        raise ValueError('Cache source video SHA-256 differs; run fresh inference')
    if meta.get('extraction_roi') != (list(roi) if roi is not None else None):
        raise ValueError('Cache extraction ROI differs; run fresh inference')
    for key, filename in [('pose_sha256', 'pose.npz'), ('candidates_sha256', 'candidates.json')]:
        if meta.get(key) != sha256(bundle/filename):
            raise ValueError(f'Cache {filename} SHA-256 differs; run fresh inference')
    return meta


def extract_pose_video(video, tracker=None, roi=None):
    """Decode all frames. Optional ROI is x,y,width,height in original pixels."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    own_tracker = False
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open source video: {video}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or min(w,h,expected) <= 0:
            raise ValueError('Source video has invalid dimensions, frame count or FPS')
        if roi is None:
            x,y,rw,rh = 0,0,w,h
        else:
            if len(roi) != 4 or any(not isinstance(v,(int,np.integer)) for v in roi):
                raise ValueError('ROI must contain four integers: x y width height')
            x,y,rw,rh = map(int,roi)
            if min(x,y) < 0 or min(rw,rh) <= 0 or x+rw>w or y+rh>h:
                raise ValueError('ROI must be inside the source video')
        if tracker is None:
            import mediapipe as mp
            if not hasattr(mp,'solutions'):
                raise RuntimeError('Use the pinned mediapipe==0.10.21 environment')
            asset = Path(mp.__file__).parent/'modules/pose_landmark/pose_landmark_heavy.tflite'
            if not asset.is_file():
                raise FileNotFoundError('Pose model missing; run python scripts/setup_models.py')
            tracker = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                            smooth_landmarks=True, enable_segmentation=False,
                                            min_detection_confidence=.5, min_tracking_confidence=.5)
            own_tracker = True
        landmarks, worlds, visibility = [], [], []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape[:2] != (h,w):
                raise ValueError('Source video dimensions change during decoding')
            result = tracker.process(cv2.cvtColor(image[y:y+rh,x:x+rw], cv2.COLOR_BGR2RGB))
            if result.pose_landmarks and getattr(result,'pose_world_landmarks',None):
                lm = np.array([[v.x,v.y,v.z] for v in result.pose_landmarks.landmark],float)
                world = np.array([[v.x,v.y,v.z] for v in result.pose_world_landmarks.landmark],float)
                vis = np.array([v.visibility for v in result.pose_landmarks.landmark],float)
                if lm.shape != (33,3) or world.shape != (33,3) or vis.shape != (33,):
                    raise ValueError('Unexpected MediaPipe landmark schema')
                lm[:,0] = (lm[:,0]*rw+x)/w
                lm[:,1] = (lm[:,1]*rh+y)/h
                lm[:,2] *= rw/w
            else:
                lm = np.full((33,3),np.nan)
                world = np.full((33,3),np.nan)
                vis = np.zeros(33)
            landmarks.append(lm); worlds.append(world); visibility.append(vis)
        n = len(landmarks)
        if n != expected:
            raise ValueError(f'Incomplete source decode: read {n} of {expected} frames')
        pose = dict(n=n,width=w,height=h,lm=np.array(landmarks),world=np.array(worlds),
                    vis=np.array(visibility),frame_numbers=np.arange(n))
        meta = dict(name=Path(video).name,frames=n,width=w,height=h,fps=fps,
                    clock='source frame / average FPS',extraction_roi=list(roi) if roi is not None else None)
        return pose, meta
    finally:
        cap.release()
        if own_tracker:
            tracker.close()


def save_pose(pose, meta, path):
    np.savez_compressed(path, frame_numbers=np.arange(pose['n'],dtype=np.int64),
                        timestamps=np.arange(pose['n'])/meta['fps'], landmarks=pose['lm'],
                        world_landmarks=pose['world'],visibility=pose['vis'],
                        frame_width=pose['width'],frame_height=pose['height'])


def infer_clip(video, bundle, model, device='cpu', roi=None):
    """Fresh inference to a new bundle. Model is one shared YOLO instance."""
    import cv2
    import torch
    import mediapipe as mp
    import ultralytics
    from redetect_balls import process_clip
    from scoring.roi_ball import PARAMETERS
    bundle=Path(bundle);bundle.mkdir(parents=True,exist_ok=False)
    source_hash=sha256(video)
    pose, meta=extract_pose_video(video,roi=roi)
    save_pose(pose,meta,bundle/'pose.npz')
    # Missing rows remain in the pose file and become missing_pose for YOLO ROI.
    valid=np.isfinite(pose['lm']).all(axis=(1,2))
    lookup={i:(pose['lm'][i],pose['vis'][i],i/meta['fps']) for i in np.flatnonzero(valid)}
    detected=process_clip(dict(video=Path(video),clip=Path(video).stem,pose=bundle/'pose.npz'),
                          dict(frame_width=meta['width'],frame_height=meta['height'],map=lookup),
                          model,cv2=cv2,torch=torch,device=device,batch_size=16)
    if sha256(video)!=source_hash:
        raise ValueError('Source video changed during inference; rerun')
    # Bundle provenance must remain portable and must not disclose source or
    # temporary staging directories when the generated bundle is shared.
    detected.update(video=Path(video).name,pose_cache='pose.npz',
                    parameters=PARAMETERS,pose_sha256=sha256(bundle/'pose.npz'),video_sha256=source_hash)
    write_json(bundle/'candidates.json',detected)
    meta.update(video_sha256=source_hash,pose_sha256=sha256(bundle/'pose.npz'),
                candidates_sha256=sha256(bundle/'candidates.json'),
                versions=dict(python=sys.version.split()[0],mediapipe=mp.__version__,
                              opencv=cv2.__version__,numpy=np.__version__,torch=torch.__version__,
                              ultralytics=ultralytics.__version__),
                pose_model_sha256=sha256(Path(mp.__file__).parent/'modules/pose_landmark/pose_landmark_heavy.tflite'))
    write_json(bundle/'metadata.json',meta)
    return pose,detected,meta


def _source_metadata(video):
    """Read the original clock/dimensions without loading any inference models."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open source video for cache verification')
        source = {key: float(capture.get(prop)) for key, prop in (
            ('fps', cv2.CAP_PROP_FPS), ('frames', cv2.CAP_PROP_FRAME_COUNT),
            ('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT))}
        if any(not np.isfinite(value) or value <= 0 for value in source.values()):
            raise ValueError('Source video has invalid dimensions, frame count or FPS')
        for key in ('frames', 'width', 'height'):
            if not source[key].is_integer():
                raise ValueError(f'Source video {key} must be an integer')
            source[key] = int(source[key])
        return source
    finally:
        capture.release()


def _check_source_number(value, expected, name, *, fps=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f'Cache {name} must be a finite positive number')
    # Allow only floating-point/backend rounding, not a different frame clock.
    matches = np.isclose(value, expected, rtol=1e-6, atol=1e-9) if fps else value == expected
    if not matches:
        raise ValueError(f'Cache {name} differs from original source video')


def _validate_pose_clock(path, source):
    with np.load(path, allow_pickle=False) as z:
        if not {'frame_numbers', 'timestamps'} <= set(z.files):
            raise ValueError('Cached pose requires source frame_numbers and timestamps')
        frames, timestamps = z['frame_numbers'], z['timestamps']
        if (frames.ndim != 1 or frames.dtype.kind not in 'iu'
                or np.any(frames < 0) or np.any(frames >= source['frames'])
                or np.any(frames[1:] <= frames[:-1])):
            raise ValueError('Cached pose source indices must be unique, ascending and inside the video')
        if (timestamps.shape != frames.shape or timestamps.dtype.kind not in 'iuf'
                or not np.isfinite(timestamps).all()
                or not np.allclose(timestamps, frames / source['fps'], rtol=1e-7, atol=1e-7)):
            raise ValueError('Cached pose timestamps differ from source frame / FPS')


def load_cached_clip(bundle, video, roi):
    """Verify hashes and original source clock before expanding cached poses."""
    from scoring.current_metrics import load_pose
    meta=validate_cache(bundle,video,roi)
    source=_source_metadata(video)
    for key, value in source.items():
        _check_source_number(meta.get(key),value,key,fps=key=='fps')
    _validate_pose_clock(Path(bundle)/'pose.npz',source)
    pose=load_pose(Path(bundle)/'pose.npz',source['frames'])
    if (pose['width'],pose['height']) != (source['width'],source['height']):
        raise ValueError('Cached pose dimensions differ from original source video')
    candidates=json.loads((Path(bundle)/'candidates.json').read_text(encoding='utf-8'))
    if not isinstance(candidates,dict) or not isinstance(candidates.get('counts'),dict):
        raise ValueError('Cached candidates require source frame counts and FPS')
    _check_source_number(candidates.get('source_average_fps'),source['fps'],'candidate FPS',fps=True)
    _check_source_number(candidates.get('source_frame_count_metadata'),source['frames'],'candidate frame count')
    _check_source_number(candidates['counts'].get('source_frames'),source['frames'],'candidate source frames')
    # Sanitize older bundles in memory without changing their verified bytes.
    candidates={**candidates,'video':Path(video).name,'pose_cache':'pose.npz'}
    # Label reflects the supplied filename, not a prior cache filename.
    meta={**meta,**source,'name':Path(video).name}
    return pose,candidates,meta
