"""Exercise the original SAM2 tracking policy without model weights or downloads."""
from contextlib import nullcontext
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('scoring.sam2_tracking'), 'Local SAM2 backend is required'
    return importlib.import_module('scoring.sam2_tracking')


@pytest.fixture
def clip(tmp_path):
    import cv2
    video = tmp_path / 'video.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (96, 64))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((64, 96, 3), i * 20, np.uint8))
    writer.release()
    pose = dict(n=8, width=96, height=64, lm=np.full((8, 33, 3), .5), vis=np.ones((8, 33)))
    pose['lm'][:, [27, 28], :2] = [40 / 96, 50 / 64]
    checkpoint = tmp_path / 'sam.pt'
    checkpoint.write_bytes(b'local fixture')
    return video, pose, checkpoint


def box(x, y, radius=4, cls=32):
    return SimpleNamespace(cls=[cls], xyxy=np.array([[x-radius, y-radius, x+radius, y+radius]]))


class Detector:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [box(40, 50)]
        self.devices = []

    def __call__(self, image, *, conf, verbose, device):
        assert conf == .3 and verbose is False
        self.devices.append(device)
        return [SimpleNamespace(boxes=self.boxes)]


class Mask:
    def __init__(self, x, y):
        self.value = np.zeros((64, 96), dtype=float)
        self.value[y:y+2, x:x+2] = 1

    def __getitem__(self, index):
        assert index == 0
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Predictor:
    def __init__(self, fail=False):
        self.directory = None
        self.prompts = []
        self.fail = fail

    def init_state(self, *, video_path):
        self.directory = Path(video_path)
        assert len(list(self.directory.glob('*.jpg'))) == 8
        return {}

    def reset_state(self, state):
        pass

    def add_new_points_or_box(self, **kwargs):
        self.prompts.append(kwargs)

    def propagate_in_video(self, state, *, start_frame_idx):
        if self.fail:
            raise RuntimeError('predictor failure')
        for i in range(start_frame_idx, 8):
            if i != 3:  # Missing mask remains a missing source index.
                yield i, [1], [Mask(40 + i, 50)]


def fake_builder(monkeypatch, predictor):
    calls = []
    def build(config, checkpoint, *, device):
        calls.append((config, checkpoint, device))
        return predictor
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'sam2.build_sam', SimpleNamespace(build_sam2_video_predictor=build))
    return calls


def test_tracks_keep_original_indices_centroids_and_local_device_with_cleanup(clip, monkeypatch):
    module = api()
    video, pose, checkpoint = clip
    predictor, detector = Predictor(), Detector()
    calls = fake_builder(monkeypatch, predictor)
    tracks = module.track_ball(video, pose, detector, checkpoint, device='cpu')
    assert sorted(tracks) == [0, 1, 2, 4, 5, 6, 7]
    assert tracks[4] == pytest.approx((44.5, 50.5, np.sqrt(4 / np.pi)))
    assert calls == [('configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint.resolve()), 'cpu')]
    assert detector.devices and set(detector.devices) == {'cpu'}
    assert not predictor.directory.exists()
    assert predictor.prompts[0]['frame_idx'] == 0


def test_prediction_failure_cleans_extracted_frames(clip, monkeypatch):
    module = api()
    video, pose, checkpoint = clip
    predictor = Predictor(fail=True)
    fake_builder(monkeypatch, predictor)
    with pytest.raises(RuntimeError, match='predictor failure'):
        module.track_ball(video, pose, Detector(), checkpoint)
    assert not predictor.directory.exists()


def test_missing_checkpoint_fails_before_any_model_or_decode(clip, monkeypatch):
    module = api()
    video, pose, checkpoint = clip
    checkpoint.unlink()
    monkeypatch.setitem(sys.modules, 'sam2.build_sam', None)
    with pytest.raises(FileNotFoundError, match='SAM2|checkpoint'):
        module.track_ball(video, pose, Detector(), checkpoint)


def test_incomplete_decode_fails_and_cleans_temp_directory(clip, monkeypatch):
    module = api()
    video, pose, checkpoint = clip
    pose['n'] = 9
    pose['lm'] = np.concatenate([pose['lm'], pose['lm'][-1:]])
    created = []
    original = module.tempfile.TemporaryDirectory
    def temporary(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(Path(result.name))
        return result
    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', temporary)
    with pytest.raises(ValueError, match='decode|frames'):
        module.track_ball(video, pose, Detector(), checkpoint)
    assert created and all(not path.exists() for path in created)


def test_no_yolo_prompt_returns_empty_track_without_roi_fallback(clip, monkeypatch):
    module = api()
    video, pose, checkpoint = clip
    monkeypatch.setitem(sys.modules, 'sam2.build_sam', None)
    assert module.track_ball(video, pose, Detector([]), checkpoint) == {}


def test_cache_producer_ankles_use_all_valid_source_frames():
    module = api()
    pose = dict(n=25, width=100, height=200, lm=np.full((25, 33, 3), .5), vis=np.zeros((25, 33)))
    pose['lm'][3, 27, 0] = np.nan
    ankles = module.ankle_positions(pose)
    assert sorted(ankles) == [i for i in range(25) if i != 3]
    assert ankles[0] == [[50., 100.], [50., 100.]]


def test_backend_module_import_is_lazy():
    api()
    script = """import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'sam2', 'torch', 'ultralytics', 'cv2'}:
        raise AssertionError('eager inference import: ' + name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
import scoring.sam2_tracking
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_loss_gate_after_frame_twenty_discards_bad_segment_and_reprompts(tmp_path, monkeypatch):
    import cv2
    module = api()
    video = tmp_path / 'long.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30., (640, 64))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((64, 640, 3), i * 6, np.uint8))
    writer.release()
    pose = dict(n=30, width=640, height=64, lm=np.full((30, 33, 3), .5))
    pose['lm'][:, [27, 28], :2] = [40 / 640, 50 / 64]
    pose['lm'][20:, [27, 28], 0] = 600 / 640
    checkpoint = tmp_path / 'sam.pt'
    checkpoint.write_bytes(b'local fixture')

    class MovingDetector(Detector):
        def __call__(self, image, **kwargs):
            self.boxes = [box(600 if image[0, 0, 0] >= 120 else 40, 50)]
            return super().__call__(image, **kwargs)

    class WideMask(Mask):
        def __init__(self, x):
            self.value = np.zeros((64, 640), dtype=float)
            self.value[50:52, x:x+2] = 1.

    class LosingPredictor(Predictor):
        def init_state(self, *, video_path):
            self.directory = Path(video_path)
            assert len(list(self.directory.glob('*.jpg'))) == 30
            return {}

        def propagate_in_video(self, state, *, start_frame_idx):
            for i in range(start_frame_idx, 30):
                # First propagation sticks to the background ball after frame20.
                yield i, [1], [WideMask(40 if start_frame_idx == 0 else 600)]

    predictor = LosingPredictor()
    fake_builder(monkeypatch, predictor)
    tracks = module.track_ball(video, pose, MovingDetector(), checkpoint)
    assert [prompt['frame_idx'] for prompt in predictor.prompts] == [0, 21]
    assert 20 not in tracks
    assert tracks[19][0] == 40.5
    assert tracks[21][0] == tracks[29][0] == 600.5
    assert not predictor.directory.exists()
