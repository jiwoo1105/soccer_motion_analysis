"""Numeric and CLI boundary tests; run with Python 3.11's unittest runner."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

import numpy as np

from scoring.roi_ball import remap_detections, roi_for
import redetect_balls as cli
import extract_pose as pose_cli


def pose_fixture():
    landmarks = np.zeros((33, 3), dtype=float)
    landmarks[[11, 12], :2] = [[0.4, 0.25], [0.6, 0.25]]
    landmarks[[23, 24], :2] = [[0.4, 0.5], [0.6, 0.5]]
    landmarks[[27, 28], :2] = [[0.4, 0.75], [0.6, 0.8]]
    return landmarks, np.ones(33)


class RoiTests(unittest.TestCase):
    def test_two_ankles_use_twice_horizontal_and_once_vertical_torso(self):
        lm, vis = pose_fixture()
        row = roi_for(lm, vis, 7, (200, 400, 3), timestamp=0.25)
        self.assertEqual(row['roi_xyxy'], [60, 100, 340, 200])
        self.assertEqual(row['torso_length_px'], 50.0)
        self.assertEqual(row['ankles_global_xy'], [[160, 150], [240, 160]])
        self.assertEqual(row['pose_timestamp_s'], 0.25)
        self.assertEqual(row['status'], 'pending')

    def test_minimum_margin_and_floor_ceil_clipping(self):
        lm, vis = pose_fixture()
        lm[[11, 12, 23, 24], :2] = 0.5
        lm[[27, 28], :2] = [[0.0125, 0.02], [0.10, 0.11]]
        row = roi_for(lm, vis, 0, (100, 200))
        self.assertEqual(row['roi_xyxy'], [0, 0, 68, 35])
        self.assertEqual(row['torso_length_px'], 0.0)

    def test_missing_pose_has_no_roi_or_candidates(self):
        row = roi_for(None, None, 3, (200, 400))
        self.assertEqual(row['status'], 'missing_pose')
        self.assertIsNone(row['roi_xyxy'])
        self.assertEqual(row['detections'], [])
        json.dumps(row, allow_nan=False)

    def test_invalid_pose_never_becomes_whole_frame_crop(self):
        for invalid in ['nan_coordinate', 'inf_visibility', 'short_pose', 'bad_timestamp']:
            with self.subTest(invalid=invalid):
                lm, vis = pose_fixture()
                timestamp = 0.0
                if invalid == 'nan_coordinate':
                    lm[27, 0] = np.nan
                elif invalid == 'inf_visibility':
                    vis[11] = np.inf
                elif invalid == 'short_pose':
                    lm = lm[:20]
                else:
                    timestamp = np.nan
                row = roi_for(lm, vis, 0, (200, 400), timestamp)
                self.assertEqual(row['status'], 'invalid_roi')
                self.assertIsNone(row['roi_xyxy'])
                self.assertEqual(row['detections'], [])
                json.dumps(row, allow_nan=False)

    def test_ankles_outside_image_produce_invalid_empty_roi(self):
        lm, vis = pose_fixture()
        lm[[27, 28], :2] = [[3.0, 0.7], [3.2, 0.8]]
        row = roi_for(lm, vis, 0, (200, 400))
        self.assertEqual(row['status'], 'invalid_roi')
        self.assertEqual(row['roi_xyxy'], [400, 90, 400, 200])
        self.assertEqual(row['detections'], [])

    def test_low_visibility_is_recorded_without_filtering(self):
        lm, vis = pose_fixture()
        vis[:] = 0.0
        row = roi_for(lm, vis, 0, (200, 400))
        self.assertEqual(row['status'], 'pending')
        self.assertEqual(row['ankle_visibility'], [0.0, 0.0])

    def test_remapping_retains_all_boxes_and_fractional_geometry(self):
        boxes = np.array([[1.5, 2, 11.5, 22], [0, 1, 4, 7]])
        before = boxes.copy()
        detections = remap_detections(boxes, [0.8, 0.11], [32, 32], [60, 100, 340, 200])
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]['bbox_global_xyxy'], [61.5, 102, 71.5, 122])
        self.assertEqual(detections[0]['center_global_xy'], [66.5, 112])
        self.assertEqual(detections[0]['radius_px'], 7.5)
        self.assertEqual(detections[1]['confidence'], 0.11)
        np.testing.assert_array_equal(boxes, before)

    def test_remapping_rejects_nonfinite_or_misaligned_outputs(self):
        for boxes, confidence, classes in [
            ([[np.nan, 0, 1, 1]], [0.8], [32]),
            ([[0, 0, 1, 1]], [], [32]),
            ([[0, 0, 1, 1]], [np.inf], [32]),
        ]:
            with self.subTest(boxes=boxes, confidence=confidence):
                with self.assertRaises(ValueError):
                    remap_detections(boxes, confidence, classes, [0, 0, 100, 100])


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video_dir = self.root / 'videos'
        self.cache_dir = self.root / 'cache'
        self.video_dir.mkdir()
        self.cache_dir.mkdir()
        self.manifest = self.root / 'manifest.json'
        self.name = '인,인 3-1'
        self.manifest.write_text(json.dumps({'clips': [
            {'name': self.name, 'grade': 3, 'reference': False, 'exclude_reason': None},
            {'name': 'excluded', 'exclude_reason': 'Excluded from analysis'},
        ]}), encoding='utf-8')
        self.video = self.video_dir / (unicodedata.normalize('NFD', self.name) + '.mOv')
        self.video.touch()
        self.cache = self.cache_dir / (self.name + '_pose.npz')
        lm, vis = pose_fixture()
        np.savez(self.cache, frame_numbers=[0], timestamps=[0.0],
                 landmarks=[lm], visibility=[vis], frame_width=400, frame_height=200)

    def discover(self):
        return cli.discover_clips(self.video_dir, self.cache_dir, self.manifest)

    def test_unicode_and_extension_case_match_and_exclusions_skip_missing_files(self):
        clips = self.discover()
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]['clip'], self.name)
        self.assertEqual(clips[0]['video'], self.video)
        self.assertEqual(clips[0]['pose'], self.cache)

    def test_duplicate_normalized_video_stem_is_rejected(self):
        (self.video_dir / (self.name + '.mp4')).touch()
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
            self.discover()

    def test_duplicate_normalized_cache_stem_is_rejected(self):
        alias = self.cache_dir / (self.name + '_POSE.NPZ')
        alias.touch()
        # APFS aliases these paths. Supply the directory listing a case-sensitive
        # filesystem would expose; file reads and the discovery code stay real.
        original_iterdir = Path.iterdir

        def entries(directory):
            if directory == self.cache_dir:
                return iter([self.cache, alias])
            return original_iterdir(directory)

        with patch.object(Path, 'iterdir', entries):
            with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
                self.discover()

    def test_missing_video_or_cache_is_named(self):
        self.video.unlink()
        with self.assertRaisesRegex(FileNotFoundError, self.name):
            self.discover()
        self.video.touch()
        self.cache.unlink()
        with self.assertRaisesRegex(FileNotFoundError, self.name):
            self.discover()

    def test_duplicate_manifest_name_is_rejected(self):
        self.manifest.write_text(json.dumps({'clips': [
            {'name': self.name}, {'name': unicodedata.normalize('NFD', self.name)},
        ]}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
            self.discover()

    def test_pose_cache_schema_and_duplicate_frame_rejection(self):
        pose = cli.load_pose_cache(self.cache)
        self.assertEqual(pose['frame_width'], 400)
        self.assertEqual(list(pose['map']), [0])
        lm, vis = pose_fixture()
        np.savez(self.cache, frame_numbers=[0, 0], timestamps=[0.0, 0.1],
                 landmarks=[lm, lm], visibility=[vis, vis], frame_width=400, frame_height=200)
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
            cli.load_pose_cache(self.cache)

    def test_missing_weights_fail_before_heavy_imports_or_outputs(self):
        output = self.root / 'results'
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = cli.main(['--video-dir', str(self.video_dir), '--cache-dir', str(self.cache_dir),
                               '--manifest', str(self.manifest), '--output-dir', str(output),
                               '--model', str(self.root / 'absent.pt')])
        self.assertNotEqual(result, 0)
        self.assertIn('absent.pt', errors.getvalue())
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved_without_overwrite(self):
        target = self.root / 'candidates.json'
        target.write_text('existing', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            cli.save_json(target, {'new': True}, overwrite=False)
        self.assertEqual(target.read_text(), 'existing')
        cli.save_json(target, {'new': True}, overwrite=True)
        self.assertEqual(json.loads(target.read_text()), {'new': True})

    def test_existing_clip_aborts_cli_before_loading_weights(self):
        output = self.root / 'results'
        output.mkdir()
        target = output / (self.name + '_candidates.json')
        target.write_text('existing', encoding='utf-8')
        weights = self.root / 'weights.pt'
        weights.touch()  # Not loadable; preflight must reject the output first.
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = cli.main(['--video-dir', str(self.video_dir), '--cache-dir', str(self.cache_dir),
                               '--manifest', str(self.manifest), '--output-dir', str(output),
                               '--model', str(weights)])
        self.assertNotEqual(result, 0)
        self.assertIn('--overwrite', errors.getvalue())
        self.assertEqual(target.read_text(), 'existing')

    def test_source_paths_are_relative_only_inside_cwd(self):
        self.assertEqual(cli.source_path(Path.cwd() / 'some/file.mov'), 'some/file.mov')
        self.assertEqual(cli.source_path(self.video), str(self.video.resolve()))


class RuntimeTests(unittest.TestCase):
    def test_decode_skips_bad_pose_and_flushes_partial_batch_with_all_candidates(self):
        # Only the external decoder/model are replaced. Real ROI calculation,
        # batching, coordinate restoration, timing fields and counts run below.
        source = np.zeros((200, 400, 3), dtype=np.uint8)

        class Capture:
            def __init__(self):
                self.frames = iter([source] * 5)
                self.released = False

            def isOpened(self):
                return True

            def get(self, key):
                return {1: 5, 2: 25.0}[key]

            def read(self):
                frame = next(self.frames, None)
                return frame is not None, frame

            def release(self):
                self.released = True

        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        batches = []

        def predict(crops, **kwargs):
            self.assertEqual([crop.shape for crop in crops], [(100, 280, 3)] * len(crops))
            self.assertEqual(kwargs, {
                'device': 'cpu', 'imgsz': 640, 'conf': 0.1, 'iou': 0.7, 'classes': [32],
                'rect': False, 'half': False, 'verbose': False, 'save': False,
                'save_txt': False, 'save_crop': False, 'batch': len(crops),
            })
            batches.append(len(crops))
            return [SimpleNamespace(
                boxes=SimpleNamespace(xyxy=Tensor([[1, 2, 11, 22], [3, 4, 9, 12]]),
                                      conf=Tensor([0.9, 0.11]), cls=Tensor([32, 32])),
                speed={'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0},
            ) for _ in crops]

        capture = Capture()
        cv2 = SimpleNamespace(VideoCapture=lambda path: capture, CAP_PROP_FRAME_COUNT=1,
                              CAP_PROP_FPS=2)
        lm, vis = pose_fixture()
        bad = lm.copy()
        bad[27, 0] = np.nan
        pose = {'frame_width': 400, 'frame_height': 200, 'map': {
            0: (lm, vis, 0.0), 2: (bad, vis, 0.08),
            3: (lm, vis, 0.12), 4: (lm, vis, 0.16),
        }}
        clip = {'clip': 'test', 'video': Path('input/test.mov'), 'pose': Path('cache/test_pose.npz')}
        result = cli.process_clip(clip, pose, SimpleNamespace(predict=predict), cv2=cv2,
                                  torch=None, device='cpu', batch_size=2)
        self.assertEqual(batches, [2, 1])
        self.assertEqual([row['status'] for row in result['frames']],
                         ['observed', 'missing_pose', 'invalid_roi', 'observed', 'observed'])
        self.assertEqual(result['counts']['source_frames'], 5)
        self.assertEqual(result['counts']['inferred_frames'], 3)
        self.assertEqual(result['counts']['total_candidates'], 6)
        self.assertEqual(result['frames'][4]['source_frame_over_avg_fps_s'], 0.16)
        self.assertEqual(result['frames'][4]['detections'][0]['bbox_global_xyxy'], [61, 102, 71, 122])
        self.assertTrue(capture.released)
        json.dumps(result, allow_nan=False)

    def test_imports_and_help_do_not_import_inference_stack(self):
        root = Path(__file__).resolve().parents[1]
        program = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'cv2', 'torch', 'ultralytics'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import scoring.roi_ball
import redetect_balls
sys.argv = ['redetect_balls.py', '--help']
runpy.run_module('redetect_balls', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=root,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--overwrite', result.stdout)
        self.assertIn('--batch-size', result.stdout)

    def test_truncated_decode_reports_expected_and_actual_frame_count(self):
        class TruncatedCapture:
            released = False

            def isOpened(self):
                return True

            def get(self, key):
                return {1: 3, 2: 30.0}[key]

            def read(self):
                return False, None

            def release(self):
                self.released = True

        capture = TruncatedCapture()
        cv2 = SimpleNamespace(VideoCapture=lambda path: capture, CAP_PROP_FRAME_COUNT=1,
                              CAP_PROP_FPS=2)
        clip = {'clip': 'truncated', 'video': Path('truncated.mov'), 'pose': Path('unused.npz')}
        pose = {'map': {}, 'frame_width': 400, 'frame_height': 200}
        with self.assertRaisesRegex(RuntimeError, r'[Dd]ecode.*0.*3'):
            cli.process_clip(clip, pose, None, cv2=cv2, torch=None, device='cpu', batch_size=16)
        self.assertTrue(capture.released)


class PoseExtractionTests(unittest.TestCase):
    def test_cache_preserves_world_landmarks_and_sparse_frame_numbers(self):
        lm, vis = pose_fixture()
        world = np.full((33, 3), 0.125)
        frames = [SimpleNamespace(
            frame_number=number, timestamp=number / 25, landmarks=lm,
            world_landmarks=world, visibility=vis, frame_width=400, frame_height=200,
        ) for number in [0, 2]]
        arrays = pose_cli.pose_arrays(frames)
        self.assertEqual(set(arrays), {'frame_numbers', 'timestamps', 'landmarks',
                                      'world_landmarks', 'visibility', 'frame_width', 'frame_height'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_pose.npz'
            pose_cli.save_pose_cache(path, arrays)
            loaded = cli.load_pose_cache(path)
            self.assertEqual(list(loaded['map']), [0, 2])
            with np.load(path, allow_pickle=False) as cache:
                np.testing.assert_array_equal(cache['world_landmarks'], [world, world])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                pose_cli.save_pose_cache(path, arrays)
            self.assertEqual(path.read_bytes(), original)
            pose_cli.save_pose_cache(path, arrays, overwrite=True)

    def test_empty_extraction_fails_before_creating_cache(self):
        with self.assertRaisesRegex(ValueError, '[Ee]mpty|[Nn]o pose'):
            pose_cli.pose_arrays([])

    def test_nonfinite_or_mismatched_pose_rows_are_rejected(self):
        lm, vis = pose_fixture()
        frame = SimpleNamespace(frame_number=0, timestamp=0.0, landmarks=lm,
                                world_landmarks=lm.copy(), visibility=vis,
                                frame_width=400, frame_height=200)
        frame.world_landmarks[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pose_cli.pose_arrays([frame])
        frame.world_landmarks = lm.copy()
        frame.landmarks = lm[:20]
        with self.assertRaises(ValueError):
            pose_cli.pose_arrays([frame])

    def test_help_does_not_import_pose_or_detector_libraries(self):
        program = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'cv2', 'mediapipe', 'core', 'torch', 'ultralytics', 'sam2'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import extract_pose
sys.argv = ['extract_pose.py', '--help']
runpy.run_module('extract_pose', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-B', '-c', program],
                                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--manifest', result.stdout)
        self.assertIn('manual', result.stdout.lower())


if __name__ == '__main__':
    unittest.main()
