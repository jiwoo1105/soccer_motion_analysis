# -*- coding: utf-8 -*-
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scoring.video_pipeline'),
                             'The single-video pipeline must exist')
        from scoring import video_pipeline
        self.mod = video_pipeline
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def video(self, frames=5):
        import cv2
        path = self.root / 'test video.avi'
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (96,64))
        self.assertTrue(out.isOpened())
        for i in range(frames):
            out.write(np.full((64,96,3), i*25, np.uint8))
        out.release()
        return path

    def cache(self):
        video = self.video()
        bundle = self.root / 'cache'
        bundle.mkdir(exist_ok=True)
        pose = dict(n=5, width=96, height=64, lm=np.full((5,33,3), .5),
                    world=np.full((5,33,3), .5), vis=np.ones((5,33)))
        pose['lm'][2] = pose['world'][2] = np.nan
        pose['vis'][2] = 0
        meta = dict(name=video.name, frames=5, width=96, height=64, fps=30.,
                    extraction_roi=None, video_sha256=self.mod.sha256(video))
        self.mod.save_pose(pose, meta, bundle/'pose.npz')
        candidates = dict(counts={'source_frames': 5}, frames=[],
                          source_average_fps=30., source_frame_count_metadata=5,
                          video=str(video), pose_cache=str(bundle/'pose.npz'))
        self.mod.write_json(bundle/'candidates.json', candidates)
        self.refresh_cache(bundle, meta)
        return video, bundle, meta

    def refresh_cache(self, bundle, meta):
        meta.update(pose_sha256=self.mod.sha256(bundle/'pose.npz'),
                    candidates_sha256=self.mod.sha256(bundle/'candidates.json'))
        self.mod.write_json(bundle/'metadata.json', meta)

    def test_replay_preserves_missing_source_rows(self):
        video, bundle, _ = self.cache()
        pose, _, meta = self.mod.load_cached_clip(bundle, video, None)
        self.assertEqual((pose['n'], meta['fps']), (5, 30.))
        np.testing.assert_array_equal(pose['frame_numbers'], [0, 1, 2, 3, 4])
        self.assertTrue(np.isnan(pose['world'][2]).all())
        self.assertTrue(np.isfinite(pose['world'][3]).all())

    def test_replay_rejects_metadata_not_matching_original_video(self):
        for field, value in [('fps', 60.), ('fps', 0), ('fps', True),
                             ('fps', '30'), ('frames', 6), ('width', 192), ('height', 128)]:
            with self.subTest(field=field, value=value):
                video, bundle, meta = self.cache()
                meta[field] = value
                self.refresh_cache(bundle, meta)
                with self.assertRaises(ValueError):
                    self.mod.load_cached_clip(bundle, video, None)

    def test_replay_rejects_candidate_fps_and_counts_even_with_updated_hash(self):
        for field, value in [('source_average_fps', 60.), ('source_average_fps', None),
                             ('source_average_fps', True), ('source_average_fps', '30'),
                             ('source_frame_count_metadata', 6), ('counts', {'source_frames': 6})]:
            with self.subTest(field=field, value=value):
                video, bundle, meta = self.cache()
                candidates = json.loads((bundle/'candidates.json').read_text())
                candidates[field] = value
                self.mod.write_json(bundle/'candidates.json', candidates)
                self.refresh_cache(bundle, meta)
                with self.assertRaises(ValueError):
                    self.mod.load_cached_clip(bundle, video, None)

    def test_replay_checks_pose_timestamps_against_stored_source_indices(self):
        for timestamps in (np.arange(5)/60., np.arange(5)/30. + .1,
                           np.array([0., 1/30, np.nan, .1, 4/30]),
                           np.zeros((5,1)), np.array(['0']*5)):
            with self.subTest(timestamps=timestamps):
                video, bundle, meta = self.cache()
                with np.load(bundle/'pose.npz', allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                arrays['timestamps'] = timestamps
                np.savez_compressed(bundle/'pose.npz', **arrays)
                self.refresh_cache(bundle, meta)
                with self.assertRaises(ValueError):
                    self.mod.load_cached_clip(bundle, video, None)

    def test_replay_checks_original_even_when_cache_files_agree(self):
        for field, value in [('fps', 60.), ('width', 192), ('height', 128), ('frames', 6)]:
            with self.subTest(field=field):
                video, bundle, meta = self.cache()
                meta[field] = value
                with np.load(bundle/'pose.npz', allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                candidates = json.loads((bundle/'candidates.json').read_text())
                if field == 'fps':
                    arrays['timestamps'] = np.arange(5)/60.
                    candidates['source_average_fps'] = 60.
                elif field == 'frames':
                    candidates['source_frame_count_metadata'] = 6
                    candidates['counts']['source_frames'] = 6
                else:
                    arrays['frame_' + field] = value
                np.savez_compressed(bundle/'pose.npz', **arrays)
                self.mod.write_json(bundle/'candidates.json', candidates)
                self.refresh_cache(bundle, meta)
                with self.assertRaisesRegex(ValueError, 'source|Source'):
                    self.mod.load_cached_clip(bundle, video, None)

    def test_replay_rejects_missing_timestamps_and_bad_source_indices(self):
        for frames in (None, [0, 1, 1, 3, 4], [0, 1, 2, 3, 5],
                       np.array([0, 2, 1, 3, 4], dtype=np.uint64)):
            with self.subTest(frames=frames):
                video, bundle, meta = self.cache()
                with np.load(bundle/'pose.npz', allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                if frames is None:
                    arrays.pop('timestamps')
                else:
                    arrays['frame_numbers'] = np.asarray(frames)
                np.savez_compressed(bundle/'pose.npz', **arrays)
                self.refresh_cache(bundle, meta)
                with self.assertRaises(ValueError):
                    self.mod.load_cached_clip(bundle, video, None)

    def test_replay_accepts_sparse_source_indices_without_retiming(self):
        video, bundle, meta = self.cache()
        with np.load(bundle/'pose.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for key in ('frame_numbers', 'timestamps', 'landmarks', 'world_landmarks', 'visibility'):
            arrays[key] = arrays[key][[0, 3, 4]]
        np.savez_compressed(bundle/'pose.npz', **arrays)
        self.refresh_cache(bundle, meta)
        pose, _, _ = self.mod.load_cached_clip(bundle, video, None)
        self.assertEqual(pose['n'], 5)
        self.assertTrue(np.isnan(pose['world'][1:3]).all())
        self.assertTrue(np.isfinite(pose['world'][3:]).all())

    def test_replay_sanitizes_legacy_paths_without_mutating_cache(self):
        video, bundle, _ = self.cache()
        before = (bundle/'candidates.json').read_bytes()
        _, candidates, _ = self.mod.load_cached_clip(bundle, video, None)
        self.assertEqual(candidates['video'], 'test video.avi')
        self.assertEqual(candidates['pose_cache'], 'pose.npz')
        self.assertEqual((bundle/'candidates.json').read_bytes(), before)

    def test_fresh_bundle_saves_portable_candidate_paths(self):
        video, old_bundle, meta = self.cache()
        from scoring.current_metrics import load_pose
        pose = load_pose(old_bundle/'pose.npz', 5)
        asset = self.root/'fake-package/modules/pose_landmark/pose_landmark_heavy.tflite'
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b'fixture asset')
        fake_mp = SimpleNamespace(__file__=str(self.root/'fake-package/__init__.py'), __version__='test')
        bundle = self.root/'.dribble-report-staging/clips/clip-01'
        detected = json.loads((old_bundle/'candidates.json').read_text())
        detected['pose_cache'] = str(bundle/'pose.npz')
        with patch.dict(sys.modules, {'torch': SimpleNamespace(__version__='test'),
                                     'mediapipe': fake_mp, 'ultralytics': SimpleNamespace(__version__='test')}), \
                patch.object(self.mod, 'extract_pose_video', return_value=(pose, meta)), \
                patch('redetect_balls.process_clip', return_value=detected):
            _, returned, saved_meta = self.mod.infer_clip(video, bundle, object())
        saved = json.loads((bundle/'candidates.json').read_text())
        for candidates in (returned, saved):
            self.assertEqual(candidates['video'], 'test video.avi')
            self.assertEqual(candidates['pose_cache'], 'pose.npz')
        self.assertNotIn(str(self.root), json.dumps(saved))
        self.assertEqual(saved_meta['candidates_sha256'], self.mod.sha256(bundle/'candidates.json'))
        self.mod.load_cached_clip(bundle, video, None)

    def test_missing_pose_preserves_source_clock_and_crop_coordinates(self):
        class Tracker:
            calls = 0
            def process(self, image):
                self.calls += 1
                if self.calls == 3:
                    return SimpleNamespace(pose_landmarks=None)
                lm = SimpleNamespace(x=.5,y=.5,z=.1,visibility=1.)
                return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[lm]*33),
                                       pose_world_landmarks=SimpleNamespace(landmark=[lm]*33))
        pose, meta = self.mod.extract_pose_video(self.video(), Tracker(), roi=(24,16,48,32))
        self.assertEqual(meta['frames'], 5)
        self.assertEqual(pose['world'].shape, (5,33,3))
        self.assertTrue(np.isnan(pose['world'][2]).all())
        self.assertTrue(np.isfinite(pose['world'][3]).all())
        np.testing.assert_allclose(pose['lm'][0,0], [.5,.5,.05])
        self.assertAlmostEqual(meta['fps'], 30)

    def test_cache_rejects_changed_source_even_same_name(self):
        path = self.root / 'video.mov'
        path.write_bytes(b'original')
        bundle = self.root / 'cache'
        bundle.mkdir()
        (bundle/'pose.npz').write_bytes(b'pose')
        (bundle/'candidates.json').write_text('{}')
        meta = dict(video_sha256=self.mod.sha256(path),
                    pose_sha256=self.mod.sha256(bundle/'pose.npz'),
                    candidates_sha256=self.mod.sha256(bundle/'candidates.json'),
                    extraction_roi=None)
        (bundle/'metadata.json').write_text(json.dumps(meta))
        self.mod.validate_cache(bundle,path,None)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source|Source|video'):
            self.mod.validate_cache(bundle,path,None)

    def test_cache_rejects_changed_pose_or_crop(self):
        path=self.root/'v.mov';path.write_bytes(b'v')
        c=self.root/'cache';c.mkdir()
        (c/'pose.npz').write_bytes(b'p')
        (c/'candidates.json').write_text('{}')
        meta=dict(video_sha256=self.mod.sha256(path),
                  pose_sha256=self.mod.sha256(c/'pose.npz'),
                  candidates_sha256=self.mod.sha256(c/'candidates.json'),
                  extraction_roi=None)
        (c/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'ROI|roi|crop'):
            self.mod.validate_cache(c,path,(0,0,5,5))
        (c/'pose.npz').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError,'pose|hash|SHA'):
            self.mod.validate_cache(c,path,None)

    def test_empty_video_and_invalid_crop_fail_before_scoring(self):
        path=self.root/'bad.mp4';path.write_bytes(b'not a video')
        with self.assertRaises(ValueError):
            self.mod.extract_pose_video(path,None)
        with self.assertRaisesRegex(ValueError,'ROI|roi|crop'):
            self.mod.extract_pose_video(self.video(),None,roi=(80,0,40,50))

    def test_existing_output_never_deleted_on_invalid_input(self):
        out=self.root/'out';out.mkdir()
        marker=out/'mine.txt';marker.write_text('keep')
        proc=subprocess.run([sys.executable,'analyze_video.py','--video',str(self.root/'missing.mov'),
                             '--output',str(out)],text=True,capture_output=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertEqual(marker.read_text(),'keep')

    def test_help_is_available_without_model_packages(self):
        program = """import builtins,runpy,sys
real=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','mediapipe','ultralytics','cv2'}:
        raise AssertionError('heavy import on help')
    return real(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['analyze_video.py','--help']
runpy.run_path('analyze_video.py',run_name='__main__')
"""
        p=subprocess.run([sys.executable,'-c',program],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--video',p.stdout)
        self.assertIn('--weights',p.stdout)


if __name__ == '__main__':
    unittest.main()
