"""Legacy retracking must preserve caches unless replacement succeeds."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import retrack_balls as retrack


class RetrackSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cache, self.backups, self.videos = root / 'cache', root / 'backups', root / 'videos'
        self.cache.mkdir()
        self.videos.mkdir()
        for name, value in [('CACHE', self.cache), ('BACKUP', self.backups), ('VIDEO_DIR', self.videos)]:
            self.enter_patch(patch.object(retrack, name, value))
        self.stem = '인,인 3-1'
        (self.videos / f'{self.stem}.MOV').touch()
        self.ball = self.cache / f'{self.stem}_ball.json'
        self.original = b'{"2": [10, 20, 3], "5": [70, 80, 3]}\n'
        self.ball.write_bytes(self.original)
        self.ball.chmod(0o640)
        os.utime(self.ball, ns=(1_600_000_000_000_000_000,) * 2)
        self.pose = self.cache / f'{self.stem}_pose.npz'
        lm = np.zeros((2, 33, 3))
        lm[0, [27, 28], :2] = [.1, .1]
        lm[1, [27, 28], :2] = [.7, .4]
        np.savez(self.pose, frame_numbers=[2, 5], landmarks=lm,
                 frame_width=100, frame_height=200, timestamps=[.2, .5],
                 provenance=np.array('retain this metadata'))
        self.pose_bytes = self.pose.read_bytes()
        # Only model/decoder dependencies are replaced. Cache IO stays real.
        self.new_tracks = {2: (11, 20, 3), 5: (71, 80, 3)}
        self.enter_patch(patch('scoring.extraction_cache.load', return_value=([], {})))
        cap = SimpleNamespace(get=lambda prop: 8, release=lambda: None)
        self.enter_patch(patch.dict(sys.modules, {
            'cv2': SimpleNamespace(VideoCapture=lambda path: cap, CAP_PROP_FRAME_COUNT=7),
            'extract_depth_metrics': SimpleNamespace(run_sam2=lambda video, poses: self.new_tracks),
        }))
        self.enter_patch(contextlib.redirect_stdout(io.StringIO()))

    def enter_patch(self, manager):
        result = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return result

    def test_no_arguments_show_help_without_retracking(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['retrack_balls.py']), contextlib.redirect_stdout(output):
            with patch.object(retrack, 'retrack', side_effect=AssertionError('Unexpected inference')) as run:
                retrack.main()
        self.assertIn('usage:', output.getvalue())
        run.assert_not_called()
        self.assertEqual(self.ball.read_bytes(), self.original)

    def test_explicit_names_only_run_requested_clips(self):
        with patch.object(sys, 'argv', ['retrack_balls.py', '3-1', '기준1']):
            with patch.object(retrack, 'retrack', return_value=True) as run:
                retrack.main()
        self.assertEqual([call.args[0] for call in run.call_args_list], ['3-1', '기준1'])

    def test_all_requires_explicit_flag(self):
        with patch.object(sys, 'argv', ['retrack_balls.py', '--all']):
            with patch.object(retrack, 'retrack', return_value=True) as run:
                retrack.main()
        self.assertEqual([call.args[0] for call in run.call_args_list], retrack.ALL)

    def test_unknown_flag_and_mixed_targets_are_rejected(self):
        for arguments in [['--al'], ['--all', '3-1']]:
            with self.subTest(arguments=arguments), patch.object(sys, 'argv', ['retrack_balls.py'] + arguments):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    retrack.main()
                self.assertEqual(exc.exception.code, 2)
        self.assertEqual(self.ball.read_bytes(), self.original)

    def test_repeated_runs_keep_distinct_backups_and_pose_metadata(self):
        metadata = self.ball.stat()
        self.assertTrue(retrack.retrack('3-1'))
        first_replacement = self.ball.read_bytes()
        self.new_tracks = {2: (12, 20, 3), 5: (72, 80, 3)}
        self.assertTrue(retrack.retrack('3-1'))
        backups = list(self.backups.rglob('*.json'))
        self.assertEqual(len(backups), 2)
        self.assertEqual({p.read_bytes() for p in backups}, {self.original, first_replacement})
        self.assertEqual(json.loads(self.ball.read_text())['2'], [12, 20, 3])
        self.assertEqual(self.pose.read_bytes(), self.pose_bytes)
        self.assertEqual(stat.S_IMODE(self.ball.stat().st_mode), stat.S_IMODE(metadata.st_mode))
        self.assertEqual(self.ball.stat().st_mtime_ns, metadata.st_mtime_ns)

    def test_failed_atomic_replace_leaves_original_cache_readable(self):
        with patch.object(retrack.os, 'replace', side_effect=OSError('Simulated rename failure')):
            with self.assertRaises(OSError):
                retrack.retrack('3-1')
        self.assertEqual(self.ball.read_bytes(), self.original)
        self.assertEqual(self.pose.read_bytes(), self.pose_bytes)
        self.assertEqual([p.read_bytes() for p in self.backups.rglob('*.json')], [self.original])
        self.assertEqual(set(self.cache.iterdir()), {self.ball, self.pose})

    def test_empty_tracking_keeps_cache_without_backup(self):
        self.new_tracks = {}
        self.assertFalse(retrack.retrack('3-1'))
        self.assertEqual(self.ball.read_bytes(), self.original)
        self.assertFalse(self.backups.exists())

    def test_nonfinite_tracking_cannot_replace_valid_cache(self):
        self.new_tracks = {2: (float('nan'), 20, 3)}
        with self.assertRaises(ValueError):
            retrack.retrack('3-1')
        self.assertEqual(self.ball.read_bytes(), self.original)
        self.assertEqual(self.pose.read_bytes(), self.pose_bytes)

    def test_stats_use_source_frame_ids_and_source_frame_denominator(self):
        self.ball.write_text(json.dumps({'2': [10, 20, 3], '3': [90, 90, 3], '5': [70, 80, 3]}))
        coverage, distance = retrack.ball_foot_stats(self.stem, source_frames=8)
        self.assertEqual(coverage, 37.5)
        self.assertEqual(distance, 0.)


if __name__ == '__main__':
    unittest.main()
