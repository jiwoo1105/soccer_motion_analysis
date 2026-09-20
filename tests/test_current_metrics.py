"""Input invariants and failure semantics of the published numeric path."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scoring.current_metrics import body_alignment, score_measurements, trunk_angle, load_pose, analyze_headup, head_timing_sensitivity


def fixture(n=80):
    lm = np.zeros((n, 33, 3), float)
    lm[:, 11, :2] = [.3, .2]
    lm[:, 12, :2] = [.7, .2]
    lm[:, 23, :2] = [.4, .6]
    lm[:, 24, :2] = [.6, .6]
    lm[:, 12, 1] += .08 * np.sin(np.arange(n) * .2)
    return dict(lm=lm, world=lm.copy(), vis=np.ones((n, 33)),
                width=640., height=480., n=n)


def head_fixture(n=180):
    p = fixture(n)
    p['lm'][:, 27, :2] = [.45, .7]
    p['lm'][:, 28, :2] = [.55, .7]
    p['world'][:, [11, 12]] = 0
    angle = np.deg2rad(20 + 10 * np.sin(np.arange(n) * .13))
    p['world'][:, 2, 0] = p['world'][:, 5, 0] = np.sin(angle)
    p['world'][:, 2, 1] = p['world'][:, 5, 1] = -np.cos(angle)
    frames = []
    for t in range(n):
        x, y = 320 + 50 * np.cos(2 * np.pi * t / 50), 336
        frames.append(dict(frame_number=t, status='observed', torso_length_px=100,
                           ankles_global_xy=[[288, 336], [352, 336]], detections=[dict(
                               center_global_xy=[x, y], radius_px=20,
                               bbox_global_xyxy=[x-20, y-20, x+20, y+20])]))
    return p, dict(counts={'source_frames': n}, frames=frames)


class CurrentMetricsTests(unittest.TestCase):
    def test_body_ignores_estimated_depth(self):
        p = fixture(); before = body_alignment(p)['M']
        p['lm'][:, :, 2] = np.nan
        p['world'][:] = np.nan
        self.assertAlmostEqual(body_alignment(p)['M'], before, places=12)

    def test_common_pixel_translation_rotation_and_scale(self):
        p = fixture(); before = body_alignment(p)['M']
        xy = p['lm'][:, :, :2] * [p['width'], p['height']]
        a = .6; rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        p['lm'][:, :, :2] = ((xy @ rot.T) * 2.7 + [300, 20]) / [p['width'], p['height']]
        self.assertAlmostEqual(body_alignment(p)['M'], before, places=12)

    def test_missing_frame_is_not_bridged(self):
        p = fixture(); p['vis'][35, 11] = .1
        result = body_alignment(p)
        self.assertFalse(result['mask'][29:42].any())
        self.assertTrue(result['mask'][28])
        self.assertTrue(result['mask'][42])

    def test_same_frame_support_for_all_windows(self):
        p = fixture(); p['vis'][35, 11] = .1
        masks = [body_alignment(p, w)['mask'] for w in [5, 9, 13]]
        np.testing.assert_array_equal(masks[0], masks[1])
        np.testing.assert_array_equal(masks[1], masks[2])

    def test_short_clip_is_missing_not_zero(self):
        self.assertIsNone(body_alignment(fixture(12))['M'])

    def test_one_supported_sample_is_missing_not_zero(self):
        self.assertIsNone(body_alignment(fixture(13))['M'])

    def test_source_gaps_and_missing_tail_are_retained(self):
        p = fixture(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pose.npz'
            np.savez(path, frame_numbers=np.array([0, 2, 5]), landmarks=p['lm'],
                     world_landmarks=p['world'], visibility=p['vis'], frame_width=640, frame_height=480)
            result = load_pose(path, 8)
            self.assertEqual(result['n'], 8)
            np.testing.assert_array_equal(result['frame_numbers'], [0, 2, 5])
            self.assertTrue(np.isnan(result['lm'][[1, 3, 4, 6, 7]]).all())
            np.savez(path, frame_numbers=np.array([0, 2, 2]), landmarks=p['lm'],
                     world_landmarks=p['world'], visibility=p['vis'], frame_width=640, frame_height=480)
            with self.assertRaises(ValueError): load_pose(path, 8)

    def test_head_timing_uses_common_valid_support(self):
        head = dict(head_angles=np.arange(80, dtype=float), head_valid=np.ones(80, bool),
                    events=[{'frame': 20}, {'frame': 50}])
        head['head_valid'][29] = False
        result = head_timing_sensitivity(head)
        self.assertEqual(result['common_event_count'], 1)
        self.assertEqual(result['means'], [None] * 5)

    def test_head_missing_track_does_not_create_events(self):
        p = fixture()
        result = analyze_headup(p, {'counts': {'source_frames': 80}, 'frames': []})
        self.assertIsNone(result['headup'])
        self.assertEqual(result['events'], [])

    def test_head_rejects_duplicate_source_indices(self):
        rec = {'frame_number': 0, 'status': 'missing_pose'}
        with self.assertRaises(ValueError):
            analyze_headup(fixture(), {'counts': {'source_frames': 80}, 'frames': [rec, rec]})

    def test_head_detects_periodic_direction_changes_on_source_clock(self):
        p, d = head_fixture()
        result = analyze_headup(p, d)
        self.assertEqual([e['frame'] for e in result['events']], [25, 50, 75, 100, 125, 150])
        self.assertTrue(np.isfinite(result['headup']))

    def test_head_interpolates_short_gap_but_rejects_imputed_center(self):
        p, d = head_fixture()
        d['frames'] = [r for r in d['frames'] if r['frame_number'] not in (49, 50)]
        result = analyze_headup(p, d)
        self.assertAlmostEqual(result['imputed_fraction'], 2 / 180)
        self.assertNotIn(50, [e['frame'] for e in result['events']])
        self.assertIn('imputed_center', next(e for e in result['rejected'] if e['frame'] == 50)['reasons'])

    def test_head_does_not_interpolate_long_gap(self):
        p, d = head_fixture()
        d['frames'] = [r for r in d['frames'] if r['frame_number'] not in (49, 50, 51, 52)]
        result = analyze_headup(p, d)
        self.assertEqual(result['imputed_fraction'], 0)
        self.assertTrue(np.isnan(result['ball'][49:53]).all())

    def test_single_valid_event_has_no_head_measurement(self):
        p, d = head_fixture(60)
        result = analyze_headup(p, d)
        self.assertEqual([e['frame'] for e in result['events']], [25])
        self.assertIsNone(result['headup'])

    def test_invalid_torso_length_is_missing(self):
        p = fixture(); p['lm'][:] = 0
        self.assertIsNone(body_alignment(p)['M'])

    def test_missing_metric_produces_no_total(self):
        cal = dict(headup={'opt':10., 'tau':5.}, trunk={'opt':140., 'tau':10.}, body_reference_mean=.1)
        out = score_measurements(dict(head_raw=None, trunk_raw=145., body_M=.05), cal)
        self.assertIsNone(out['total'])
        self.assertEqual(out['trunk_score'], 5.)
        self.assertEqual(out['body_score'], 5.)

    def test_equal_weight_total_and_clipping(self):
        cal = dict(headup={'opt':10., 'tau':5.}, trunk={'opt':140., 'tau':10.}, body_reference_mean=.1)
        out = score_measurements(dict(head_raw=12.5, trunk_raw=180., body_M=.2), cal)
        self.assertEqual(out['head_score'], 5.)
        self.assertEqual(out['trunk_score'], 0.)
        self.assertEqual(out['body_score'], 10.)
        self.assertEqual(out['total'], 5.)

    def test_nonfinite_trunk_does_not_poison_other_frames(self):
        p = fixture(); p['world'][:, 25, :]=[.4, .9, 0]
        p['world'][:, 26, :]=[.6, .9, 0]
        p['world'][0,11,:]=np.nan
        self.assertTrue(np.isfinite(trunk_angle(p)))


if __name__ == '__main__':
    unittest.main()
