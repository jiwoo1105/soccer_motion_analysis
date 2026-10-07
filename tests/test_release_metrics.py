"""Model-free release contracts, with hand-derived source-clock fixtures."""
import copy
import importlib
import json
from pathlib import Path
import unittest

import numpy as np


def pose_fixture(n=180, speed=30., fps=30., trunk_degrees=150.):
    lm = np.full((n, 33, 3), .5)
    lm[:, 27, :2], lm[:, 28, :2] = [.45, .7], [.55, .7]
    world = np.zeros((n, 33, 3))
    phase = np.deg2rad(np.arange(n) * speed / fps)
    direction = np.column_stack((np.cos(phase), np.zeros(n), np.sin(phase)))
    world[:, 11], world[:, 12] = -.2 * direction, .2 * direction
    # Each hip-to-shoulder vector is vertical; hip-to-knee angle is known.
    for shoulder, hip, knee in ((11, 23, 25), (12, 24, 26)):
        world[:, hip] = world[:, shoulder] + [0, .5, 0]
        theta = np.deg2rad(trunk_degrees)
        world[:, knee] = world[:, hip] + [.5 * np.sin(theta), -.5 * np.cos(theta), 0]
    head_angle = np.deg2rad(20 + .25 * np.arange(n))
    for eye in (2, 5):
        world[:, eye] = np.column_stack((np.sin(head_angle), -np.cos(head_angle), np.zeros(n)))
    return dict(n=n, width=640., height=480., lm=lm, world=world, vis=np.ones((n, 33)))


def balls(n=180, period=50., phase=0.):
    frames = []
    for t in range(n):
        x, y = 320 + 50 * np.cos(2 * np.pi * (t - phase) / period), 336.
        frames.append(dict(frame_number=t, status='observed', torso_length_px=100.,
                           ankles_global_xy=[[288., 336.], [352., 336.]], detections=[dict(
                               center_global_xy=[x, y], radius_px=20.,
                               bbox_global_xyxy=[x - 20, y - 20, x + 20, y + 20])]))
    return dict(counts={'source_frames': n}, frames=frames)


class ReleaseMetricsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.measure = importlib.import_module('scoring.release_metrics').measure_video
        except ModuleNotFoundError as exc:
            if exc.name != 'scoring.release_metrics':
                raise
            self.fail('The release measure_video API has not been implemented')

    def test_constant_xyz_rotation_has_known_speed_and_source_support(self):
        result = self.measure(pose_fixture(), balls(), 30.)
        sh = result['metrics']['shoulder']
        self.assertAlmostEqual(sh['raw'], 30., places=8)
        self.assertEqual(sh['unit'], 'deg/s')
        self.assertEqual(sh['valid_frames'], 158)
        self.assertEqual(sh['valid_pairs'], 157)
        diag = result['diagnostics']['shoulder']
        np.testing.assert_array_equal(np.flatnonzero(diag['support']), np.arange(11, 169))
        np.testing.assert_array_equal(np.flatnonzero(diag['pair_support']), np.arange(12, 169))
        self.assertTrue(np.isnan(diag['geo_speed'][~diag['pair_support']]).all())

    def test_arbitrary_rest_direction_has_exactly_zero_speed(self):
        p = pose_fixture(speed=0)
        p['world'][:, 12] = [.2, .1, .3]
        p['world'][:, 11] = [-.2, -.1, -.3]
        metric = self.measure(p, None, 30.)['metrics']['shoulder']
        self.assertEqual(metric['raw'], 0.)
        self.assertEqual(metric['score'], 0.)

    def test_direction_speed_is_invariant_to_gradual_length_and_translation(self):
        p = pose_fixture()
        p['world'] *= np.linspace(1, 3, p['n'])[:, None, None]
        p['world'] += [10, -20, 40]
        self.assertAlmostEqual(self.measure(p, None, 30.)['metrics']['shoulder']['raw'], 30., places=8)

    def test_xyz_tilt_motion_is_measured_not_only_yaw(self):
        p = pose_fixture()
        a = np.deg2rad(np.arange(p['n']) * .3)
        direction = np.column_stack((np.cos(a), np.sin(a), np.zeros(p['n'])))
        p['world'][:, 11], p['world'][:, 12] = -.2 * direction, .2 * direction
        self.assertAlmostEqual(self.measure(p, None, 30.)['metrics']['shoulder']['raw'], 9., places=8)

    def test_fps_scales_adjacent_speed(self):
        p = pose_fixture()
        self.assertAlmostEqual(self.measure(p, None, 60.)['metrics']['shoulder']['raw'], 60., places=8)

    def test_gap_splits_filter_runs_and_never_bridges_pairs(self):
        p = pose_fixture(100, speed=0)
        p['vis'][50, 11] = .1
        p['world'][51:, 11] = [0, 0, -.2]
        p['world'][51:, 12] = [0, 0, .2]
        r = self.measure(p, None, 30.)
        sh = r['diagnostics']['shoulder']
        self.assertFalse(sh['support'][39:63].any())
        self.assertTrue(sh['support'][38])
        self.assertTrue(sh['support'][63])
        self.assertFalse(sh['pair_support'][39:64].any())
        self.assertEqual(r['metrics']['shoulder']['raw'], 0.)

    def test_minimum_run_is_24_with_two_centers_and_one_pair(self):
        for n, pairs in ((23, 0), (24, 1)):
            with self.subTest(n=n):
                r = self.measure(pose_fixture(n), None, 30.)['metrics']['shoulder']
                self.assertEqual(r['valid_pairs'], pairs)
                if pairs:
                    self.assertAlmostEqual(r['raw'], 30., places=8)
                else:
                    self.assertIsNone(r['raw'])
                    self.assertTrue(r['reason'])

    def test_automatic_angle_and_length_jumps_remove_both_endpoints(self):
        for angle, scale in ((60., 1.), (0., 1.6), (0., 0.)):
            with self.subTest(angle=angle, scale=scale):
                p = pose_fixture(100, speed=0)
                a = np.deg2rad(angle)
                v = scale * np.array([.2 * np.cos(a), 0, .2 * np.sin(a)])
                p['world'][50:, 11], p['world'][50:, 12] = -v, v
                d = self.measure(p, None, 30.)['diagnostics']['shoulder']
                np.testing.assert_array_equal(np.flatnonzero(d['auto_excluded']), [49, 50])
                self.assertFalse(d['native_valid'][49:51].any())
                self.assertFalse(d['support'][38:62].any())

    def test_visibility_bounds_nonfinite_and_image_bounds_mask_shoulders(self):
        for field, index, value in [('vis', (50, 11), 1.1), ('vis', (50, 11), np.inf),
                                    ('world', (50, 11, 0), np.nan), ('lm', (50, 11, 0), 1.01),
                                    ('lm', (50, 11, 2), np.nan)]:
            with self.subTest(field=field, value=value):
                p = pose_fixture(100)
                p[field][index] = value
                d = self.measure(p, None, 30.)['diagnostics']['shoulder']
                self.assertFalse(d['native_valid'][50])
                self.assertFalse(d['support'][50])

    def test_shoulder_support_does_not_depend_on_hips_or_ball(self):
        p = pose_fixture()
        p['world'][:, [23, 24, 25, 26]] = np.nan
        r = self.measure(p, None, 30.)
        self.assertAlmostEqual(r['metrics']['shoulder']['raw'], 30., places=8)
        self.assertIsNone(r['metrics']['trunk']['raw'])
        self.assertIsNone(r['total'])

    def test_explicit_exclusions_apply_on_source_clock_without_input_mutation(self):
        p, c = pose_fixture(100), balls(100)
        before_p, before_c = copy.deepcopy(p), copy.deepcopy(c)
        r = self.measure(p, c, 30., excluded_frames=[50])
        d = r['diagnostics']
        self.assertTrue(d['excluded_frames'][50])
        self.assertFalse(d['trunk']['valid'][50])
        self.assertFalse(d['headup']['head_valid'][50])
        self.assertFalse(d['shoulder']['support'][39:62].any())
        mask = np.zeros(100, bool)
        mask[50] = True
        r2 = self.measure(p, c, 30., excluded_frames=mask)
        np.testing.assert_array_equal(d['shoulder']['support'], r2['diagnostics']['shoulder']['support'])
        for key in ('lm', 'world', 'vis'):
            np.testing.assert_array_equal(p[key], before_p[key])
        self.assertEqual(c, before_c)

    def test_head_expands_old_events_to_33_frame_range(self):
        r = self.measure(pose_fixture(), balls(), 30.)
        head = r['metrics']['headup']
        self.assertAlmostEqual(head['raw'], 8., places=7)
        self.assertEqual(head['valid_events'], 6)
        d = r['diagnostics']['headup']
        self.assertEqual([e['frame'] for e in d['events']], [25, 50, 75, 100, 125, 150])
        self.assertTrue(all(e['end'] - e['start'] == 32 for e in d['events']))

    def test_head_rejects_outer_window_missing_pose_after_old_acceptance(self):
        p = pose_fixture()
        p['vis'][9, 2] = .1  # Outside event25 ±8, inside its ±16.
        r = self.measure(p, balls(), 30.)
        d = r['diagnostics']['headup']
        self.assertIn(25, [e['frame'] for e in d['legacy_events']])
        self.assertNotIn(25, [e['frame'] for e in d['events']])
        self.assertIn('head_pose_invalid', next(e for e in d['rejected_expanded'] if e['frame'] == 25)['reasons'])

    def test_head_expansion_rejects_source_edges(self):
        d = self.measure(pose_fixture(100), balls(100, phase=15.), 30.)['diagnostics']['headup']
        self.assertIn(15, [e['frame'] for e in d['legacy_events']])
        self.assertNotIn(15, [e['frame'] for e in d['events']])
        self.assertIn('source_edge', next(e for e in d['rejected_expanded'] if e['frame'] == 15)['reasons'])

    def test_head_requires_two_valid_events_and_missing_means_no_total(self):
        r = self.measure(pose_fixture(60), balls(60), 30., weights={'headup': 0, 'trunk': 1, 'shoulder': 0})
        self.assertEqual(r['metrics']['headup']['valid_events'], 1)
        self.assertIsNone(r['metrics']['headup']['raw'])
        self.assertIsNone(r['metrics']['headup']['score'])
        self.assertTrue(r['metrics']['headup']['reason'])
        self.assertIsNone(r['total'])

    def test_old_ball_proxy_filters_still_reject_imputed_center(self):
        c = balls()
        c['frames'] = [r for r in c['frames'] if r['frame_number'] not in (49, 50)]
        d = self.measure(pose_fixture(), c, 30.)['diagnostics']['headup']
        self.assertNotIn(50, [e['frame'] for e in d['events']])
        self.assertTrue(d['imputed'][49:51].all())
        self.assertIn('imputed_center', next(e for e in d['legacy_rejected'] if e['frame'] == 50)['reasons'])

    def test_trunk_uses_bilateral_knee_hip_shoulder_mean_and_frozen_reference(self):
        p = pose_fixture(trunk_degrees=150.)
        p['world'][:, 26] = p['world'][:, 24] + [0, .5, 0]  # Right180; left150 =>165.
        r = self.measure(p, None, 30.)['metrics']['trunk']
        self.assertAlmostEqual(r['raw'], 165., places=10)
        self.assertAlmostEqual(r['score'], 10 * 15 / 36.4628217436476, places=10)
        self.assertEqual(r['valid_frames'], 180)

    def test_trunk_degenerate_or_incomplete_bilateral_frames_are_missing(self):
        p = pose_fixture()
        p['world'][:, 25] = p['world'][:, 23]
        r = self.measure(p, None, 30.)['metrics']['trunk']
        self.assertIsNone(r['raw'])
        self.assertEqual(r['valid_frames'], 0)

    def test_head_scores_use_frozen_anchors_and_clip_without_refitting(self):
        observed = []
        for raw, want in ((0., 3.), (10.225053625082124, 3.),
                          (12.765022011980992, 6.5), (15.30499039887986, 10.), (20., 10.)):
            p = pose_fixture()
            a = np.deg2rad(10 + np.arange(p['n']) * raw / 32)
            p['world'][:, 2] = p['world'][:, 5] = np.column_stack((np.sin(a), -np.cos(a), np.zeros(p['n'])))
            r = self.measure(p, balls(), 30.)
            self.assertAlmostEqual(r['metrics']['headup']['score'], want, places=7)
            observed.append(r['metrics']['headup']['score'])
        self.assertEqual(observed, sorted(observed))

    def test_default_and_override_weight_totals_use_unrounded_component_scores(self):
        p, c = pose_fixture(), balls()
        r = self.measure(p, c, 30.)
        want = .1 * 3 + .8 * (10 * 30 / 36.4628217436476) + .1 * (10 * 30 / 93.9849170623855)
        self.assertAlmostEqual(r['total'], want, places=8)
        self.assertEqual(r['weights'], {'headup': .1, 'trunk': .8, 'shoulder': .1})
        r = self.measure(p, c, 30., weights={'headup': .3, 'trunk': .6, 'shoulder': .1})
        self.assertAlmostEqual(r['total'], .3 * 3 + .6 * (10 * 30 / 36.4628217436476) + .1 * (10 * 30 / 93.9849170623855), places=8)

    def test_frozen_config_can_be_supplied_explicitly(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/scoring.json').read_text())
        default = self.measure(pose_fixture(), balls(), 30.)
        explicit = self.measure(pose_fixture(), balls(), 30., calibration=config['calibration'], weights=config['weights'])
        self.assertEqual(default['metrics'], explicit['metrics'])

    def test_cli_full_config_and_public_calibration_preflight(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/scoring.json').read_text())
        validate = getattr(importlib.import_module('scoring.release_metrics'), 'validate_calibration', None)
        self.assertTrue(callable(validate), 'CLI needs public validate_calibration before inference')
        self.assertEqual(validate(config), config['calibration'])
        self.assertEqual(validate(config['calibration']), config['calibration'])
        before = copy.deepcopy(config)
        r = self.measure(pose_fixture(), balls(), 30., calibration=config)
        self.assertEqual(r['calibration'], config['calibration'])
        self.assertEqual(config, before)
        config['calibration']['headup']['max_deg'] = 0
        with self.assertRaises(ValueError):
            validate(config)

    def test_public_parameter_preflight_validates_weights_before_inference(self):
        module = importlib.import_module('scoring.release_metrics')
        validate = getattr(module, 'validate_parameters', None)
        self.assertTrue(callable(validate), 'CLI needs public validate_parameters(calibration, weights)')
        c, w = validate()
        self.assertEqual(c, module.validate_calibration())
        self.assertEqual(w, {'headup': .1, 'trunk': .8, 'shoulder': .1})
        with self.assertRaises(ValueError):
            validate(c, {'headup': .3, 'trunk': .3, 'shoulder': .3})
        c['headup']['min_deg'] = 0
        w['headup'] = 0
        clean_c, clean_w = validate()
        self.assertGreater(clean_c['headup']['min_deg'], 10)
        self.assertEqual(clean_w['headup'], .1)

    def test_common_support_rejects_sg21_cancellation_even_when_sg13_is_valid(self):
        # Thirty-degree steps pass native QA. SG21 cancels this periodic signal
        # (gain about .0087), while SG13's gain is about .7033.
        r = self.measure(pose_fixture(speed=900.), None, 30.)
        d = r['diagnostics']['shoulder']
        self.assertTrue(d['native_valid'].all())
        self.assertGreater(d['smooth_norms'][13][50], .7)
        self.assertLess(d['smooth_norms'][21][50], .01)
        self.assertFalse(d['support'].any())
        self.assertIsNone(r['metrics']['shoulder']['raw'])

    def test_long_ball_gaps_are_not_interpolated(self):
        c = balls()
        c['frames'] = [r for r in c['frames'] if r['frame_number'] not in (49, 50, 51, 52)]
        d = self.measure(pose_fixture(), c, 30.)['diagnostics']['headup']
        self.assertFalse(d['imputed'].any())
        self.assertTrue(np.isnan(d['ball'][49:53]).all())

    def test_trunk_and_shoulder_scores_clip_at_ten(self):
        r = self.measure(pose_fixture(speed=300., trunk_degrees=90.), balls(), 30.)
        self.assertEqual(r['metrics']['trunk']['score'], 10.)
        self.assertEqual(r['metrics']['shoulder']['score'], 10.)

    def test_invalid_weights_are_rejected_not_silently_normalized(self):
        for weights in ({'headup': -.1, 'trunk': 1., 'shoulder': .1},
                        {'headup': .3, 'trunk': .3, 'shoulder': .3},
                        {'headup': np.nan, 'trunk': .8, 'shoulder': .1},
                        {'headup': .1, 'trunk': .8, 'body': .1}, [1, 0, 0],
                        {'headup': True, 'trunk': 0., 'shoulder': 0.}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.measure(pose_fixture(), None, 30., weights=weights)

    def test_invalid_shapes_metadata_and_fps_raise_value_error(self):
        for key, value in (('n', True), ('n', 0), ('n', 180.), ('width', 0), ('height', np.inf),
                           ('lm', np.zeros((180, 32, 3))), ('world', np.zeros((179, 33, 3))),
                           ('vis', np.zeros((180, 33, 1)))):
            p = pose_fixture()
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.measure(p, None, 30.)
        for fps in (0, -1, np.inf, np.nan, True, '30'):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.measure(pose_fixture(), None, fps)

    def test_candidate_timeline_and_geometry_shapes_are_validated(self):
        for mutate in (lambda c: c['counts'].update(source_frames=179),
                       lambda c: c['frames'].append(copy.deepcopy(c['frames'][0])),
                       lambda c: c['frames'][0].update(frame_number=True),
                       lambda c: c['frames'][0].update(frame_number=180),
                       lambda c: c['frames'][0].update(ankles_global_xy=[1, 2]),
                       lambda c: c['frames'][0]['detections'][0].update(center_global_xy=[1, 2, 3])):
            c = balls()
            mutate(c)
            with self.assertRaises(ValueError):
                self.measure(pose_fixture(), c, 30.)

    def test_nonfinite_ball_candidates_are_missing_not_poisonous(self):
        c = balls()
        c['frames'][50]['detections'][0]['center_global_xy'][0] = np.nan
        d = self.measure(pose_fixture(), c, 30.)['diagnostics']['headup']
        self.assertFalse(d['observed'][50])
        self.assertNotIn(50, [e['frame'] for e in d['events']])

    def test_invalid_exclusion_indices_raise(self):
        for indices in ([-1], [180], [1.5], [True], np.zeros(179, bool), [[1, 2]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.measure(pose_fixture(), None, 30., excluded_frames=indices)

    def test_invalid_calibration_is_rejected(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/scoring.json').read_text())['calibration']
        for key, field, value in (('headup', 'max_deg', 0), ('headup', 'min_deg', np.nan),
                                  ('trunk', 'reference_flexion_deg', 0),
                                  ('shoulder', 'reference_speed_deg_s', np.inf)):
            calibration = copy.deepcopy(config)
            calibration[key][field] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.measure(pose_fixture(), None, 30., calibration=calibration)

    def test_absent_pose_is_na_with_auditable_reasons_and_quality_caveats(self):
        p = pose_fixture(1)
        p['world'][:] = p['lm'][:] = p['vis'][:] = np.nan
        r = self.measure(p, None, 30.)
        for key in ('headup', 'trunk', 'shoulder'):
            self.assertIsNone(r['metrics'][key]['raw'])
            self.assertIsNone(r['metrics'][key]['score'])
            self.assertTrue(r['metrics'][key]['reason'])
        self.assertIsNone(r['total'])
        warnings = ' '.join(r['quality']['warnings']).lower()
        for term in ('pixel', 'width', 'identity', 'provisional'):
            self.assertIn(term, warnings)


if __name__ == '__main__':
    unittest.main()
