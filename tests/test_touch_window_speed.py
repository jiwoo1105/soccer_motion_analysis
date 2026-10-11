"""Touch-window selection keeps the original source-frame pair clock."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scoring import release_metrics as metrics
from test_release_metrics import balls, pose_fixture


def measurement(halfwidth=24):
    return {'shoulder': {'aggregation': 'touch_window_mean',
                         'window_halfwidth_frames': halfwidth}}


def profile(halfwidth=24):
    result = json.loads((Path(__file__).resolve().parents[1] / 'configs/scoring.json').read_text())
    result.update(schema_version=2, version='2026-10-11-touch-window',
                  measurement=measurement(halfwidth))
    return result


def summarize(speed, centers, halfwidth=24, **kwargs):
    helper = getattr(metrics, 'summarize_touch_window_speed', None)
    assert callable(helper), 'Need a pure helper for replaying known centers and original pair speeds'
    return helper(speed, centers, window_halfwidth_frames=halfwidth, **kwargs)


@pytest.mark.parametrize('centers,halfwidth,indices,mean', [
    ([3, 5, 3], 2, [2, 3, 4, 5, 6, 7], 4.5),  # Overlap and duplicate count once.
    ([1, 4], 1, [1, 2, 4, 5], 3.),  # (2,3) spans DIFFERENT adjacent windows.
    ([0, 9], 2, [1, 2, 8, 9], 5.),  # Clipped edges retain their valid pairs.
    ([], 2, [], None),
])
def test_union_counts_original_pairs_once_and_requires_same_window(centers, halfwidth, indices, mean):
    speed = np.arange(10, dtype=float)
    speed[0] = np.nan
    before = speed.copy()
    result = summarize(speed, centers, halfwidth)
    np.testing.assert_array_equal(np.flatnonzero(result['selected_pair_support']), indices)
    assert result['selected_pairs'] == len(indices)
    assert result['selected_mean_deg_s'] == mean
    assert result['whole_pairs'] == 9
    assert result['whole_mean_deg_s'] == 5.
    assert result['window_halfwidth_frames'] == halfwidth
    np.testing.assert_array_equal(speed, before)


def test_gaps_and_partial_windows_use_only_finite_original_pairs():
    # Index i is already the speed of pair (i-1,i), never a compacted frame list.
    result = summarize([np.nan, 10., 20., np.nan, np.nan, 50., 60., 70.], [3], 3)
    np.testing.assert_array_equal(np.flatnonzero(result['selected_pair_support']), [1, 2, 5, 6])
    assert result['selected_mean_deg_s'] == 35.
    assert result['selected_pairs'] == 4
    assert result['whole_pairs'] == 5
    assert result['whole_mean_deg_s'] == 42.


def test_union_mean_weights_pairs_instead_of_averaging_window_means():
    result = summarize([np.nan, 10., 10., np.nan, 90., np.nan], [1, 4], 1)
    assert result['selected_pairs'] == 3
    assert result['selected_mean_deg_s'] == pytest.approx(110 / 3)


def test_explicit_pair_support_and_nonfinite_speeds_remain_excluded():
    support = np.array([True, True, False, True, True, True])
    result = summarize([999., 10., 200., np.inf, np.nan, 50.], [2], 5, pair_support=support)
    np.testing.assert_array_equal(np.flatnonzero(result['selected_pair_support']), [1, 5])
    assert result['whole_pairs'] == result['selected_pairs'] == 2
    assert result['selected_mean_deg_s'] == 30.
    np.testing.assert_array_equal(support, [True, True, False, True, True, True])


def test_window_with_no_valid_pairs_is_na_despite_whole_video_support():
    result = summarize([np.nan, 10., np.nan, np.nan, np.nan, 50.], [3], 1)
    assert result['selected_pairs'] == 0
    assert result['selected_mean_deg_s'] is None
    assert result['whole_pairs'] == 2
    assert result['whole_mean_deg_s'] == 30.


@pytest.mark.parametrize('halfwidth', [0, -1, 1.5, True, '24', np.nan, None])
def test_invalid_window_width_is_rejected(halfwidth):
    with pytest.raises(ValueError, match='window_halfwidth_frames'):
        summarize([np.nan, 1.], [1], halfwidth)


@pytest.mark.parametrize('centers', [[-1], [3], [1.5], [True], [[1]]])
def test_invalid_source_centers_are_rejected(centers):
    with pytest.raises(ValueError, match='centers'):
        summarize([np.nan, 1., 2.], centers)


def test_no_ball_is_na_but_whole_speed_and_quality_support_are_preserved():
    result = metrics.measure_video(pose_fixture(), None, 30.)
    assert result['metrics']['shoulder']['raw'] is None
    assert result['metrics']['shoulder']['score'] is None
    assert 'touch' in result['metrics']['shoulder']['reason'].lower()
    assert result['metrics']['shoulder']['valid_pairs'] == 0
    diag = result['diagnostics']['shoulder']
    assert diag['touch_events'] == []
    assert not diag['selected_pair_support'].any()
    assert diag['whole_pairs'] == 157
    assert diag['whole_mean_deg_s'] == pytest.approx(30.)
    np.testing.assert_allclose(diag['geo_speed'][12:169], 30., atol=1e-8)


@pytest.mark.parametrize('missing', ['eyes', 'inner_head_window', 'outer_head_window'])
def test_head_invalid_proxies_still_select_shoulder_windows(missing):
    p = pose_fixture()
    if missing == 'eyes':
        p['world'][:, [2, 5]] = np.nan
        p['vis'][:, [2, 5]] = 0.
    else:
        p['vis'][25 if missing == 'inner_head_window' else 9, 2] = 0.
    result = metrics.measure_video(p, balls(), 30.)
    head, shoulder = result['diagnostics']['headup'], result['diagnostics']['shoulder']
    assert 25 not in [event['frame'] for event in head['events']]
    assert [event['frame'] for event in shoulder['touch_events']] == [25, 50, 75, 100, 125, 150]
    assert all(not event['reasons'] for event in shoulder['touch_events'])
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30.)
    if missing != 'outer_head_window':
        event = next(event for event in shoulder['touch_events'] if event['frame'] == 25)
        assert event['legacy_reasons'] == ['head_pose_invalid']


@pytest.mark.parametrize('failure,reason', [('imputed', 'imputed_center'),
                                         ('ankle', 'ankle_visibility'),
                                         ('distant', 'far_from_ankle'),
                                         ('ball_gap', 'ball_support_gap'),
                                         ('ball_jump', 'ball_jump_in_support')])
def test_head_only_rejection_is_removed_but_non_head_failures_are_retained(failure, reason):
    p, c = pose_fixture(), balls()
    p['vis'][:, [2, 5]] = 0.
    if failure == 'imputed':
        c['frames'] = [r for r in c['frames'] if r['frame_number'] not in (49, 50)]
    elif failure == 'ankle':
        p['vis'][50, [27, 28]] = 0.
    elif failure == 'distant':
        p['lm'][50, [27, 28], :2] = [0., 0.]
    elif failure == 'ball_gap':
        c['frames'] = [r for r in c['frames'] if r['frame_number'] not in (38, 39, 40, 41)]
    else:
        c['frames'][38]['detections'][0]['center_global_xy'][0] -= 90.
    diag = metrics.measure_video(p, c, 30.)['diagnostics']['shoulder']
    assert 50 not in [event['frame'] for event in diag['touch_events']]
    event = next(event for event in diag['touch_rejected'] if event['frame'] == 50)
    assert reason in event['reasons']
    assert 'head_pose_invalid' not in event['reasons']


def test_all_ball_proxies_rejected_makes_shoulder_na():
    p = pose_fixture()
    p['vis'][:, [27, 28]] = 0.
    result = metrics.measure_video(p, balls(), 30.)
    assert result['metrics']['shoulder']['raw'] is None
    assert result['diagnostics']['shoulder']['whole_pairs'] == 157
    assert len(result['diagnostics']['shoulder']['touch_rejected']) == 6


def test_short_clip_keeps_ball_proxy_with_partial_shoulder_window_at_source_edge():
    result = metrics.measure_video(pose_fixture(40), balls(40, phase=15.), 30.)
    assert result['metrics']['headup']['raw'] is None
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30.)
    diag = result['diagnostics']['shoulder']
    assert [event['frame'] for event in diag['touch_events']] == [15]
    np.testing.assert_array_equal(np.flatnonzero(diag['selected_pair_support']), np.arange(12, 29))


def test_partial_shoulder_support_is_usable_without_bridging_a_missing_frame():
    p = pose_fixture(100)
    p['vis'][50, 11] = 0.
    result = metrics.measure_video(p, balls(100), 30.)
    diag = result['diagnostics']['shoulder']
    assert [event['frame'] for event in diag['touch_events']] == [25, 50, 75]
    assert not diag['selected_pair_support'][39:63].any()
    # Runs [0,49] and [51,99] leave 28 and 27 supported frames: 27+26 pairs.
    assert diag['selected_pairs'] == 53
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30.)


@pytest.mark.parametrize('fps', [15., 30., 60.])
def test_custom_window_keeps_constant_physical_speed_at_each_fps(fps):
    result = metrics.measure_video(pose_fixture(speed=30., fps=fps), balls(), fps,
                                   measurement=measurement(2))
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30., abs=1e-8)
    assert result['metrics']['shoulder']['valid_pairs'] == 24
    assert result['metrics']['shoulder']['valid_frames'] == 30
    assert result['metrics']['shoulder']['score'] == pytest.approx(10 * 30 / 96.65652026544726)


def test_custom_window_changes_only_shoulder_aggregation_and_preserves_whole_speed():
    p = pose_fixture()
    phase = np.deg2rad(np.arange(180) ** 2 / 180.)
    direction = np.column_stack((np.cos(phase), np.zeros(180), np.sin(phase)))
    p['world'][:, 11], p['world'][:, 12] = -.2 * direction, .2 * direction
    whole = metrics.measure_video(p, balls(), 30.)
    narrow = metrics.measure_video(p, balls(), 30., measurement=measurement(2))
    assert whole['measurement'] == measurement(16)
    assert narrow['measurement'] == measurement(2)
    assert narrow['version'] == '2026-10-11-sam2-touch16'
    for key in ('headup', 'trunk'):
        assert narrow['metrics'][key] == whole['metrics'][key]
    a, b = whole['diagnostics']['shoulder'], narrow['diagnostics']['shoulder']
    for key in ('geo_speed', 'native_valid', 'auto_excluded', 'support', 'pair_support'):
        np.testing.assert_array_equal(a[key], b[key])
    assert narrow['metrics']['shoulder']['raw'] != pytest.approx(whole['metrics']['shoulder']['raw'])
    selected = np.array([i for center in (25, 50, 75, 100, 125, 150) for i in range(center - 1, center + 3)])
    assert narrow['metrics']['shoulder']['raw'] == pytest.approx(a['geo_speed'][selected].mean())
    assert b['selected_pairs'] == 24
    assert b['whole_pairs'] == 157


@pytest.mark.parametrize('settings', [{}, {'shoulder': {}},
    {'shoulder': {'aggregation': 'whole_video_mean', 'window_halfwidth_frames': 24}},
    {'shoulder': {'aggregation': 'touch_window_mean', 'window_halfwidth_frames': 0}},
    {'shoulder': {'aggregation': 'touch_window_mean', 'window_halfwidth_frames': True}},
    {'shoulder': {'aggregation': 'touch_window_mean', 'window_halfwidth_frames': 2.5}},
    {'shoulder': {'aggregation': 'touch_window_mean', 'window_halfwidth_frames': 24, 'extra': 1}},
])
def test_measurement_config_is_validated(settings):
    with pytest.raises(ValueError, match='measurement|window_halfwidth_frames'):
        metrics.measure_video(pose_fixture(), balls(), 30., measurement=settings)


def test_full_profile_honors_its_measurement_and_rejects_conflicting_explicit_settings():
    config = profile(2)
    before = copy.deepcopy(config)
    result = metrics.measure_video(pose_fixture(), balls(), 30., calibration=config)
    assert result['metrics']['shoulder']['valid_pairs'] == 24
    assert result['measurement'] == measurement(2)
    assert config == before
    with pytest.raises(ValueError, match='measurement.*match|measurement.*conflict'):
        metrics.measure_video(pose_fixture(), balls(), 30., calibration=config, measurement=measurement(3))


@pytest.mark.parametrize('legacy', ['schema', 'missing_measurement', 'whole_video'])
def test_old_whole_video_profiles_are_not_silently_accepted(legacy):
    config = profile()
    if legacy == 'schema':
        config['schema_version'] = 1
    elif legacy == 'missing_measurement':
        del config['measurement']
    else:
        config['measurement']['shoulder']['aggregation'] = 'whole_video_mean'
    with pytest.raises(ValueError, match='schema|measurement|aggregation'):
        metrics.validate_calibration(config)
    with pytest.raises(ValueError, match='schema|measurement|aggregation'):
        metrics.measure_video(pose_fixture(), balls(), 30., calibration=config)


def test_explicit_touch_frames_work_without_roi_ball_head_or_ankle_support():
    p = pose_fixture()
    p['world'][:, [2, 5]] = np.nan
    p['vis'][:, [2, 5, 27, 28]] = 0.
    result = metrics.measure_video(p, None, 30., touch_frames=[60], measurement=measurement(24))
    diag = result['diagnostics']['shoulder']
    assert diag['touch_event_source'] == 'explicit'
    assert [event['frame'] for event in diag['touch_events']] == [60]
    assert (diag['touch_events'][0]['start'], diag['touch_events'][0]['end']) == (36, 84)
    assert diag['touch_rejected'] == []
    np.testing.assert_array_equal(np.flatnonzero(diag['selected_pair_support']), np.arange(37, 85))
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30.)
    assert result['metrics']['headup']['raw'] is None


def test_explicit_empty_touch_frames_never_fall_back_to_automatic_events():
    result = metrics.measure_video(pose_fixture(), balls(), 30., touch_frames=[])
    assert result['metrics']['headup']['valid_events'] == 6
    assert result['metrics']['shoulder']['raw'] is None
    assert result['metrics']['shoulder']['score'] is None
    assert result['metrics']['shoulder']['valid_pairs'] == 0
    assert 'explicit' in result['metrics']['shoulder']['reason'].lower()
    diag = result['diagnostics']['shoulder']
    assert diag['touch_event_source'] == 'explicit'
    assert diag['touch_events'] == diag['touch_rejected'] == []
    assert diag['whole_pairs'] == 157
    assert not diag['selected_pair_support'].any()


def test_none_touch_frames_preserve_default_automatic_proxy_behavior():
    default = metrics.measure_video(pose_fixture(), balls(), 30.)
    explicit_none = metrics.measure_video(pose_fixture(), balls(), 30., touch_frames=None)
    assert explicit_none['metrics'] == default['metrics']
    diag = explicit_none['diagnostics']['shoulder']
    assert diag['touch_event_source'] == 'automatic_proxy'
    assert [event['frame'] for event in diag['touch_events']] == [25, 50, 75, 100, 125, 150]
    np.testing.assert_array_equal(diag['selected_pair_support'],
                                  default['diagnostics']['shoulder']['selected_pair_support'])


def test_explicit_centers_replace_automatic_centers_without_changing_other_measurements():
    p, c = pose_fixture(), balls()
    before_p, before_c = copy.deepcopy(p), copy.deepcopy(c)
    automatic = metrics.measure_video(p, c, 30.)
    frames = np.array([61, 60, 60], dtype=np.int64)
    result = metrics.measure_video(p, c, 30., touch_frames=frames, measurement=measurement(2))
    for key in ('headup', 'trunk'):
        assert result['metrics'][key] == automatic['metrics'][key]
        for field, value in automatic['diagnostics'][key].items():
            np.testing.assert_equal(result['diagnostics'][key][field], value)
    diag = result['diagnostics']['shoulder']
    assert [event['frame'] for event in diag['touch_events']] == [60, 61]
    assert diag['counts']['accepted_touch_events'] == 2
    np.testing.assert_array_equal(np.flatnonzero(diag['selected_pair_support']), [59, 60, 61, 62, 63])
    assert diag['selected_pairs'] == 5
    for field in ('geo_speed', 'support', 'pair_support', 'auto_excluded', 'native_valid'):
        np.testing.assert_array_equal(diag[field], automatic['diagnostics']['shoulder'][field])
    np.testing.assert_array_equal(frames, [61, 60, 60])
    for key in ('lm', 'world', 'vis'):
        np.testing.assert_array_equal(p[key], before_p[key])
    assert c == before_c


def test_explicit_centers_keep_source_edges_and_excluded_frame_gaps():
    result = metrics.measure_video(pose_fixture(100), None, 30.,
                                   touch_frames=[0, 50, 99], excluded_frames=[50], measurement=measurement(24))
    diag = result['diagnostics']['shoulder']
    assert [event['frame'] for event in diag['touch_events']] == [0, 50, 99]
    # Each disjoint window uses only its own valid pairs; exclusion guards remain.
    expected = [*range(12, 25), *range(27, 39), *range(63, 75), *range(76, 89)]
    np.testing.assert_array_equal(np.flatnonzero(diag['selected_pair_support']), expected)
    assert result['metrics']['shoulder']['raw'] == pytest.approx(30.)


@pytest.mark.parametrize('frames', [[-1], [180], [2.5], [2.0], [True], [1, False],
                                  [[25]], 25, '25', {'frame': 25}, [np.nan], [np.inf],
                                  np.zeros(180, dtype=bool)])
def test_invalid_explicit_touch_frames_are_rejected(frames):
    with pytest.raises(ValueError, match='touch_frames'):
        metrics.measure_video(pose_fixture(), balls(), 30., touch_frames=frames)
