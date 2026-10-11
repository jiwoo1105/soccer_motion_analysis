"""SAM2 tracks drive both event-conditioned metrics on the original clock."""
import numpy as np
import pytest

from test_release_metrics import pose_fixture
from scoring.release_metrics import measure_video
from scoring.sam2_metrics import detect_sam2_touches


def tracks(n=180):
    return {i: [320 + 50*np.cos(2*np.pi*i/50), 336., 12.] for i in range(n)}


def test_original_turning_rule_has_known_source_centers():
    result = detect_sam2_touches(pose_fixture(), tracks())
    assert [e['frame'] for e in result['events']] == [25, 50, 75, 100, 125, 150]


def test_sam2_centers_drive_head_and_shoulder_without_roi_candidates():
    r = measure_video(pose_fixture(), None, 30., ball_tracks=tracks())
    assert r['diagnostics']['shoulder']['touch_event_source'] == 'yolov8_sam2'
    assert r['metrics']['headup']['raw'] == pytest.approx(8., abs=1e-7)
    assert r['metrics']['shoulder']['raw'] == pytest.approx(30., abs=1e-7)
    assert r['metrics']['trunk']['raw'] == pytest.approx(150.)
    assert r['diagnostics']['headup']['event_source'] == 'yolov8_sam2'


def test_missing_eye_does_not_discard_shoulder_events():
    p=pose_fixture();p['vis'][:,2]=0.
    r=measure_video(p,None,30.,ball_tracks=tracks())
    assert r['metrics']['headup']['raw'] is None
    assert r['metrics']['shoulder']['raw'] == pytest.approx(30.,abs=1e-7)
    assert r['total'] is None


def test_empty_sam2_tracks_do_not_fall_back_to_other_touch_source():
    r=measure_video(pose_fixture(),None,30.,ball_tracks={})
    assert r['metrics']['headup']['raw'] is None
    assert r['metrics']['shoulder']['raw'] is None


def test_gapped_track_retains_source_frame_indices_and_flags_head_window():
    t=tracks();del t[47]
    r=measure_video(pose_fixture(),None,30.,ball_tracks=t)
    assert not r['diagnostics']['headup']['observed'][47]
    assert any('ball_support_gap' in e['reasons'] for e in r['diagnostics']['headup']['rejected'])
    assert len(r['diagnostics']['headup']['ball']) == 180


def test_conflicting_ball_sources_are_rejected():
    with pytest.raises(ValueError,match='one touch source'):
        measure_video(pose_fixture(),{'counts':{'source_frames':180},'frames':[]},30.,ball_tracks=tracks())
    with pytest.raises(ValueError,match='one touch source'):
        measure_video(pose_fixture(),None,30.,ball_tracks=tracks(),touch_frames=[25])


@pytest.mark.parametrize('bad',[{-1:[1,2,3]},{180:[1,2,3]},{True:[1,2,3]},{'01':[1,2,3]},{0:[1,2]},{0:[1,2,-1]}])
def test_malformed_tracks_fail_before_scoring(bad):
    with pytest.raises(ValueError):
        detect_sam2_touches(pose_fixture(),bad)
