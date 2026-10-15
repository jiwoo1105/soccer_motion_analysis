"""A report must still correspond to its source at publication time."""
from pathlib import Path
import json
from unittest.mock import patch

import pytest

import analyze_video
from scoring.video_pipeline import sha256
from test_release_metrics import balls, pose_fixture
from test_touch_window_speed import measurement, profile


@pytest.mark.parametrize('replace_source', [False, True])
def test_source_replacement_during_render_never_publishes(tmp_path, replace_source):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    destination = tmp_path / 'result'
    args = analyze_video.parser().parse_args([
        '--video', str(source), '--output', str(destination),
        '--reuse-cache', str(tmp_path / 'cache'), '--ball-backend', 'yolo11_roi',
    ])
    meta = dict(name=source.name, fps=30., video_sha256=sha256(source), ball_source='yolo11_roi')
    report = dict(metrics={k: dict(raw=1., score=5., reason=None)
                           for k in ('headup', 'trunk', 'shoulder')}, total=5.)

    def render(*unused, **kwargs):
        if replace_source:
            source.write_bytes(b'replaced')

    with patch('scoring.video_pipeline.load_cached_clip', return_value=({}, {}, meta)), \
         patch('scoring.release_metrics.measure_video', return_value=report), \
         patch('scoring.panorama_report.render_panorama', side_effect=render), \
         patch('scoring.panorama_report.write_html_report'):
        if replace_source:
            with pytest.raises(ValueError, match='Source video changed'):
                analyze_video.run(args)
            assert not destination.exists()
        else:
            analyze_video.run(args)
            assert (destination / 'report.json').is_file()
    assert not list(tmp_path.glob('.dribble-report-*'))


@pytest.mark.parametrize('invalid', ['legacy_schema', 'missing_measurement', 'whole_video',
                                    'zero', 'negative', 'fractional', 'boolean', 'unknown_field'])
def test_invalid_measurement_profile_fails_before_inference_or_outputs(tmp_path, invalid):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    config = profile()
    settings = config['measurement']['shoulder']
    if invalid == 'legacy_schema':
        config['schema_version'] = 1
        del config['measurement']
    elif invalid == 'missing_measurement':
        del config['measurement']
    elif invalid == 'whole_video':
        settings['aggregation'] = 'whole_video_mean'
    elif invalid == 'unknown_field':
        settings['window_halfwidth'] = 8
    else:
        settings['window_halfwidth_frames'] = dict(zero=0, negative=-1, fractional=2.5, boolean=True)[invalid]
    calibration = tmp_path / 'scoring.json'
    calibration.write_text(json.dumps(config))
    output = tmp_path / 'new-parent' / 'result'
    with patch('scoring.video_pipeline.load_cached_clip', side_effect=AssertionError('invalid profile reached cache')):
        status = analyze_video.main(['--video', str(source), '--output', str(output),
                                     '--calibration', str(calibration), '--reuse-cache', str(tmp_path / 'cache')])
    assert status == 1
    assert not output.parent.exists()
    assert not list(tmp_path.rglob('.dribble-report-*'))


def test_cli_passes_profile_window_to_measurement_and_publishes_actual_settings(tmp_path):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    calibration = tmp_path / 'scoring.json'
    calibration.write_text(json.dumps(profile(2)))
    output = tmp_path / 'result'
    meta = dict(name=source.name, fps=30., video_sha256=sha256(source), ball_source='yolo11_roi')
    # Only video/cache and rendering are external; scoring and JSON output are real.
    with patch('scoring.video_pipeline.load_cached_clip', return_value=(pose_fixture(), balls(), meta)), \
         patch('scoring.panorama_report.render_panorama'), \
         patch('scoring.panorama_report.write_html_report'):
        status = analyze_video.main(['--video', str(source), '--output', str(output),
                                     '--calibration', str(calibration), '--reuse-cache', str(tmp_path / 'cache'),
                                     '--ball-backend', 'yolo11_roi'])
    assert status == 0
    report = json.loads((output / 'report.json').read_text())
    clip = report['clips'][0]
    assert report['measurement'] == clip['measurement'] == measurement(2)
    assert report['version'] == clip['version'] == '2026-10-11-sam2-touch16'
    assert clip['metrics']['shoulder']['valid_pairs'] == 24
    assert clip['diagnostics']['shoulder']['whole_pairs'] == 157
    assert clip['scoring_profile']['sha256'] == sha256(calibration)


def test_default_cli_routes_sam2_raw_tracks_to_scoring(tmp_path, capsys):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    output = tmp_path / 'result'
    meta = dict(name=source.name, fps=30., video_sha256=sha256(source), ball_source='yolov8_sam2')
    tracks = {r['frame_number']: (*r['detections'][0]['center_global_xy'], 20.) for r in balls()['frames']}
    def cached(bundle, video, roi, *, ball_backend, **kwargs):
        assert ball_backend == 'yolov8_sam2'
        return pose_fixture(), tracks, meta
    with patch('scoring.video_pipeline.load_cached_clip', side_effect=cached), \
         patch('scoring.panorama_report.render_panorama') as renderer, patch('scoring.panorama_report.write_html_report'):
        status = analyze_video.main(['--video', str(source), '--output', str(output),
                                     '--reuse-cache', str(tmp_path / 'cache'), '--report-mode', 'paper'])
    assert status == 0
    report = json.loads((output / 'report.json').read_text())['clips'][0]
    assert report['video']['ball_source'] == 'yolov8_sam2'
    assert report['diagnostics']['shoulder']['touch_event_source'] == 'yolov8_sam2'
    assert renderer.call_args.kwargs['report_mode'] == 'paper'
    assert (output / 'paper_metrics.csv').is_file()
    console = capsys.readouterr().out
    assert all(symbol in console for symbol in ('S_trunk:', 'S_head:', 'V_touch:'))


def test_missing_local_sam_checkpoint_fails_before_loading_yolo_or_creating_output(tmp_path):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    yolo = tmp_path / 'yolov8n.pt'
    yolo.write_bytes(b'local fixture')
    output = tmp_path / 'new-parent' / 'result'
    with patch.dict('sys.modules', {'ultralytics': None, 'sam2': None}):
        status = analyze_video.main(['--video', str(source), '--output', str(output),
                                     '--model', str(yolo), '--sam2-checkpoint', str(tmp_path / 'missing.pt')])
    assert status == 1
    assert not output.parent.exists()


def test_cache_backend_mismatch_does_not_create_output_parent(tmp_path):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    output = tmp_path / 'new-parent' / 'result'
    with patch('scoring.video_pipeline.load_cached_clip', side_effect=ValueError('Cache backend differs')):
        status = analyze_video.main(['--video', str(source), '--output', str(output),
                                     '--reuse-cache', str(tmp_path / 'cache')])
    assert status == 1
    assert not output.parent.exists()
