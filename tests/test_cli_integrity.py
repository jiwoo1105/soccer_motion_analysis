"""A report must still correspond to its source at publication time."""
from pathlib import Path
from unittest.mock import patch

import pytest

import analyze_video
from scoring.video_pipeline import sha256


@pytest.mark.parametrize('replace_source', [False, True])
def test_source_replacement_during_render_never_publishes(tmp_path, replace_source):
    source = tmp_path / 'source.mov'
    source.write_bytes(b'original')
    destination = tmp_path / 'result'
    args = analyze_video.parser().parse_args([
        '--video', str(source), '--output', str(destination),
        '--reuse-cache', str(tmp_path / 'cache'),
    ])
    meta = dict(name=source.name, fps=30., video_sha256=sha256(source))
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
