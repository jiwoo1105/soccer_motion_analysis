"""Paper values must be the raw measurements, independent of score calibration."""
import csv
import copy

import pytest

from scoring.release_metrics import measure_video
from test_release_metrics import pose_fixture
from test_sam2_metrics import tracks


def test_paper_csv_uses_raw_units_and_preserves_missing_measurements(tmp_path):
    from scoring.paper_report import write_paper_csv
    report = measure_video(pose_fixture(), None, 30., ball_tracks=tracks())
    report['video'] = dict(name='sample.mov', ball_source='yolov8_sam2')
    # Changing a displayed score must never change the paper measurement.
    for metric in report['metrics'].values():
        metric['score'] = 1.234
    missing = copy.deepcopy(report)
    missing['video']['name'] = 'missing.mov'
    missing['metrics']['shoulder'].update(raw=None, reason='No touch')
    path = tmp_path / 'paper.csv'
    write_paper_csv([report, missing], path)
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    assert float(rows[0]['S_trunk_deg']) == pytest.approx(150.)
    assert float(rows[0]['S_head_deg']) == pytest.approx(8.)
    assert float(rows[0]['V_touch_deg_s']) == pytest.approx(30.)
    assert rows[0]['shoulder_halfwidth_frames'] == '16'
    assert rows[0]['ball_source'] == 'yolov8_sam2'
    assert rows[1]['V_touch_deg_s'] == 'N/A'
    assert rows[1]['shoulder_reason'] == 'No touch'
    assert report['metrics']['trunk']['symbol'] == 'S_trunk'
    assert report['metrics']['headup']['symbol'] == 'S_head'
    assert report['metrics']['shoulder']['symbol'] == 'V_touch'
