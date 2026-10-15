"""Paper symbols and unit-bearing exports of unconverted measurements."""
import csv
import math
from pathlib import Path


PAPER_SYMBOLS = {'trunk': 'S_trunk', 'headup': 'S_head', 'shoulder': 'V_touch'}
PAPER_COLUMNS = {'trunk': 'S_trunk_deg', 'headup': 'S_head_deg',
                 'shoulder': 'V_touch_deg_s'}


def measurement_text(value, decimals=2):
    return 'N/A' if value is None or not math.isfinite(value) else f'{value:.{decimals}f}'


def write_paper_csv(reports, output_path):
    """Keep raw precision and N/A reasons; never derive values from scores."""
    fields = ['video', *PAPER_COLUMNS.values(), 'ball_source',
              'shoulder_halfwidth_frames', 'head_valid_events',
              'trunk_valid_frames', 'shoulder_valid_pairs',
              'trunk_reason', 'headup_reason', 'shoulder_reason']
    path = Path(output_path)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for report in reports:
            metrics = report['metrics']
            row = dict(video=report['video']['name'],
                       ball_source=report['video'].get('ball_source', ''),
                       shoulder_halfwidth_frames=report.get('measurement', {}).get(
                           'shoulder', {}).get('window_halfwidth_frames', ''),
                       head_valid_events=metrics['headup'].get('valid_events', ''),
                       trunk_valid_frames=metrics['trunk'].get('valid_frames', ''),
                       shoulder_valid_pairs=metrics['shoulder'].get('valid_pairs', ''))
            for key, column in PAPER_COLUMNS.items():
                raw = metrics[key]['raw']
                row[column] = raw if raw is not None and math.isfinite(raw) else 'N/A'
                row[key + '_reason'] = metrics[key].get('reason') or ''
            writer.writerow(row)
    return path
