#!/usr/bin/env python3
"""Current experimental evaluation from source-indexed pose/ROI caches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scoring.current_metrics import VERSION, DEFAULT_WEIGHTS, analyze_headup, body_alignment, load_pose, nfc, score_measurements, trunk_angle, head_timing_sensitivity

ROOT = Path(__file__).resolve().parent


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_index(folder, suffix):
    result = {}
    for path in Path(folder).glob('*' + suffix):
        key = nfc(path.name[:-len(suffix)])
        if key in result: raise ValueError(f'Duplicate NFC-normalized name: {key}')
        result[key] = path
    return result


def rank_correlation(x, y):
    return float(spearmanr(x, y).statistic) if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else None


def summarize(rows):
    rated = [r for r in rows if not r['excluded'] and r['grade'] is not None and r['total'] is not None]
    grades = np.array([r['grade'] for r in rated])
    x = np.array([[r['trunk_score'], r['head_score'], r['body_score']] for r in rated]).reshape(-1, 3)
    comparisons = []
    for label, weights in [('legacy_weights_current_inputs', [.5, .3, .2]), ('equal', [1/3] * 3), ('body_50pct', [.3, .2, .5])]:
        total = x @ weights
        pairs = [(i, j) for i in range(len(rated)) for j in range(i + 1, len(rated)) if grades[i] != grades[j]]
        agreement = np.mean([np.sign(grades[j] - grades[i]) == np.sign(total[j] - total[i]) for i, j in pairs]) if pairs else None
        deleted = [rank_correlation(np.delete(grades, i), np.delete(total, i)) for i in range(len(rated))]
        deleted = [v for v in deleted if v is not None]
        comparisons.append(dict(label=label, weights_trunk_head_body=weights, n=len(rated), rho=rank_correlation(grades, total),
                                pair_order_agreement=agreement, delete_one_video_rho_range=[min(deleted), max(deleted)] if deleted else None))
    sensitivity = []
    for window in (5, 9, 13):
        subset = [r for r in rated if r['body_windows'][str(window)]['M'] is not None]
        sensitivity.append(dict(window=window, n=len(subset), rho=rank_correlation([r['grade'] for r in subset], [r['body_windows'][str(window)]['M'] for r in subset])))
    return dict(weights=comparisons, body_smoothing=sensitivity,
                interpretation='Same-cohort exploratory comparisons, not independent validation or cross-validation.')


def evaluate(manifest, calibration, cache_dir, candidates_dir):
    if calibration.get('version') != VERSION:
        raise ValueError('Calibration version does not match the evaluator')
    poses, candidates = file_index(cache_dir, '_pose.npz'), file_index(candidates_dir, '_candidates.json')
    names = [nfc(c['name']) for c in manifest['clips']]
    if len(set(names)) != len(names): raise ValueError('Duplicate clips in manifest')
    rows = []
    for clip in manifest['clips']:
        name = nfc(clip['name'])
        row = dict(video=name, grade=clip['grade'], reference=clip['reference'], excluded=bool(clip.get('exclude_reason')), exclude_reason=clip.get('exclude_reason'))
        if row['excluded']:
            row['total'] = None
            rows.append(row)
            continue
        if name not in poses or name not in candidates: raise ValueError(f'Missing pose cache or ROI candidates for {name}')
        det = read_json(candidates[name])
        if nfc(det['clip']) != name: raise ValueError(f'Candidate clip name mismatch: {name}')
        digest = hashlib.sha256(poses[name].read_bytes()).hexdigest()
        if det.get('pose_sha256') != digest: raise ValueError(f'Candidate pose SHA-256 mismatch: {name}; rerun the ROI detector')
        n = det['counts']['source_frames']
        if clip.get('source_frames') is not None and clip['source_frames'] != n: raise ValueError(f'Manifest source frame count mismatch: {name}')
        pose = load_pose(poses[name], n)
        fps = det.get('source_average_fps')
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Missing/invalid source average FPS: {name}')
        warnings = []
        if not 29 <= fps <= 31:
            warnings.append('Frame-based windows were tested around 30 FPS; this FPS is outside that range.')
        if (pose['width'], pose['height']) != (1920, 1080):
            warnings.append('Fixed pixel thresholds were tested at 1920x1080; this resolution differs.')
        body = {str(w): body_alignment(pose, w) for w in (5, 9, 13)}
        head = analyze_headup(pose, det)
        row.update(head_raw=head['headup'], trunk_raw=trunk_angle(pose), body_M=body['9']['M'])
        row.update(score_measurements(row, calibration))
        row.update(source_frames=n, pose_sha256=digest, candidates_sha256=hashlib.sha256(candidates[name].read_bytes()).hexdigest(),
                   events=[e['frame'] for e in head['events']], candidate_count=head['candidate_count'],
                   accepted_events=head['events'], rejected_events=head['rejected'],
                   observed_ball_fraction=head['observed_fraction'], imputed_ball_fraction=head['imputed_fraction'],
                   body_valid_frames=body['9']['valid_frames'],
                   body_windows={w: {'M': b['M'], 'valid_frames': b['valid_frames']} for w, b in body.items()})
        timing = head_timing_sensitivity(head)
        row.update(head_raw_shift_minus2_to_plus2=timing['means'], head_shift_common_event_count=timing['common_event_count'],
                   source_average_fps=fps, frame_width=pose['width'], frame_height=pose['height'],
                   detector_parameters=det.get('parameters'), detector_versions=det.get('versions'),
                   model_sha256=det.get('model_sha256'), warnings=warnings)
        if body['9']['valid_frames'] < .5 * n:
            warnings.append('Under half the source frames have body-filter support; inspect duration and coverage.')
        rows.append(row)
    return json_safe(dict(version=VERSION, weights=DEFAULT_WEIGHTS, provisional=True, calibration=calibration,
                          cohort=manifest['description'], rows=rows, comparisons=summarize(rows)))


def verify_snapshot(report, expected):
    """The snapshot is a regression target, never the measurement input."""
    if report['version'] != expected['version']: raise ValueError('Snapshot version differs')
    for obj in (report, expected):
        names = [r['video'] for r in obj['rows']]
        if len(set(names)) != len(names): raise ValueError('Duplicate snapshot row names')
    actual_by, expected_by = ({r['video']: r for r in obj['rows']} for obj in (report, expected))
    if set(actual_by) != set(expected_by): raise ValueError('Snapshot cohort differs')
    for name, want in expected_by.items():
        got = actual_by[name]
        for key in ('grade', 'reference', 'excluded', 'exclude_reason'):
            if got[key] != want[key]: raise ValueError(f'Snapshot metadata differs: {name} {key}')
        if got['excluded']: continue
        for key in ('head_raw', 'trunk_raw', 'body_M', 'head_score', 'trunk_score', 'body_score', 'total'):
            if got[key] is None and want[key] is None: continue
            if got[key] is None or want[key] is None or not np.isclose(got[key], want[key], rtol=0, atol=1e-8):
                raise ValueError(f'Snapshot mismatch: {name} {key}: {got[key]} != {want[key]}')
        if got['events'] != want['events'] or got['candidate_count'] != want['candidate_count']: raise ValueError(f'Touch candidate mismatch: {name}')


def markdown(report):
    lines = ['# 인,인 세 지표와 총점', '', '세 지표 각각 1/3. 현재 코호트의 탐색 결과이며 전문가 등급 예측 성능이 아닙니다.', '',
             '| 영상 | 헤드업 점수 (°) | 상체각도 점수 (°) | 어깨·골반 점수 (M) | 총점 | 채택/후보 |', '|---|---:|---:|---:|---:|---:|']
    def number(value, digits=2): return 'N/A' if value is None else f'{value:.{digits}f}'
    for r in report['rows']:
        if r['excluded']:
            lines.append(f"| {r['video']} | — | — | — | 제외 | {r['exclude_reason']} |")
            continue
        lines.append(f"| {r['video']} | {number(r['head_score'])} ({number(r['head_raw'])}°) | "
                     f"{number(r['trunk_score'])} ({number(r['trunk_raw'])}°) | {number(r['body_score'])} ({number(r['body_M'], 5)}) | "
                     f"{number(r['total'])} | {len(r['events'])}/{r['candidate_count']} |")
    lines.extend(['', '0점은 환산식의 하한이며 동작이 없다는 뜻이 아닙니다. M은 각도가 아닌 무차원 상대 정렬 변화량입니다.',
                  '방향 전환 후보는 접촉 시점의 대리값입니다. 세 지표 중 하나라도 계산 불가이면 총점은 N/A입니다.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/2026-09-20/manifest.json')
    parser.add_argument('--calibration', type=Path, default=ROOT / 'experiments/2026-09-20/calibration.json')
    parser.add_argument('--cache-dir', type=Path, default=Path('output/scoring_cache'))
    parser.add_argument('--candidates-dir', type=Path, default=Path('output/current/roi_candidates'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/current/evaluation'))
    parser.add_argument('--verify-snapshot', type=Path, help='Compare independent calculations with an archived result')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outputs = [args.output_dir / 'results.json', args.output_dir / 'scores.md']
    if not args.overwrite and any(p.exists() for p in outputs): parser.error('Output exists; choose another output directory or pass --overwrite')
    try:
        report = evaluate(read_json(args.manifest), read_json(args.calibration), args.cache_dir, args.candidates_dir)
        if args.verify_snapshot: verify_snapshot(report, read_json(args.verify_snapshot))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    outputs[1].write_text(markdown(report), encoding='utf-8')
    print(markdown(report))
    print('Snapshot verified.' if args.verify_snapshot else 'Evaluation complete (provisional).')
    print(f'Results: {outputs[0]}')


if __name__ == '__main__':
    main()
