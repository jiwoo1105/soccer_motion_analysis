#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original videos -> headup/trunk/shoulder scores + panorama PNG/HTML/JSON."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',nargs='+',required=True,type=Path,help='One or more original video files')
    p.add_argument('--output',type=Path,required=True,help='New output directory (never overwritten)')
    p.add_argument('--model',type=Path,default=ROOT/'models/yolo11s.pt',help='Local YOLO11s .pt weights')
    p.add_argument('--device',default='cpu',help='cpu, mps or CUDA device index')
    p.add_argument('--weights',nargs=3,type=float,metavar=('HEAD','TRUNK','SHOULDER'),help='Nonnegative weights summing to 1')
    p.add_argument('--calibration',type=Path,default=ROOT/'configs/scoring.json')
    p.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'),help='Optional fixed player crop in source pixels; single video only')
    p.add_argument('--reuse-cache',type=Path,help='A previous clips/clip-... directory; single video only, verifies source hash')
    p.add_argument('--frames',type=int,default=6,help='Source frames shown per panorama row (2 to 12)')
    p.add_argument('--font',type=Path,help='Optional TTF/TTC font supporting Korean')
    return p


def run(args):
    # Do not create outputs, load a model, or download anything on invalid input.
    videos=[p.expanduser().resolve(strict=True) for p in args.video]
    if any(not p.is_file() for p in videos):
        raise ValueError('Every --video must be a regular file')
    if len(set(videos)) != len(videos):
        raise ValueError('Duplicate source video paths')
    output=args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new directory')
    if args.roi is not None and len(videos)!=1:
        raise ValueError('--roi accepts one video per run')
    if args.reuse_cache is not None and len(videos)!=1:
        raise ValueError('--reuse-cache accepts one video per run')
    if not 2<=args.frames<=12:
        raise ValueError('--frames must be between 2 and 12')
    if args.font is not None and not args.font.is_file():
        raise FileNotFoundError(f'Font does not exist: {args.font}')
    from scoring.release_metrics import measure_video, validate_parameters
    from scoring.video_pipeline import sha256,write_json,infer_clip,load_cached_clip
    from scoring.panorama_report import render_panorama,write_html_report
    import hashlib
    profile_bytes=args.calibration.read_bytes()
    profile_hash=hashlib.sha256(profile_bytes).hexdigest()
    profile=json.loads(profile_bytes.decode('utf-8'))
    if profile.get('schema_version')!=1:
        raise ValueError('Unsupported scoring profile schema_version')
    calibration=profile['calibration']
    weights=dict(zip(('headup','trunk','shoulder'),args.weights)) if args.weights is not None else profile['weights']
    calibration,weights=validate_parameters(calibration,weights)
    model=None;model_hash=None
    if args.reuse_cache is None:
        if not args.model.is_file() or args.model.suffix.lower()!='.pt':
            raise FileNotFoundError('Local YOLO weights missing; run python scripts/setup_models.py or pass --model FILE.pt')
        import torch
        from ultralytics import YOLO
        if args.device=='mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS unavailable; use --device cpu')
        model_hash=sha256(args.model)
        model=YOLO(str(args.model.resolve()))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.dribble-report-',dir=output.parent))
    try:
        items=[];reports=[]
        for i,video in enumerate(videos):
            print(f'[{i+1}/{len(videos)}] {video.name}: '+('replaying verified cache' if args.reuse_cache else 'extracting pose and ball candidates'),flush=True)
            if args.reuse_cache:
                pose,candidates,metadata=load_cached_clip(args.reuse_cache,video,args.roi)
            else:
                bundle=staging/'clips'/f'clip-{i+1:02d}-{sha256(video)[:12]}'
                pose,candidates,metadata=infer_clip(video,bundle,model,args.device,args.roi)
                metadata['yolo_model_sha256']=model_hash
                write_json(bundle/'metadata.json',metadata)
            report=measure_video(pose,candidates,metadata['fps'],calibration=calibration,weights=weights)
            report.update(video=metadata,provisional=True,cache_replay=args.reuse_cache is not None,
                          scoring_profile=dict(name=args.calibration.name,sha256=profile_hash))
            reports.append(report)
            items.append(dict(video_path=video,pose=pose,report=report))
        render_panorama(items,staging/'panorama.png',n_frames=args.frames,font_path=args.font)
        write_json(staging/'report.json',dict(version='2026-10-07-video-report',provisional=True,
                                             calibration=calibration,clips=reports))
        write_html_report(reports,staging/'panorama.png',staging/'report.html')
        if sha256(args.calibration) != profile_hash:
            raise ValueError('Scoring profile changed during analysis; rerun')
        for item in items:
            if sha256(item['video_path']) != item['report']['video']['video_sha256']:
                raise ValueError('Source video changed during report rendering; rerun')
        if model_hash is not None and sha256(args.model)!=model_hash:
            raise ValueError('YOLO model changed during analysis')
        if output.exists():
            raise FileExistsError('Output was created by another process; choose a new directory')
        staging.rename(output)
        for report in reports:
            values=' | '.join(f"{k}: {v['score']:.2f}" if v['score'] is not None else f"{k}: N/A ({v['reason']})" for k,v in report['metrics'].items())
            total='N/A' if report['total'] is None else f"{report['total']:.2f}"
            print(f"{report['video']['name']}: {values} | total: {total}")
        print(f'Report: {output / "report.html"}')
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only this run's private temporary output.


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        run(args)
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        print(f'analyze_video: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
