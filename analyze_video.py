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
    p.add_argument('--ball-backend',choices=('yolov8_sam2','yolo11_roi'),default='yolov8_sam2',
                   help='Original YOLOv8+SAM2 (default), or YOLO11 ROI compatibility')
    p.add_argument('--model',type=Path,help='Local YOLO weights; defaults to models/yolov8n.pt or models/yolo11s.pt by backend')
    p.add_argument('--sam2-checkpoint',type=Path,help='Local SAM2.1 small weights; defaults to sam2_checkpoints/sam2.1_hiera_small.pt')
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
    from scoring.release_metrics import VERSION, measure_video, validate_measurement, validate_parameters
    from scoring.video_pipeline import sha256,write_json,infer_clip,load_cached_clip,local_weight
    from scoring.panorama_report import render_panorama,write_html_report
    import hashlib
    profile_bytes=args.calibration.read_bytes()
    profile_hash=hashlib.sha256(profile_bytes).hexdigest()
    profile=json.loads(profile_bytes.decode('utf-8'))
    if not isinstance(profile,dict):
        raise ValueError('Scoring profile must be a JSON object')
    weights=dict(zip(('headup','trunk','shoulder'),args.weights)) if args.weights is not None else profile['weights']
    calibration,weights=validate_parameters(profile,weights)
    measurement=validate_measurement(profile['measurement'])
    model=None;model_hash=None;sam_hash=None;sam_path=None;cached=None
    model_path=args.model or ROOT/'models'/('yolov8n.pt' if args.ball_backend=='yolov8_sam2' else 'yolo11s.pt')
    if args.reuse_cache is not None:
        cached=load_cached_clip(args.reuse_cache,videos[0],args.roi,ball_backend=args.ball_backend,
                                yolo_model_path=args.model,sam2_checkpoint=args.sam2_checkpoint)
    else:
        model_path=local_weight(model_path,'YOLO')
        if args.ball_backend=='yolov8_sam2':
            sam_path=local_weight(args.sam2_checkpoint or ROOT/'sam2_checkpoints/sam2.1_hiera_small.pt','SAM2 checkpoint')
            from importlib.util import find_spec
            if find_spec('sam2') is None:
                raise ImportError('SAM2 is not installed; install requirements-sam2.txt in this Python environment')
        import torch
        from ultralytics import YOLO
        device=f'cuda:{args.device}' if str(args.device).isdigit() else args.device
        if device not in ('cpu','mps','cuda') and not (device.startswith('cuda:') and device[5:].isdigit()):
            raise ValueError('--device must be cpu, mps, cuda, cuda:N, or a CUDA index')
        if device=='mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS unavailable; use --device cpu')
        if device.startswith('cuda') and (not torch.cuda.is_available() or
                (':' in device and int(device.split(':')[1])>=torch.cuda.device_count())):
            raise ValueError('Requested CUDA device unavailable; use --device cpu')
        model_hash=sha256(model_path)
        sam_hash=sha256(sam_path) if sam_path else None
        model=YOLO(str(model_path))
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.dribble-report-',dir=output.parent))
    try:
        items=[];reports=[]
        for i,video in enumerate(videos):
            print(f'[{i+1}/{len(videos)}] {video.name}: '+('replaying verified cache' if args.reuse_cache else 'extracting pose and ball candidates'),flush=True)
            if args.reuse_cache:
                pose,ball_data,metadata=cached
            else:
                bundle=staging/'clips'/f'clip-{i+1:02d}-{sha256(video)[:12]}'
                pose,ball_data,metadata=infer_clip(video,bundle,model,args.device,args.roi,
                    ball_backend=args.ball_backend,yolo_model_path=model_path,sam2_checkpoint=sam_path)
            if metadata['ball_source']!=args.ball_backend:
                raise ValueError('Extracted ball_source differs from selected backend')
            if args.ball_backend=='yolov8_sam2':
                report=measure_video(pose,None,metadata['fps'],calibration=calibration,weights=weights,
                                     measurement=measurement,ball_tracks=ball_data)
            else:
                report=measure_video(pose,ball_data,metadata['fps'],calibration=calibration,weights=weights,
                                     measurement=measurement)
            report.update(video=metadata,provisional=True,cache_replay=args.reuse_cache is not None,
                          scoring_profile=dict(name=args.calibration.name,sha256=profile_hash))
            reports.append(report)
            items.append(dict(video_path=video,pose=pose,report=report))
        render_panorama(items,staging/'panorama.png',n_frames=args.frames,font_path=args.font)
        write_json(staging/'report.json',dict(version=VERSION,provisional=True,
                                             calibration=calibration,measurement=measurement,clips=reports))
        write_html_report(reports,staging/'panorama.png',staging/'report.html')
        if sha256(args.calibration) != profile_hash:
            raise ValueError('Scoring profile changed during analysis; rerun')
        for item in items:
            if sha256(item['video_path']) != item['report']['video']['video_sha256']:
                raise ValueError('Source video changed during report rendering; rerun')
        if model_hash is not None and sha256(model_path)!=model_hash:
            raise ValueError('YOLO model changed during analysis')
        if sam_hash is not None and sha256(sam_path)!=sam_hash:
            raise ValueError('SAM2 checkpoint changed during analysis')
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
