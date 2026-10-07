#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicitly install checksum-pinned official model assets; inference never downloads."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
POSE_URL='https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite'
POSE_SHA='59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe'
YOLO_URL='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt'
YOLO_SHA='85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def install_asset(url,destination,expected_sha):
    destination=Path(destination)
    if destination.is_file() and digest(destination)==expected_sha:
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.model-',delete=False) as out:
            temp=Path(out.name)
            with urlopen(url,timeout=90) as response:
                shutil.copyfileobj(response,out)
        if digest(temp)!=expected_sha:
            raise ValueError(f'Model SHA-256 checksum differs: {destination.name}')
        os.replace(temp,destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true',help='Check installed assets without downloading')
    p.add_argument('--pose-source',type=Path,help='Offline copy of the official heavy .tflite asset')
    p.add_argument('--yolo-source',type=Path,help='Offline copy of the official YOLO11s .pt asset')
    a=p.parse_args(argv)
    try:
        import mediapipe as mp
        if not hasattr(mp,'solutions'):
            raise RuntimeError('Install requirements-runtime.txt in a Python 3.11 venv')
        assets=[
            (a.pose_source.resolve().as_uri() if a.pose_source else POSE_URL,
             Path(mp.__file__).parent/'modules/pose_landmark/pose_landmark_heavy.tflite',POSE_SHA),
            (a.yolo_source.resolve().as_uri() if a.yolo_source else YOLO_URL,
             ROOT/'models/yolo11s.pt',YOLO_SHA)]
        for url,path,checksum in assets:
            if a.check:
                if not path.is_file() or digest(path)!=checksum:
                    raise ValueError(f'Model missing or checksum differs: {path}; run setup without --check')
            else:
                install_asset(url,path,checksum)
            print(f'OK {path.name} SHA-256={checksum}')
        return 0
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'setup_models: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
