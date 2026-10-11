# -*- coding: utf-8 -*-
import importlib.util
from pathlib import Path
import tempfile
import unittest
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import sys


class ModelSetupTests(unittest.TestCase):
    def test_offline_primary_setup_installs_yolov8_and_sam2_without_network(self):
        path=Path(__file__).resolve().parents[1]/'scripts/setup_models.py'
        spec=importlib.util.spec_from_file_location('setup_models',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            pose,yolo,sam=(root/name for name in ('pose-source','yolo-source','sam-source'))
            for source,content in ((pose,b'pose'),(yolo,b'yolo8'),(sam,b'sam2')):
                source.write_bytes(content)
            fake_mp=SimpleNamespace(__file__=str(root/'mp/__init__.py'),solutions=object())
            original_open=module.urlopen
            def offline(url, **kwargs):
                self.assertTrue(url.startswith('file:'), 'offline setup attempted a network download')
                return original_open(url,**kwargs)
            with patch.dict(sys.modules,{'mediapipe':fake_mp}), patch.object(module,'ROOT',root), \
                 patch.object(module,'POSE_SHA',module.digest(pose)), \
                 patch.object(module,'YOLOV8_SHA',module.digest(yolo),create=True), \
                 patch.object(module,'SAM2_SHA',module.digest(sam),create=True), \
                 patch.object(module,'urlopen',side_effect=offline):
                status=module.main(['--pose-source',str(pose),'--yolo-source',str(yolo),'--sam2-source',str(sam)])
            self.assertEqual(status,0)
            self.assertEqual((root/'models/yolov8n.pt').read_bytes(),b'yolo8')
            self.assertEqual((root/'sam2_checkpoints/sam2.1_hiera_small.pt').read_bytes(),b'sam2')
            self.assertFalse((root/'models/yolo11s.pt').exists())

    def test_primary_setup_check_never_downloads_or_creates_assets(self):
        path=Path(__file__).resolve().parents[1]/'scripts/setup_models.py'
        spec=importlib.util.spec_from_file_location('setup_models',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            fake_mp=SimpleNamespace(__file__=str(root/'mp/__init__.py'),solutions=object())
            with patch.dict(sys.modules,{'mediapipe':fake_mp}), patch.object(module,'ROOT',root), \
                 patch.object(module,'urlopen',side_effect=AssertionError('check attempted download')):
                status=module.main(['--check','--ball-backend','yolov8_sam2'])
            self.assertEqual(status,1)
            self.assertEqual(list(root.iterdir()),[])

    def test_bad_model_checksum_never_replaces_existing_file(self):
        path=Path(__file__).resolve().parents[1]/'scripts/setup_models.py'
        self.assertTrue(path.is_file(),'Explicit model setup helper is required')
        spec=importlib.util.spec_from_file_location('setup_models',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';source.write_bytes(b'incorrect')
            dest=Path(td)/'dest';dest.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError,'SHA|checksum'):
                module.install_asset(source.as_uri(),dest,hashlib.sha256(b'correct').hexdigest())
            self.assertEqual(dest.read_bytes(),b'preserve')

    def test_model_setup_verifies_bytes_and_skips_existing_matching_file(self):
        path=Path(__file__).resolve().parents[1]/'scripts/setup_models.py'
        self.assertTrue(path.is_file(),'Explicit model setup helper is required')
        spec=importlib.util.spec_from_file_location('setup_models',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source';source.write_bytes(b'weights')
            dest=Path(td)/'models/model'
            checksum=hashlib.sha256(b'weights').hexdigest()
            module.install_asset(source.as_uri(),dest,checksum)
            source.unlink()
            module.install_asset(source.as_uri(),dest,checksum)
            self.assertEqual(dest.read_bytes(),b'weights')


if __name__=='__main__':
    unittest.main()
