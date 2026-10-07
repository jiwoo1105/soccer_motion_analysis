# -*- coding: utf-8 -*-
import importlib.util
from pathlib import Path
import tempfile
import unittest
import hashlib


class ModelSetupTests(unittest.TestCase):
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
