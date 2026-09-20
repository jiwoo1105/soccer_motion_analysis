import copy
import unittest
from evaluate import verify_snapshot
from scoring.current_metrics import VERSION


def snapshot():
    row = dict(video='sample', grade=3, reference=False, excluded=False,
               exclude_reason=None, head_raw=None, trunk_raw=140., body_M=.1,
               head_score=None, trunk_score=10., body_score=10., total=None,
               events=[], candidate_count=0)
    return dict(version=VERSION, rows=[row])


class SnapshotTests(unittest.TestCase):
    def test_paired_missing_values_match(self):
        verify_snapshot(snapshot(), snapshot())

    def test_one_sided_missing_values_fail(self):
        a, b = snapshot(), snapshot(); a['rows'][0]['head_raw'] = 1.
        with self.assertRaises(ValueError): verify_snapshot(a, b)

    def test_metadata_and_duplicates_fail(self):
        for key, value in [('grade', 9), ('reference', True), ('exclude_reason', 'changed')]:
            a, b = snapshot(), snapshot(); a['rows'][0][key] = value
            with self.assertRaises(ValueError): verify_snapshot(a, b)
        a, b = snapshot(), snapshot(); a['version'] = 'different'
        with self.assertRaises(ValueError): verify_snapshot(a, b)
        a, b = snapshot(), snapshot(); a['rows'].append(copy.deepcopy(a['rows'][0]))
        with self.assertRaises(ValueError): verify_snapshot(a, b)
