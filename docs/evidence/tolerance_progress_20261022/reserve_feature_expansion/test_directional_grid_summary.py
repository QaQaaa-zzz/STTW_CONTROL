"""Behavior checks for finite-grid evidence, independent of the GPU runtime."""
import runpy
import unittest
from pathlib import Path

module = runpy.run_path(str(Path(__file__).with_name('summarize_physical_results.py')))


def rows(labels):
    base = dict(panel='low', policy='test', route='left', speed=1.5,
                alpha=0, onset_s=3.2, prefix_features={'z': .2})
    return [dict(base, force_N=f, balance_label=label) for f, label in labels]


S, F, U = 'COMPLETE_NO_CONTACT', 'FLOOR_CONTACT', 'UNKNOWN'


class DirectionalGridTest(unittest.TestCase):
    def summarize(self, labels):
        return module['directional_grid_summary'](rows(labels))

    def test_signed_bracket_and_censoring(self):
        result = self.summarize([(0, S), (32, S), (48, F), (-32, S), (-48, S)])
        positive = next(r for r in result if r['direction'] == 1)
        negative = next(r for r in result if r['direction'] == -1)
        self.assertEqual(positive['status'], 'GRID_BRACKET')
        self.assertEqual(positive['largest_observed_safe_N'], 32)
        self.assertEqual(positive['smallest_observed_contact_N'], 48)
        self.assertEqual(negative['status'], 'RIGHT_CENSORED_GRID')

    def test_nonmonotonic_is_not_a_threshold(self):
        self.assertEqual(self.summarize([(0, S), (32, F), (48, S)])[0]['status'],
                         'NONMONOTONIC_GRID')

    def test_unknown_and_nominal_contact(self):
        self.assertEqual(self.summarize([(0, S), (32, U), (48, F)])[0]['status'],
                         'UNKNOWN_IN_GRID')
        self.assertEqual(self.summarize([(0, F), (32, S)])[0]['status'],
                         'ZERO_FORCE_NOT_COMPLETE_SAFE')

    def test_changed_or_missing_prefix(self):
        sample = rows([(0, S), (32, S)])
        sample[1]['prefix_features'] = None
        self.assertEqual(module['directional_grid_summary'](sample)[0]['status'],
                         'PREFIX_NOT_REACHED')
        sample[1]['prefix_features'] = {'z': .3}
        with self.assertRaises(ValueError):
            module['directional_grid_summary'](sample)


if __name__ == '__main__':
    unittest.main()
