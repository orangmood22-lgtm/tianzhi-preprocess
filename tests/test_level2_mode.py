"""CPU checks for the optional depth-2 classifier policy."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UTIL = ROOT / 'SARdetect示例/D.codes/2.utils'
sys.path.insert(0, str(UTIL))
sys.modules.setdefault('cv2', types.ModuleType('cv2'))
import competition_metrics as metrics

projection_path = ROOT / 'SARdetect示例/D.codes/1.model/modelcs-HierarchicalEnsemble/level2_projection.py'
spec = importlib.util.spec_from_file_location('level2_projection', projection_path)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


TAX = {
    'level1': ['A', 'B'],
    'level2': [{'name': 'A1', 'parent': 0}, {'name': 'A2', 'parent': 0}, {'name': 'B1', 'parent': 1}],
    'leaves': [
        {'l1': 0, 'l2': 0, 'l3': 0, 'level3': 'A11'},
        {'l1': 0, 'l2': 0, 'l3': 1, 'level3': 'A12'},
        {'l1': 0, 'l2': 1, 'l3': 2, 'level3': 'A21'},
        {'l1': 1, 'l2': 2, 'l3': 3, 'level3': 'B11'},
    ],
}


class Level2PolicyTests(unittest.TestCase):
    def test_mixed_decoder_keeps_original_selection(self):
        probabilities = np.random.default_rng(42).dirichlet(np.ones(4), size=100)
        actions = metrics.actions(TAX)
        utility = probabilities @ metrics.reward_matrix(TAX).T
        original = []
        for row in utility:
            candidates = np.flatnonzero(row >= row.max() - 1e-8)
            depth = min(actions[i][0] for i in candidates)
            candidates = [i for i in candidates if actions[i][0] == depth]
            original.append(int(max(candidates, key=lambda i: row[i])))
        actual, _ = metrics.decode(probabilities, TAX)
        np.testing.assert_array_equal(actual, original)

    def test_fixed_level2_decoder_only_emits_level2(self):
        probabilities = np.random.default_rng(5).dirichlet(np.ones(4), size=50)
        indices, _ = metrics.decode(probabilities, TAX, output_depth=2)
        self.assertTrue(all(metrics.actions(TAX)[i][0] == 2 for i in indices))
        with self.assertRaises(ValueError):
            metrics.decode(probabilities, TAX, output_depth=4)

    def test_parent_projection_preserves_level2_mass(self):
        parent = np.asarray([[.2, .3, .5], [.8, .1, .1]])
        leaves = projection.expand_parent_probabilities(parent, TAX)
        np.testing.assert_allclose(leaves.sum(axis=1), np.ones(2))
        for index in range(3):
            members = [j for j, row in enumerate(TAX['leaves']) if row['l2'] == index]
            np.testing.assert_allclose(leaves[:, members].sum(axis=1), parent[:, index])


if __name__ == '__main__':
    unittest.main()
