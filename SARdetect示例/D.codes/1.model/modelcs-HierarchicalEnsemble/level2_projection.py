"""Project a level-2 retrieval distribution onto leaves for the existing ensemble."""
import numpy as np


def expand_parent_probabilities(probabilities, taxonomy):
    """Give each leaf an equal share of its parent's mass; parent totals stay exact."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(taxonomy['level2']):
        raise ValueError('Expected one probability per level-2 class')
    parents = np.asarray([row['l2'] for row in taxonomy['leaves']], dtype=np.int64)
    if len(parents) == 0 or np.any(parents < 0) or np.any(parents >= values.shape[1]):
        raise ValueError('Invalid leaf-to-parent taxonomy')
    counts = np.bincount(parents, minlength=values.shape[1])
    if np.any(counts == 0):
        raise ValueError('Every level-2 class needs at least one leaf')
    return values[:, parents] / counts[parents]
