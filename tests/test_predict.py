"""Toy example from the paper; expected values come from the original replication scripts."""
import unittest

import numpy as np

from hybrid_cf.predict import predict
from hybrid_cf.similarity import attribute_similarity, rating_similarity

RATINGS = np.array([[3, 2, 5, 4], [0, 5, 1, 0], [2, 5, 0, 3], [2, 1, 3, 2], [2, 0, 5, 5]], dtype=float)
GENRES = np.array([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1], [1, 0, 1, 1, 1], [0, 1, 1, 1, 1]])


class HybridPredictionTest(unittest.TestCase):
    def test_similarities(self):
        np.testing.assert_allclose(rating_similarity(RATINGS)[0], [1.0, 0.797053397, 0.978838916, 0.950262193])
        np.testing.assert_allclose(attribute_similarity(GENRES)[0], [1.0, 0.0, 0.8, 0.4])

    def test_predict(self):
        sim, attr = rating_similarity(RATINGS), attribute_similarity(GENRES)
        users, items = np.array([1, 1, 2, 4]), np.array([0, 3, 2, 1])  # 0-based (user, item)
        expected = {
            1: [-0.25, 1.0, 3.0, 4.75],
            2: [-0.25, 1.0, 3.14217616915225, 4.75],
            3: [-0.25, 2.9605395665152052, 3.299548572051549, 4.75],
        }
        for k, want in expected.items():
            np.testing.assert_allclose(predict(RATINGS, sim, attr, users, items, k), want, err_msg=f"k={k}")


if __name__ == "__main__":
    unittest.main()
