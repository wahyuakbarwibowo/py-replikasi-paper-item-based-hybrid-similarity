"""Mean Absolute Error on a MovieLens train/test fold."""
import numpy as np

from .data import load_genres, load_ratings
from .predict import item_means, predict
from .similarity import attribute_similarity, rating_similarity


def mae_by_k(fold, neighbour_counts):
    """Return {k: MAE} for one fold."""
    train = load_ratings(fold, "base")
    test = load_ratings(fold, "test")
    rating_sim = rating_similarity(train)
    attr_sim = attribute_similarity(load_genres())
    means = item_means(train)

    users, items = np.nonzero(test)
    actual = test[users, items]
    return {
        k: float(np.abs(predict(train, rating_sim, attr_sim, users, items, k, means) - actual).mean())
        for k in neighbour_counts
    }
