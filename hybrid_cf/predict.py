"""Rating prediction with item-based hybrid similarity."""
import numpy as np


def item_means(ratings):
    """Mean rating of each item over users who rated it; 0 for unrated items."""
    counts = (ratings > 0).sum(axis=0)
    return ratings.sum(axis=0) / np.maximum(counts, 1)


def nearest_neighbours(rating_sim, k):
    """Top-k neighbour indices per item: the k highest similarities after the first.

    Each similarity value maps back to the first item holding it, so tied values
    select the same item more than once. This keeps the original replication's
    MAE figures; use np.argsort directly for distinct neighbours.
    """
    neighbours = np.empty((len(rating_sim), k), dtype=int)
    for item, row in enumerate(rating_sim):
        top = np.sort(row)[::-1][1:k + 1]
        values, first_index = np.unique(row, return_index=True)
        neighbours[item] = first_index[np.searchsorted(values, top)]
    return neighbours


def predict(ratings, rating_sim, attr_sim, users, items, k, means=None):
    """Predict ratings for (users[j], items[j]) pairs (0-based) using k neighbours."""
    if means is None:
        means = item_means(ratings)
    nbrs = nearest_neighbours(rating_sim, k)[items]
    neighbour_ratings = ratings[users[:, None], nbrs]
    rated = neighbour_ratings > 0
    weight = rating_sim[items[:, None], nbrs] * attr_sim[items[:, None], nbrs]

    numerator = (weight * (neighbour_ratings - means[nbrs]) * rated).sum(axis=1)
    denominator = (np.abs(weight) * rated).sum(axis=1)
    denominator[denominator == 0] = 1
    return means[items] + numerator / denominator
