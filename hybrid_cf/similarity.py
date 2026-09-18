"""Item-item similarity matrices."""
import numpy as np


def rating_similarity(ratings):
    """Cosine similarity between item columns, computed over co-rated users only."""
    rated = (ratings > 0).astype(float)
    squared = ratings ** 2
    dot = ratings.T @ ratings
    norm = np.sqrt(squared.T @ rated) * np.sqrt(rated.T @ squared)
    norm[norm == 0] = 1
    sim = dot / norm
    np.fill_diagonal(sim, 1.0)
    return sim


def attribute_similarity(genres):
    """Fraction of genre flags two items have in common (both set or both unset)."""
    same = genres @ genres.T + (1 - genres) @ (1 - genres).T
    return same / genres.shape[1]
