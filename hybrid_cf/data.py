"""Load MovieLens 100k into dense matrices."""
import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "ml-100k"
N_USERS = 943
N_ITEMS = 1682

# u.item columns 5..18 (genres "unknown".."Romance"), matching the original replication.
# ponytail: the last 5 genres (Sci-Fi..Western) are left out; widen the slice to use all 19.
GENRE_COLUMNS = slice(5, 19)


def load_ratings(fold, split, data_dir=DATA_DIR):
    """Return a users x items rating matrix for u{fold}.{split}; 0 means unrated."""
    rows = np.loadtxt(data_dir / f"u{fold}.{split}", dtype=int, usecols=(0, 1, 2))
    matrix = np.zeros((N_USERS, N_ITEMS))
    matrix[rows[:, 0] - 1, rows[:, 1] - 1] = rows[:, 2]
    return matrix


def load_genres(data_dir=DATA_DIR):
    """Return an items x genres 0/1 matrix from u.item."""
    with open(data_dir / "u.item", encoding="latin-1", newline="") as f:
        return np.array([row[GENRE_COLUMNS] for row in csv.reader(f, delimiter="|")], dtype=int)
