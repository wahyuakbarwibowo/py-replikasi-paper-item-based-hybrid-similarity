# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Replication of the paper "An Item-based Collaborative Filtering Method using Item-based Hybrid Similarity" (PDF and Indonesian translation `.docx` in repo root) for a recommender-systems course. Dataset: MovieLens 100k (`ml-100k/`, 943 users × 1682 movies, 5-fold splits `u1..u5.base/.test`).

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy, matplotlib
.venv/bin/python -m hybrid_cf --fold 1 [--k 2 5 10] [--plot]         # MAE per k
.venv/bin/python -m unittest                                          # all tests
.venv/bin/python -m unittest tests.test_predict.HybridPredictionTest.test_predict
```

## Architecture (`hybrid_cf/`)

Everything is dense numpy matrices, recomputed each run (no cached artifacts):

- `data.py` — `load_ratings(fold, "base"|"test")` → users×items matrix (0 = unrated); `load_genres()` → items×genres 0/1 from `u.item`.
- `similarity.py` — `rating_similarity` (cosine over co-rated users only, diagonal 1.0) and `attribute_similarity` (fraction of matching genre flags).
- `predict.py` — `predict(...)` = item mean + Σ w·(r − mean_nbr) / Σ|w| over the user's rated top-k neighbours, w = rating_sim × attr_sim. Takes 0-based index arrays, vectorized over all pairs.
- `evaluate.py` — `mae_by_k(fold, ks)`; similarities are built from the **training** split only.

## Replication quirks (kept on purpose so MAE matches the original scripts)

- `nearest_neighbours` maps sorted similarity values back to the first item with that value, so ties pick the same item repeatedly. Changing this changes the MAE figures.
- `GENRE_COLUMNS` uses 14 of the 19 genres.
- Predictions are not clipped to 1–5.
