# Replikasi Paper / Prosiding dengan Judul "An Item-based Collaborative Filtering Method using Item-based Hybrid Similarity"
Proyek Replikasi Paper merupakan tugas kuliah mata kuliah sistem rekomendasi dengan judul paper/prosiding yaitu "An Item-based Collaborative Filtering Method using Item-based Hybrid Similarity".

## Menjalankan

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m hybrid_cf --fold 1 --plot   # MAE untuk k = 2..40, fold 1..5
.venv/bin/python -m unittest                    # test contoh toy dari paper
```

Dataset: MovieLens 100k (`ml-100k/`). Matriks rating dan similarity dihitung langsung saat run (±1–2 detik per fold).
