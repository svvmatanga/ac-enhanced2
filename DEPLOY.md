# Deploying on Vercel (single project, no separate backend)

Layout: `index.py` (entrypoint) · `public/` (all pages) · `backend/` (Flask API, algorithms, data, OCR cache) · `requirements.txt` (slim) · `vercel.json`.

1. Copy your backfilled `ocr_cache.sqlite3` over `backend/ocr_cache.sqlite3`
   (or run `py backfill_fingerprints.py "<folder of original prescription images>"` inside `backend/`).
2. `git add . && git commit -m "..." && git push`
3. vercel.com -> Add New -> Project -> import the repo. Root Directory: repository root. Framework: Other/Flask. No build command. Deploy.

Limits on Vercel: no live handwriting OCR (torch/TrOCR does not fit); images already in the cache (including phone copies) are read from it, new images return an OCR error. Request bodies are limited to 4.5 MB (the page shrinks big photos first). Cache entries added online last only until the instance restarts.
For live OCR run the app locally (`cd backend && py app.py`) with `pip install -r requirements-full.txt`.
