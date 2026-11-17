# Deploying MedTerm Extractor

**Architecture:** frontend (static) on Vercel, backend (Flask + TrOCR) on a Docker host.
Vercel serverless can't run torch/TrOCR (250 MB limit, no persistent disk), so the API lives elsewhere.

## 1. Push to GitHub
```bash
cd AC-Enhancement-first-def
git init && git add . && git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## 2. Deploy the backend (Hugging Face Space, free 16 GB RAM)
1. huggingface.co → New Space → SDK: **Docker**.
2. Push the contents of `medterm-extractor/backend/` to the Space repo (Dockerfile at its root). This must include `ocr_cache.sqlite3`.
3. Space → Settings → Variables: `ALLOWED_ORIGINS=https://<your-app>.vercel.app`
4. URL will be `https://<user>-<space>.hf.space`. Check `/health`.

(Alternatives: Render/Railway/Fly with Docker; needs ≥4 GB RAM for trocr-large.)

## 3. Deploy the frontend on Vercel
1. vercel.com → Add New Project → import the GitHub repo.
2. **Root Directory:** `medterm-extractor/frontend`. Framework: Other. No build command.
3. Edit `frontend/config.js`: `window.API_BASE = "https://<user>-<space>.hf.space";` commit & push.
4. Add your final Vercel URL to the backend's `ALLOWED_ORIGINS`.

## Phone uploads (what's handled)
- **Rotation:** EXIF orientation is applied server-side, so portrait phone photos aren't read sideways.
- **HEIC (iPhone):** decoded server-side via `pillow-heif`.
- **Big photos:** files > 2.5 MB are downsized in the browser (max 2400 px, JPEG) before upload; the server also caps at `MAX_IMAGE_SIDE` (default 2400). Server upload limit is `MAX_UPLOAD_MB` (default 30).
- **Cold starts:** the TrOCR model is baked into the Docker image and preloaded at startup (`PRELOAD_MODEL=1`). Free HF Spaces sleep after inactivity, so the first request after a nap can take ~1 min; the UI shows a "server waking up" message instead of a cryptic error.
- **OCR cache:** results are looked up by exact file hash first, then by image *content* (`backend/image_utils.py` + `ocr_cache.py`), so a phone re-save of a cached image (re-compressed, resized by the browser, HEIC, rotated) still hits the cache. `backend/ocr_cache.sqlite3` is now deployed with the backend (it is no longer git/docker-ignored).
  **One-time step before pushing to the Space:** attach fingerprints to the cache rows by running, locally, `cd medterm-extractor/backend && py backfill_fingerprints.py "<folder with your original prescription images>"` (needs `pip install pillow numpy pillow-heif`). Rows added later are fingerprinted automatically. A brand-new photo is a different picture and goes through live TrOCR. On a free Space new cache entries disappear on restart, so re-run the backfill locally and re-push to make them permanent.
- **Accuracy tips for users:** good light, no shadows, hold the phone parallel to the page, fill the frame, keep lines horizontal. TrOCR reads one line at a time, so tilted or curved pages hurt most.
