FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Bake the TrOCR model into the image so cold starts don't re-download ~2 GB
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel as M; \
TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
M.from_pretrained('microsoft/trocr-large-handwritten')"

COPY . .
ENV PORT=7860 HF_HOME=/app/hf_cache TRANSFORMERS_CACHE=/app/hf_cache
EXPOSE 7860
# 1 worker: the model and automatons live in memory. Long timeout for first model download/load.
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 300
