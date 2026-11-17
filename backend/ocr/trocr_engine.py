"""
Transformer-based Optical Character Recognition (TrOCR) engine.

Per the thesis (3.4.2, "Context-Aware Validation"):
    "TrOCR, implemented via the Hugging Face Transformers library with
    PyTorch as the underlying deep learning engine, converts prescription
    images into raw text strings. OpenCV (cv2) and Pillow (PIL) are used
    for image preprocessing prior to TrOCR: OpenCV converts the image to
    grayscale, applies adaptive thresholding, dilation, and contour
    detection to isolate handwritten regions, while Pillow handles image
    loading and format normalization."

This module is shared by BOTH the original and enhanced pipelines --
OCR is the input layer that happens before either algorithm runs; it is
not itself part of the Aho-Corasick enhancement being studied.

NOTE: the first call downloads TrOCR's handwritten-text model weights
(microsoft/trocr-large-handwritten) from Hugging Face, which requires
an internet connection. The model is cached locally afterwards.
"""

import io
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps

# Register HEIC/HEIF support for Pillow
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed; HEIC support unavailable

_processor = None
_model = None

# Phone cameras produce 8-12 MP photos. Cap the long side so segmentation/OCR
# behave consistently and stay fast + memory-friendly on a CPU server.
import os as _os
MAX_IMAGE_SIDE = int(_os.environ.get("MAX_IMAGE_SIDE", "2400"))


def _open_image(image_bytes):
    """Open an uploaded image robustly (phone photos included).

    - Applies EXIF orientation, so a portrait photo taken on a phone is not
      processed sideways (Pillow ignores EXIF rotation by default).
    - Supports HEIC/HEIF (iPhone) via pillow-heif.
    - Flattens transparency onto white and converts to RGB.
    - Downscales very large photos to MAX_IMAGE_SIDE on the long edge.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    longest = max(img.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


def _fallback_ocr(image_bytes):
    """Best-effort OCR fallback when the TrOCR model cannot be loaded.

    This keeps the web app usable in environments without internet access or
    without a cached handwritten TrOCR model, by using Tesseract if it is
    available on the system.
    """
    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "OCR model could not be loaded and pytesseract is not available. "
            "Install Tesseract OCR or make sure the TrOCR model can download."
        ) from exc

    pil_img = _open_image(image_bytes)
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    text = pytesseract.image_to_string(Image.fromarray(binary), config="--psm 6")
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        raise RuntimeError("OCR fallback produced no readable text from the uploaded image.")
    return cleaned


def _load_model():
    """Lazily loads the TrOCR model/processor once per process."""
    global _processor, _model
    if _model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        _model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        _model.eval()
    return _processor, _model


def segment_lines(image_bytes):
    """
    Splits a prescription image into individual text-line images using a
    horizontal ink-projection profile, rather than OCRing the whole
    multi-line image in one shot (TrOCR's handwritten model is trained to
    read a single line at a time; feeding it several lines at once
    produces garbled/near-empty output).

    Steps:
      1. Grayscale + light blur + adaptive threshold to isolate ink from
         a textured/creased paper background.
      2. Sum ink pixels per row -> a 1-D profile of "how much handwriting
         is on this row". Rows above a small threshold are "active".
      3. Group consecutive active rows into bands, merging bands that are
         only a few pixels apart (so a single line with an ascender/
         descender gap doesn't get split in two).
      4. Crop each band using the FULL image width (so a fragment like
         "#14" positioned far to the right of a line isn't cut off), pad
         it slightly, and upscale short strips so TrOCR gets a usable
         input height.
    """
    pil_img = _open_image(image_bytes)
    img_np = np.array(pil_img)
    h_img, w_img = img_np.shape[:2]

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold handles textured/creased paper backgrounds better
    # than a single global (Otsu) threshold, since it looks at local
    # neighborhoods instead of the whole image's brightness distribution.
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    # Horizontal projection: how much ink is on each row.
    # `binary` is 0/255 per pixel, so np.sum() over a row gives a value
    # scaled by 255, NOT an ink-pixel count. Dividing by 255 here converts
    # it back to an actual pixel count so the threshold below (expressed
    # as "N% of row width in pixels") is compared against the right units.
    # Skipping this fix means the effective threshold is ~255x too low,
    # so almost every row counts as "active" and tightly-spaced or cursive
    # handwriting (little/no fully blank row between lines) collapses into
    # one giant multi-line band -- which TrOCR's single-line model then
    # reads as near-garbage output.
    row_ink_counts = np.sum(binary, axis=1) / 255.0
    threshold = w_img * 0.06  # 6% of row width must be ink to count as "active"
    active = (row_ink_counts > threshold).astype(np.uint8)

    # Group consecutive active rows into line bands.
    line_bands = []
    in_band = False
    start = 0
    for i, val in enumerate(active):
        if val and not in_band:
            in_band = True
            start = i
        elif not val and in_band:
            in_band = False
            line_bands.append((start, i))
    if in_band:
        line_bands.append((start, h_img))

    # Drop trivial noise bands, then merge bands that are very close
    # together (small gaps within the same line, e.g. dotted "i"s).
    line_bands = [(y1, y2) for y1, y2 in line_bands if y2 - y1 >= 3]
    merged_bands = []
    for band in line_bands:
        if merged_bands and band[0] - merged_bands[-1][1] < 4:
            merged_bands[-1] = (merged_bands[-1][0], band[1])
        else:
            merged_bands.append(list(band))

    # Filter out bands that are still too thin to be real text.
    merged_bands = [b for b in merged_bands if b[1] - b[0] >= 10]

    if not merged_bands:
        # Fallback: no lines detected, OCR the whole image as-is.
        return [pil_img]

    pad = 6
    line_images = []
    for (y_start, y_end) in merged_bands:
        y1 = max(0, y_start - pad)
        y2 = min(h_img, y_end + pad)

        # Full-width crop so nothing on the line gets cut off horizontally.
        line_img = pil_img.crop((0, y1, w_img, y2)).convert("RGB")

        # Upscale short strips so TrOCR receives a usable input height.
        lw, lh = line_img.size
        if lh < 32:
            scale = 32 / lh
            line_img = line_img.resize((int(lw * scale), 32), Image.LANCZOS)

        line_images.append(line_img)

    return line_images


# Backwards-compatible alias (older callers may import preprocess_image).
preprocess_image = segment_lines


def extract_text(image_bytes):
    """Runs the full OCR pipeline on raw image bytes and returns extracted text.

    TrOCR is the primary engine, but a lightweight Tesseract fallback is used if
    the model cannot be downloaded or initialized locally. This keeps the app
    usable in environments with limited internet or inconsistent model cache
    availability.
    """
    try:
        processor, model = _load_model()
        line_images = segment_lines(image_bytes)

        lines_text = []
        batch_size = 8
        for start in range(0, len(line_images), batch_size):
            line_batch = line_images[start:start + batch_size]
            pixel_values = processor(
                images=line_batch,
                return_tensors="pt",
            ).pixel_values
            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=200,
                    num_beams=5,
                    early_stopping=True,
                )
            lines_text.extend(
                line.strip()
                for line in processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                if line.strip()
            )

        return "\n".join(lines_text)
    except Exception:
        return _fallback_ocr(image_bytes)