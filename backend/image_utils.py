"""Image loading/normalisation shared by the OCR engine and the OCR cache.

Phone uploads differ from desktop files in ways that break a byte-exact
pipeline: HEIC/HEIF containers, EXIF rotation flags (the pixels are stored
sideways), 12 MP resolutions, and re-compression by the gallery / chat apps.
Everything here is Pillow + NumPy only (no torch / OpenCV) so the cache can
use it without the heavy OCR dependencies.
"""

import io

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC / HEIF (iPhone default format)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

# Longest side used for OCR. Phone photos (4000x3000) are only ever
# downscaled to this, never upscaled.
OCR_MAX_SIDE = 2000

# Fingerprint grid (gray levels) used to recognise the *same* image after it
# was re-compressed / resized / re-saved by a phone.
FP_SIZE = 48


def load_normalized(image_bytes, max_side=OCR_MAX_SIDE):
    """Decode bytes -> upright, opaque RGB PIL image, capped at `max_side`."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)  # honour the phone's rotation flag
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")
    longest = max(img.size)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.LANCZOS,
        )
    return img


def _gray_grid(img):
    gray = img.convert("L").resize((FP_SIZE, FP_SIZE), Image.BOX)
    return np.asarray(gray, dtype=np.uint8)


def fingerprint(image_bytes):
    """Return (grid_bytes, aspect) describing the image content."""
    img = load_normalized(image_bytes, max_side=512)
    return _gray_grid(img).tobytes(), img.width / img.height


def fingerprint_variants(image_bytes):
    """Fingerprints of the upright image and its 90/180/270 rotations, so a
    picture that was rotated by hand before upload still matches."""
    img = load_normalized(image_bytes, max_side=512)
    variants = []
    for angle in (0, 90, 180, 270):
        rotated = img.rotate(angle, expand=True) if angle else img
        variants.append((_gray_grid(rotated).tobytes(), rotated.width / rotated.height))
    return variants


def similarity(grid_a, grid_b):
    """Pearson correlation (-1..1) between two stored gray grids."""
    a = np.frombuffer(grid_a, dtype=np.uint8).astype(np.float64)
    b = np.frombuffer(grid_b, dtype=np.uint8).astype(np.float64)
    if a.size != b.size:
        return -1.0
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return -1.0
    return float((a * b).sum() / denom)
