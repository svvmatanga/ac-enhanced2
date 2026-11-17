"""One-off: let phone copies of your ORIGINAL prescription images hit the cache.

The existing cache rows only know the SHA-256 of the original files. Point this
script at the folder holding those original images and it attaches a content
fingerprint to every image that is already in the cache.

    cd backend
    python backfill_fingerprints.py path/to/original_images
"""
import os
import sys

from ocr_cache import OCRCache

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def main(folder):
    cache = OCRCache(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache.sqlite3"))
    done = skipped = 0
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in EXTS:
                continue
            with open(os.path.join(root, name), "rb") as f:
                ok = cache.add_fingerprint(f.read())
            print(("fingerprinted  " if ok else "not in cache   ") + name)
            done, skipped = done + ok, skipped + (not ok)
    print(f"\n{done} image(s) fingerprinted, {skipped} not found in the cache.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
