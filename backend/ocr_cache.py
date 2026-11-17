"""Persistent cache for OCR results.

Lookup order for an uploaded image:
  1. exact SHA-256 of the file bytes (identical file), then
  2. content fingerprint (same picture after a phone re-compressed, resized,
     rotated or converted it, e.g. HEIC -> JPEG, WhatsApp/Messenger, gallery).

The fingerprint match is deliberately strict because a wrong hit would return
another prescription's text.
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager

from image_utils import fingerprint, fingerprint_variants, similarity

# Pearson correlation of the 48x48 gray grids required to call two images
# "the same picture", and the allowed aspect-ratio difference.
MATCH_THRESHOLD = 0.97
ASPECT_TOLERANCE = 0.03


class OCRCache:
    def __init__(self, database_path):
        self.database_path = database_path
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_results (
                    image_hash TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {r[1] for r in connection.execute("PRAGMA table_info(ocr_results)")}
            if "fingerprint" not in columns:
                connection.execute("ALTER TABLE ocr_results ADD COLUMN fingerprint BLOB")
            if "aspect" not in columns:
                connection.execute("ALTER TABLE ocr_results ADD COLUMN aspect REAL")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def image_hash(image_bytes):
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def _safe_fingerprint(image_bytes):
        try:
            return fingerprint(image_bytes)
        except Exception:  # undecodable image -> exact-hash caching only
            return None, None

    def lookup(self, image_bytes):
        """Return (raw_text, match_type, score) or (None, None, 0.0)."""
        image_hash = self.image_hash(image_bytes)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT raw_text FROM ocr_results WHERE image_hash = ?",
                (image_hash,),
            ).fetchone()
            if row:
                return row[0], "exact", 1.0
            candidates = connection.execute(
                "SELECT raw_text, fingerprint, aspect FROM ocr_results "
                "WHERE fingerprint IS NOT NULL"
            ).fetchall()
        if not candidates:
            return None, None, 0.0
        try:
            variants = fingerprint_variants(image_bytes)
        except Exception:
            return None, None, 0.0

        best_text, best_score = None, 0.0
        for raw_text, stored_grid, stored_aspect in candidates:
            for grid, aspect in variants:
                if stored_aspect and abs(aspect - stored_aspect) / stored_aspect > ASPECT_TOLERANCE:
                    continue
                score = similarity(grid, stored_grid)
                if score > best_score:
                    best_text, best_score = raw_text, score
        if best_text is not None and best_score >= MATCH_THRESHOLD:
            return best_text, "similar", best_score
        return None, None, best_score

    def get(self, image_bytes):
        return self.lookup(image_bytes)[0]

    def put(self, image_bytes, raw_text):
        image_hash = self.image_hash(image_bytes)
        grid, aspect = self._safe_fingerprint(image_bytes)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ocr_results (image_hash, raw_text, fingerprint, aspect)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    raw_text = excluded.raw_text,
                    fingerprint = excluded.fingerprint,
                    aspect = excluded.aspect,
                    created_at = CURRENT_TIMESTAMP
                """,
                (image_hash, raw_text, grid, aspect),
            )

    def add_fingerprint(self, image_bytes):
        """Attach a fingerprint to an existing row (matched by exact hash)."""
        grid, aspect = self._safe_fingerprint(image_bytes)
        if grid is None:
            return False
        with self._connect() as connection:
            cur = connection.execute(
                "UPDATE ocr_results SET fingerprint = ?, aspect = ? WHERE image_hash = ?",
                (grid, aspect, self.image_hash(image_bytes)),
            )
            return cur.rowcount > 0
