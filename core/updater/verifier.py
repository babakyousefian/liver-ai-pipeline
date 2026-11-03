# core/updater/verifier.py
# -*- coding: utf-8 -*-
"""
Download verification: size, SHA256, domain whitelist.
"""

import hashlib
import os
from typing import Optional


def sha256_of_file(path: str, chunk_size: int = 65536,
                   progress_cb=None) -> str:
    """
    Compute SHA256 of a file.
    Optionally call progress_cb(bytes_read, total_bytes) every chunk.
    """
    total = os.path.getsize(path)
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if progress_cb:
                try:
                    progress_cb(read, total)
                except Exception:
                    pass
    return h.hexdigest()


def verify_checksum(path: str, expected_sha256: str,
                    progress_cb=None) -> bool:
    """
    Verify SHA256. Returns True if match, False otherwise.
    If expected is empty, returns True (no checksum to verify).
    """
    if not expected_sha256:
        return True
    if not os.path.isfile(path):
        return False
    actual = sha256_of_file(path, progress_cb=progress_cb)
    ok = actual.lower() == expected_sha256.lower()
    if not ok:
        print(f"❌ Updater: checksum mismatch\n  expected: {expected_sha256}\n  actual:   {actual}")
    return ok


def verify_size(path: str, expected_size: int, tolerance: int = 0) -> bool:
    """Verify file size matches expected (with optional tolerance in bytes)."""
    if expected_size <= 0:
        return True
    if not os.path.isfile(path):
        return False
    actual = os.path.getsize(path)
    return abs(actual - expected_size) <= tolerance


def verify_extension(path: str, allowed: tuple = (".exe", ".msi", ".zip", ".tar.gz", ".dmg")) -> bool:
    """Ensure the downloaded file has a sane extension."""
    return path.lower().endswith(allowed)


def is_safe_filename(name: str) -> bool:
    """Reject path traversal and weird names."""
    if not name:
        return False
    # No path separators, no '..', no null bytes
    bad = ("/", "\\", "..", "\x00")
    return not any(b in name for b in bad)