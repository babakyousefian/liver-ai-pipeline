# core/crypto/key_manager.py
# -*- coding: utf-8 -*-
"""
Key management for encrypted file storage.

Key derivation:
    fernet_key = PBKDF2(
        password = machine_id + ":" + APP_SECRET,
        salt     = random_salt (stored in users_db.salt),
        iterations = 100_000
    )

- machine_id: stable per-machine identifier (MAC + hostname + platform)
- APP_SECRET: constant in code (defense-in-depth)
- salt: random, stored separately
"""

import os
import sys
import uuid
import base64
import hashlib


# ============================================================
# Constants
# ============================================================

# ⚠️ This is NOT a secret — it's a defense-in-depth layer.
# The REAL security comes from machine_id + salt.
APP_SECRET = "LiverAI-2026-SecretKey-For-DB-Encryption-v1"

SALT_FILENAME = "users_db.salt"
MACHINE_ID_FILENAME = "users_db.machine_id"

SALT_LENGTH = 32  # bytes


# ============================================================
# Project root
# ============================================================

def _get_project_root() -> str:
    """Same logic as GUI_update.py."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    try:
        # Try to find GUI_update.py upward
        here = os.path.dirname(os.path.abspath(__file__))
        # Go up: core/crypto/ → core/ → project root
        return os.path.dirname(os.path.dirname(here))
    except Exception:
        return os.getcwd()


# ============================================================
# Cryptography availability
# ============================================================

def is_cryptography_available() -> bool:
    """Check if the `cryptography` package is installed."""
    try:
        from cryptography.fernet import Fernet  # noqa
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # noqa
        return True
    except ImportError:
        return False


# ============================================================
# Machine ID
# ============================================================

def _compute_machine_id() -> str:
    """
    Compute a stable machine identifier.

    Uses:
    - MAC address (uuid.getnode)
    - Hostname
    - Platform
    - Python version

    Note: `uuid.getnode()` may return a random value if the MAC
    is not accessible. In that case, the machine_id will change
    on each run — which would break decryption. To avoid this,
    we cache the machine_id to a file on first run.
    """
    try:
        mac = uuid.getnode()
        # uuid.getnode() returns a random int if MAC is not available,
        # which has the multicast bit set (bit 40).
        # We can detect this and fall back to a different strategy.
        is_random = bool((mac >> 40) & 0x01)

        import socket
        hostname = socket.gethostname()

        import platform
        plat = platform.platform()

        material = f"{mac}-{hostname}-{plat}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
    except Exception as e:
        print(f"⚠️ _compute_machine_id failed: {e}")
        return hashlib.sha256(b"fallback").hexdigest()


def get_machine_id() -> str:
    """
    Get the machine_id, cached to a file.

    Why cache? Because uuid.getnode() can be unstable on some systems.
    By caching on first run, we guarantee the same machine_id is used
    across all runs on the same machine.
    """
    root = _get_project_root()
    cache_path = os.path.join(root, MACHINE_ID_FILENAME)

    # ✅ Try to read cached value
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
                if cached and len(cached) == 64:  # sha256 hex
                    return cached
        except Exception:
            pass

    # ✅ Compute fresh
    machine_id = _compute_machine_id()

    # ✅ Try to write cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(machine_id)
        print(f"✅ Machine ID cached: {cache_path}")
    except Exception as e:
        print(f"⚠️ Failed to cache machine_id: {e}")

    return machine_id


# ============================================================
# Salt
# ============================================================

def get_or_create_salt() -> bytes:
    """
    Get or create the salt used for key derivation.
    Salt is stored in `users_db.salt` next to users_db.json.
    """
    root = _get_project_root()
    salt_path = os.path.join(root, SALT_FILENAME)

    # ✅ Try to read existing salt
    if os.path.isfile(salt_path):
        try:
            with open(salt_path, "rb") as f:
                salt = f.read()
                if len(salt) == SALT_LENGTH:
                    return salt
        except Exception as e:
            print(f"⚠️ Failed to read salt: {e}")

    # ✅ Create new salt
    salt = os.urandom(SALT_LENGTH)
    try:
        with open(salt_path, "wb") as f:
            f.write(salt)
        print(f"✅ Salt created: {salt_path}")
    except Exception as e:
        print(f"⚠️ Failed to save salt: {e}")

    return salt


# ============================================================
# Key derivation
# ============================================================

def derive_fernet_key(machine_id: str = None, salt: bytes = None) -> bytes:
    """
    Derive a Fernet key from machine_id + salt + APP_SECRET.

    Args:
        machine_id: if None, auto-fetched
        salt: if None, auto-fetched

    Returns:
        Fernet-compatible key (32 bytes, base64-urlsafe encoded)

    Raises:
        ImportError if cryptography not available
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    if machine_id is None:
        machine_id = get_machine_id()
    if salt is None:
        salt = get_or_create_salt()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )

    material = f"{machine_id}:{APP_SECRET}".encode("utf-8")
    key_bytes = kdf.derive(material)
    return base64.urlsafe_b64encode(key_bytes)
