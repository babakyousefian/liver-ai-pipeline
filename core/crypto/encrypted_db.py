# core/crypto/encrypted_db.py
# -*- coding: utf-8 -*-
"""
Encrypted JSON store.

Provides a drop-in replacement for raw JSON file I/O:
    store = EncryptedJSONStore("users_db.json")
    data = store.load()        # decrypts
    store.save(data)           # encrypts
"""

import os
import json
import shutil
from datetime import datetime

from .key_manager import (
    is_cryptography_available,
    derive_fernet_key,
)


# ============================================================
# Constants
# ============================================================

# Header for encrypted files (12 bytes)
# If the first bytes match this, we know it's encrypted.
_ENCRYPTED_MAGIC = b"LIVERAI_ENC1"
_ENCRYPTED_EXT = ".enc"


# ============================================================
# Helpers
# ============================================================

def is_encrypted_file(path: str) -> bool:
    """Check if a file starts with our encryption magic header."""
    try:
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as f:
            header = f.read(len(_ENCRYPTED_MAGIC))
            return header == _ENCRYPTED_MAGIC
    except Exception:
        return False


def _backup_plain_file(path: str):
    """Create a .json.plain.bak backup of the original plain file."""
    try:
        if not os.path.isfile(path):
            return
        backup_path = path + ".plain.bak"
        if os.path.isfile(backup_path):
            return  # already backed up
        shutil.copy2(path, backup_path)
        print(f"📦 Plain backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️ Failed to backup plain file: {e}")


def _backup_encrypted_file(path: str):
    """Create a .json.bak backup of the current file."""
    try:
        if not os.path.isfile(path):
            return
        backup_path = path + ".bak"
        shutil.copy2(path, backup_path)
    except Exception as e:
        print(f"⚠️ Failed to backup file: {e}")


# ============================================================
# EncryptedJSONStore
# ============================================================

class EncryptedJSONStore:
    """
    Drop-in replacement for raw JSON file I/O with encryption.

    Usage:
        store = EncryptedJSONStore("users_db.json")
        data = store.load()      # returns dict (decrypts if needed)
        store.save(data)         # writes encrypted

    Behavior:
        - If file doesn't exist → returns {} (or default)
        - If file is plaintext JSON → loads it, and on next save it will be encrypted
        - If file is encrypted → decrypts
        - If `cryptography` not available → falls back to plaintext with warning
    """

    def __init__(self, file_path: str, default: dict = None):
        self.file_path = file_path
        self.default = default if default is not None else {"users": {}}
        self._crypto_available = is_cryptography_available()
        self._fernet = None  # lazy-loaded

        if not self._crypto_available:
            print(
                "⚠️ EncryptedJSONStore: `cryptography` not installed.\n"
                "   → Files will be stored as PLAINTEXT.\n"
                "   → Install: pip install cryptography"
            )

    # ============================================================
    # Fernet (lazy)
    # ============================================================
    def _get_fernet(self):
        if self._fernet is None:
            from cryptography.fernet import Fernet
            key = derive_fernet_key()
            self._fernet = Fernet(key)
        return self._fernet

    # ============================================================
    # Load
    # ============================================================
    def load(self) -> dict:
        """Load and decrypt (if needed) the JSON file."""
        if not os.path.isfile(self.file_path):
            return dict(self.default)

        # ✅ Check if encrypted
        if is_encrypted_file(self.file_path):
            if not self._crypto_available:
                print(
                    f"❌ File is encrypted but `cryptography` not installed: "
                    f"{self.file_path}"
                )
                return dict(self.default)
            return self._load_encrypted()

        # ✅ Plaintext
        return self._load_plain()

    def _load_plain(self) -> dict:
        """Load a plaintext JSON file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ Loaded plaintext: {self.file_path}")
            return data
        except Exception as e:
            print(f"⚠️ Failed to load plaintext: {e}")
            return dict(self.default)

    def _load_encrypted(self) -> dict:
        """Load and decrypt an encrypted file."""
        try:
            with open(self.file_path, "rb") as f:
                raw = f.read()

            # ✅ Strip magic header
            if not raw.startswith(_ENCRYPTED_MAGIC):
                raise ValueError("Missing encryption magic header")
            payload = raw[len(_ENCRYPTED_MAGIC):]

            # ✅ Decrypt
            fernet = self._get_fernet()
            decrypted = fernet.decrypt(payload)

            # ✅ Parse JSON
            data = json.loads(decrypted.decode("utf-8"))
            print(f"🔓 Loaded encrypted: {self.file_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to decrypt {self.file_path}: {e}")
            print(
                "   Possible reasons:\n"
                "   - File was encrypted on a different machine\n"
                "   - Salt file (users_db.salt) was deleted or changed\n"
                "   - Machine ID file (users_db.machine_id) was deleted or changed\n"
                "   - File was tampered with"
            )
            return dict(self.default)

    # ============================================================
    # Save
    # ============================================================
    def save(self, data: dict):
        """Save the data (encrypted if crypto available)."""
        if not self._crypto_available:
            self._save_plain(data)
            return

        self._save_encrypted(data)

    def _save_plain(self, data: dict):
        """Fallback: save as plaintext."""
        tmp = self.file_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp, self.file_path)
            print(f"💾 Saved plaintext: {self.file_path}")
        except Exception as e:
            print(f"⚠️ Failed to save plaintext: {e}")

    def _save_encrypted(self, data: dict):
        """Save as encrypted."""
        # ✅ Backup plain if exists (one-time)
        if os.path.isfile(self.file_path) and not is_encrypted_file(self.file_path):
            _backup_plain_file(self.file_path)

        # ✅ Backup current encrypted
        if os.path.isfile(self.file_path) and is_encrypted_file(self.file_path):
            _backup_encrypted_file(self.file_path)

        try:
            # ✅ Serialize to JSON bytes
            json_bytes = json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8")

            # ✅ Encrypt
            fernet = self._get_fernet()
            encrypted = fernet.encrypt(json_bytes)

            # ✅ Write with magic header
            tmp = self.file_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(_ENCRYPTED_MAGIC)
                f.write(encrypted)
            os.replace(tmp, self.file_path)
            print(f"🔒 Saved encrypted: {self.file_path}")
        except Exception as e:
            print(f"❌ Failed to save encrypted: {e}")
            # ✅ Fallback to plain (so user doesn't lose data)
            try:
                self._save_plain(data)
                print("⚠️ Fallback to plaintext save succeeded")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
