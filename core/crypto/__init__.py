# core/crypto/__init__.py
# -*- coding: utf-8 -*-

from .key_manager import (
    get_machine_id,
    get_or_create_salt,
    derive_fernet_key,
    is_cryptography_available,
)
from .encrypted_db import (
    EncryptedJSONStore,
    is_encrypted_file,
)

__all__ = [
    "get_machine_id",
    "get_or_create_salt",
    "derive_fernet_key",
    "is_cryptography_available",
    "EncryptedJSONStore",
    "is_encrypted_file",
]