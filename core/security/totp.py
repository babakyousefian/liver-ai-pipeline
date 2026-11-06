# core/security/totp.py
# -*- coding: utf-8 -*-
"""
TOTP-based Two-Factor Authentication.

Standards:
- RFC 6238 (TOTP) [citation:3]
- RFC 4226 (HOTP) [citation:3]

Security:
- valid_window=1 (±30s) — استاندارد تحمل انحراف ساعت [citation:13]
- replay protection با last_used_step [citation:16]
- backup codes یک‌بارمصرف با hash [citation:1][citation:20]
"""

import os
import io
import hmac
import hashlib
import secrets
import base64
from datetime import datetime, timezone
from typing import Optional, List, Tuple

# ✅ Optional imports (graceful degradation)
try:
    import pyotp
    HAS_PYOTP = True
except ImportError:
    HAS_PYOTP = False
    pyotp = None

try:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_M
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
    qrcode = None


# ============================================================
# Constants
# ============================================================

ISSUER_NAME = "Liver AI Pipeline"
TOTP_DIGITS = 6
TOTP_INTERVAL = 30          # seconds — RFC 6238 default
TOTP_VALID_WINDOW = 1       # ±1 step = ±30s tolerance
BACKUP_CODE_COUNT = 10
BACKUP_CODE_LENGTH = 8      # hex chars (4 bytes)


# ============================================================
# Secret generation
# ============================================================

def generate_secret() -> str:
    """
    Generate a random base32 secret compatible with
    Google Authenticator, Authy, Microsoft Authenticator.

    Returns:
        32-character base32 string (160 bits entropy)
    """
    if not HAS_PYOTP:
        raise RuntimeError(
            "pyotp not installed. Run: pip install pyotp"
        )
    return pyotp.random_base32()


# ============================================================
# QR code generation
# ============================================================

def build_provisioning_uri(secret: str, username: str, email: str = "") -> str:
    """
    Build otpauth:// URI for authenticator apps.

    Format:
        otpauth://totp/Liver%20AI%20Pipeline:babak?secret=XXX&issuer=...
    """
    if not HAS_PYOTP:
        raise RuntimeError("pyotp not installed")
    totp = pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL)
    account_name = email or username
    return totp.provisioning_uri(
        name=account_name,
        issuer_name=ISSUER_NAME,
    )


def generate_qr_pixmap(provisioning_uri: str, size: int = 300):
    """
    Generate a QR code that phone cameras can reliably scan.

    Uses QImage + manual pixel drawing (no PIL scaling blur).
    FastTransformation preserves crisp black/white modules.
    """
    if not HAS_QRCODE:
        return None
    try:
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import Qt

        qr = qrcode.QRCode(
            version=None,
            error_correction=ERROR_CORRECT_M,
            box_size=1,          # 1 pixel per module in memory
            border=4,            # QR standard quiet zone
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        matrix = qr.get_matrix()      # list[list[bool]]
        module_count = len(matrix)

        # ✅ Draw directly as QImage — no PIL, no blur
        img = QImage(module_count, module_count, QImage.Format_RGB888)
        img.fill(Qt.white)
        black = 0x000000
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    img.setPixel(x, y, black)

        pixmap = QPixmap.fromImage(img)
        if size > 0:
            pixmap = pixmap.scaled(
                size, size,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,   # ← crisp edges, no interpolation
            )
        return pixmap
    except Exception as e:
        print(f"⚠️ generate_qr_pixmap failed: {e}")
        return None


# ============================================================
# TOTP verification (with replay protection)
# ============================================================

def verify_code(
    secret: str,
    code: str,
    last_used_step: Optional[int] = None,
) -> Tuple[bool, Optional[int]]:
    """
    Verify a TOTP code with replay protection.

    Args:
        secret: base32 secret
        code: 6-digit string entered by user
        last_used_step: last accepted time-step (for replay prevention)

    Returns:
        (is_valid, matched_step)
        - matched_step is None if invalid
        - caller MUST store matched_step to prevent replay [citation:16]
    """
    if not HAS_PYOTP:
        return False, None
    if not code:
        return False, None

    code = code.strip().replace(" ", "").replace("-", "")
    if len(code) != TOTP_DIGITS or not code.isdigit():
        return False, None

    totp = pyotp.TOTP(secret, digits=TOTP_DIGITS, interval=TOTP_INTERVAL)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    current_step = now_ts // TOTP_INTERVAL

    # ✅ Check each step in window individually for replay safety
    for offset in range(-TOTP_VALID_WINDOW, TOTP_VALID_WINDOW + 1):
        candidate_step = current_step + offset
        at_time = datetime.fromtimestamp(
            candidate_step * TOTP_INTERVAL, tz=timezone.utc
        )
        expected = totp.at(at_time)

        # ✅ Constant-time comparison (prevent timing attacks) [citation:13]
        if hmac.compare_digest(expected, code):
            # ✅ Replay protection: reject already-used or older steps
            if last_used_step is not None and candidate_step <= last_used_step:
                return False, None
            return True, candidate_step

    return False, None


# ============================================================
# Backup codes
# ============================================================

def generate_backup_codes(count: int = BACKUP_CODE_COUNT) -> List[str]:
    """
    Generate one-time backup codes.

    Format: XXXX-XXXX (8 hex chars with dash)
    Example: A3F8-K2M1
    """
    codes = []
    for _ in range(count):
        raw = secrets.token_hex(BACKUP_CODE_LENGTH // 2).upper()
        codes.append(f"{raw[:4]}-{raw[4:]}")
    return codes


def hash_backup_codes(codes: List[str]) -> List[str]:
    """
    Hash backup codes before storing.
    Codes are recovery credentials — treat like passwords [citation:5].
    """
    hashed = []
    for code in codes:
        normalized = code.replace("-", "").upper()
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        hashed.append(h)
    return hashed


def verify_backup_code(code: str, hashed_codes: List[str]) -> Tuple[bool, Optional[int]]:
    """
    Verify a backup code against hashed list.

    Returns:
        (is_valid, index_of_matched_code)
        - index is None if invalid
        - caller MUST remove that index from stored list (single-use) [citation:1]
    """
    normalized = (code or "").replace("-", "").replace(" ", "").upper()
    if not normalized:
        return False, None

    target_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    for i, stored in enumerate(hashed_codes):
        if hmac.compare_digest(stored, target_hash):
            return True, i
    return False, None
