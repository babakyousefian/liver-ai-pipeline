# core/audit/__init__.py
# -*- coding: utf-8 -*-

from .models import (
    AuditEntry, AuditLevel, AuditCategory,
    LEVEL_EMOJI, CATEGORY_EMOJI, LEVEL_COLORS,
)
from .logger import AuditLogger, get_audit_logger
from .storage import AuditStorage
from .reader import read_filtered

# ✅ Global singleton
audit_log = AuditLogger.get()

__all__ = [
    "AuditEntry", "AuditLevel", "AuditCategory",
    "LEVEL_EMOJI", "CATEGORY_EMOJI", "LEVEL_COLORS",
    "AuditLogger", "get_audit_logger",
    "AuditStorage", "read_filtered",
    "audit_log",
]