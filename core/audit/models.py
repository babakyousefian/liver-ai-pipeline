# core/audit/models.py
# -*- coding: utf-8 -*-
"""
Data models for the audit log system.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
import uuid


class AuditLevel(str, Enum):
    """Severity levels for audit events."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditCategory(str, Enum):
    """Categories for grouping audit events."""
    AUTH = "AUTH"           # login, signup, logout, lock, unlock
    DATA = "DATA"           # inference, save, clear
    SETTINGS = "SETTINGS"   # theme, language, glassmorphism
    SECURITY = "SECURITY"   # rate limit, invalid password
    SYSTEM = "SYSTEM"       # app start/close, update, crash
    EXPORT = "EXPORT"       # PDF, Grad-CAM, image save


@dataclass
class AuditEntry:
    """
    A single audit log entry.

    Example:
        {
            "id": "abc-123",
            "timestamp": "2026-11-04T14:30:00.123456",
            "level": "INFO",
            "category": "AUTH",
            "action": "login_success",
            "user": "babak",
            "session_id": "xyz-789",
            "details": {"os": "Windows 10", "extra": "..."}
        }
    """
    action: str
    category: str
    level: str = AuditLevel.INFO.value
    user: str = ""
    session_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        """Create from dict (safe — ignore unknown keys)."""
        allowed = {
            "id", "timestamp", "level", "category", "action",
            "user", "session_id", "details",
        }
        filtered = {k: v for k, v in data.items() if k in allowed}
        # ✅ Ensure details is a dict
        if "details" in filtered and not isinstance(filtered["details"], dict):
            filtered["details"] = {}
        return cls(**filtered)

    @property
    def display_time(self) -> str:
        """Human-readable time (HH:MM:SS)."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%H:%M:%S")
        except Exception:
            return "—"

    @property
    def display_date(self) -> str:
        """Human-readable date (YYYY-MM-DD)."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "—"

    @property
    def display_datetime(self) -> str:
        """Full datetime (YYYY-MM-DD HH:MM:SS)."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "—"


# ✅ Helper: level → emoji
LEVEL_EMOJI = {
    AuditLevel.INFO.value: "ℹ️",
    AuditLevel.WARNING.value: "⚠️",
    AuditLevel.ERROR.value: "❌",
    AuditLevel.CRITICAL.value: "🚨",
}

# ✅ Helper: category → emoji
CATEGORY_EMOJI = {
    AuditCategory.AUTH.value: "🔐",
    AuditCategory.DATA.value: "📊",
    AuditCategory.SETTINGS.value: "⚙️",
    AuditCategory.SECURITY.value: "🛡️",
    AuditCategory.SYSTEM.value: "🖥️",
    AuditCategory.EXPORT.value: "📄",
}

# ✅ Helper: level → color (for UI)
LEVEL_COLORS = {
    AuditLevel.INFO.value: "#00e5ff",
    AuditLevel.WARNING.value: "#fbbf24",
    AuditLevel.ERROR.value: "#f87171",
    AuditLevel.CRITICAL.value: "#dc2626",
}
