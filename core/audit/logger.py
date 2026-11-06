# core/audit/logger.py
# -*- coding: utf-8 -*-
"""
Thread-safe singleton audit logger with buffered writes.

Usage:
    from core.audit import audit_log

    audit_log.info(
        action="login_success",
        category="AUTH",
        user="babak",
        details={"os": "Windows 10"},
    )
"""

import threading
import uuid
from typing import Dict, Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from .models import AuditEntry, AuditLevel, AuditCategory
from .storage import AuditStorage


class AuditLogger(QObject):
    """
    Singleton audit logger with:
    - Thread-safe operations (threading.Lock)
    - Buffered writes (flush every N entries or every X seconds)
    - Session tracking (session_id auto-generated per login)
    - Signals for UI updates
    """

    # ✅ Signals
    entry_added = pyqtSignal(object)  # AuditEntry

    # ✅ Buffer settings
    FLUSH_INTERVAL_MS = 5000  # هر ۵ ثانیه flush
    FLUSH_THRESHOLD = 10  # یا هر ۱۰ رویداد

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self._storage = AuditStorage()
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._current_user = ""
        self._current_session_id = ""

        # ✅ Timer for periodic flush
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self.flush)
        self._flush_timer.start(self.FLUSH_INTERVAL_MS)

    @classmethod
    def get(cls) -> "AuditLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AuditLogger()
        return cls._instance

    # ============================================================
    # Session management
    # ============================================================
    def set_user(self, username: str):
        """Set current user (call on login). Generates new session_id."""
        self._current_user = username or ""
        self._current_session_id = str(uuid.uuid4()) if username else ""

    def clear_user(self):
        """Clear current user (call on logout)."""
        self._current_user = ""
        self._current_session_id = ""

    # ============================================================
    # Log methods
    # ============================================================
    def log(
            self,
            action: str,
            category: str = AuditCategory.SYSTEM.value,
            level: str = AuditLevel.INFO.value,
            user: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
    ):
        """Log a single event."""
        try:
            entry = AuditEntry(
                action=action,
                category=category,
                level=level,
                user=user if user is not None else self._current_user,
                session_id=self._current_session_id,
                details=details or {},
            )
            self._enqueue(entry)
        except Exception as e:
            print(f"⚠️ AuditLogger.log failed: {e}")

    def info(self, action: str, category: str = "SYSTEM",
             user: Optional[str] = None, details: Optional[Dict] = None):
        self.log(action, category, AuditLevel.INFO.value, user, details)

    def warning(self, action: str, category: str = "SYSTEM",
                user: Optional[str] = None, details: Optional[Dict] = None):
        self.log(action, category, AuditLevel.WARNING.value, user, details)

    def error(self, action: str, category: str = "SYSTEM",
              user: Optional[str] = None, details: Optional[Dict] = None):
        self.log(action, category, AuditLevel.ERROR.value, user, details)

    def critical(self, action: str, category: str = "SECURITY",
                 user: Optional[str] = None, details: Optional[Dict] = None):
        self.log(action, category, AuditLevel.CRITICAL.value, user, details)

    # ============================================================
    # Buffer management
    # ============================================================
    def _enqueue(self, entry: AuditEntry):
        """Add entry to buffer; flush if threshold reached."""
        with self._buffer_lock:
            self._buffer.append(entry)
            should_flush = len(self._buffer) >= self.FLUSH_THRESHOLD

        # ✅ Emit signal immediately (for UI updates)
        try:
            self.entry_added.emit(entry)
        except Exception:
            pass

        if should_flush:
            self.flush()

    def flush(self):
        """Write buffer to disk."""
        with self._buffer_lock:
            if not self._buffer:
                return
            to_write = self._buffer
            self._buffer = []

        try:
            self._storage.append_entries(to_write)
        except Exception as e:
            print(f"⚠️ AuditLogger.flush failed: {e}")

    # ============================================================
    # Read
    # ============================================================
    def read_all(self) -> list:
        """Read all entries from disk."""
        return self._storage.load_entries()

    def clear(self):
        """Clear all entries (main file)."""
        with self._buffer_lock:
            self._buffer = []
        self._storage.clear_entries()

    # ============================================================
    # Export
    # ============================================================
    def export_csv(self, output_path: str, entries=None) -> bool:
        """Export entries to CSV."""
        try:
            import csv
            if entries is None:
                entries = self.read_all()

            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Level", "Category", "Action",
                    "User", "Session ID", "Details",
                ])
                for e in entries:
                    writer.writerow([
                        e.display_datetime,
                        e.level,
                        e.category,
                        e.action,
                        e.user,
                        e.session_id,
                        json_safe_details(e.details),
                    ])
            return True
        except Exception as e:
            print(f"⚠️ AuditLogger.export_csv failed: {e}")
            return False

    def export_json(self, output_path: str, entries=None) -> bool:
        """Export entries to JSON."""
        try:
            import json
            if entries is None:
                entries = self.read_all()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in entries], f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"⚠️ AuditLogger.export_json failed: {e}")
            return False


def json_safe_details(details: Dict) -> str:
    """Convert details dict to a short JSON string for CSV."""
    try:
        import json
        return json.dumps(details, ensure_ascii=False, default=str)[:500]
    except Exception:
        return ""


# ✅ Singleton access
def get_audit_logger() -> AuditLogger:
    return AuditLogger.get()
