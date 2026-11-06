# core/audit/storage.py
# -*- coding: utf-8 -*-
"""
JSON storage for audit log entries.

- Main file: `audit_log.json` (last N entries)
- Daily files: `audit_logs/audit_YYYY-MM-DD.json` (archived)
- Rotation: entries older than `MAX_ENTRIES` are moved to daily files
- Cleanup: daily files older than `RETENTION_DAYS` are deleted
"""

import os
import sys
import json
import threading
from datetime import datetime, timedelta
from typing import List, Optional

from .models import AuditEntry


# ============================================================
# Constants
# ============================================================

MAX_ENTRIES = 1000            # حداکثر تعداد رویداد در فایل اصلی
RETENTION_DAYS = 30           # حداکثر عمر فایل‌های روزانه
MAIN_FILENAME = "audit_log.json"
ARCHIVE_DIRNAME = "audit_logs"


# ============================================================
# Paths
# ============================================================

def _get_project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # core/audit/storage.py → core/audit/ → core/ → project_root
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


# ============================================================
# Storage
# ============================================================

class AuditStorage:
    """
    Thread-safe JSON storage for audit log entries.

    - Main file: audit_log.json (list of entries)
    - Daily files: audit_logs/audit_YYYY-MM-DD.json
    """

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = base_dir or _get_project_root()
        self._main_path = os.path.join(self._base_dir, MAIN_FILENAME)
        self._archive_dir = os.path.join(self._base_dir, ARCHIVE_DIRNAME)
        self._lock = threading.Lock()

        # ✅ Ensure archive dir exists
        try:
            os.makedirs(self._archive_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ AuditStorage: cannot create archive dir: {e}")

        # ✅ Cleanup old archives (best effort, silent)
        self._cleanup_old_archives()

    # ============================================================
    # Read
    # ============================================================
    def load_entries(self) -> List[AuditEntry]:
        """Load all entries from main file."""
        with self._lock:
            return self._load_entries_unlocked()

    def _load_entries_unlocked(self) -> List[AuditEntry]:
        if not os.path.isfile(self._main_path):
            return []
        try:
            with open(self._main_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ AuditStorage: failed to load: {e}")
            return []

        if not isinstance(data, list):
            return []

        entries = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(AuditEntry.from_dict(item))
            except Exception:
                continue
        return entries

    # ============================================================
    # Write
    # ============================================================
    def append_entries(self, new_entries: List[AuditEntry]):
        """
        Append new entries to main file.
        If file exceeds MAX_ENTRIES, oldest are moved to daily archive.
        """
        if not new_entries:
            return
        with self._lock:
            entries = self._load_entries_unlocked()
            entries.extend(new_entries)

            # ✅ Rotate if too many
            if len(entries) > MAX_ENTRIES:
                overflow = len(entries) - MAX_ENTRIES
                to_archive = entries[:overflow]
                entries = entries[overflow:]
                self._archive_entries_unlocked(to_archive)

            # ✅ Write main file atomically
            self._write_main_unlocked(entries)

    def _write_main_unlocked(self, entries: List[AuditEntry]):
        """Write entries to main JSON file atomically."""
        try:
            tmp = self._main_path + ".tmp"
            data = [e.to_dict() for e in entries]
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._main_path)
        except Exception as e:
            print(f"⚠️ AuditStorage: failed to write main: {e}")

    def _archive_entries_unlocked(self, entries: List[AuditEntry]):
        """Append entries to today's archive file."""
        if not entries:
            return
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            archive_path = os.path.join(self._archive_dir, f"audit_{today}.json")

            # ✅ Load existing archive (if any)
            existing = []
            if os.path.isfile(archive_path):
                try:
                    with open(archive_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []

            existing.extend([e.to_dict() for e in entries])

            # ✅ Write back
            tmp = archive_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            os.replace(tmp, archive_path)
        except Exception as e:
            print(f"⚠️ AuditStorage: failed to archive: {e}")

    # ============================================================
    # Clear
    # ============================================================
    def clear_entries(self):
        """Clear main file (keeps archives)."""
        with self._lock:
            try:
                if os.path.isfile(self._main_path):
                    os.remove(self._main_path)
            except Exception as e:
                print(f"⚠️ AuditStorage: failed to clear: {e}")

    # ============================================================
    # Cleanup
    # ============================================================
    def _cleanup_old_archives(self):
        """Delete archive files older than RETENTION_DAYS."""
        try:
            if not os.path.isdir(self._archive_dir):
                return
            cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
            for fname in os.listdir(self._archive_dir):
                if not fname.startswith("audit_") or not fname.endswith(".json"):
                    continue
                try:
                    date_str = fname[len("audit_"):-len(".json")]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        os.remove(os.path.join(self._archive_dir, fname))
                        print(f"🗑️ AuditStorage: removed old archive: {fname}")
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ AuditStorage: cleanup failed: {e}")