# core/updater/checker.py
# -*- coding: utf-8 -*-
"""
UpdateChecker: QThread that fetches latest version and compares.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime, timedelta
from typing import Optional

from .models import UpdateInfo, UpdateConfig
from .source import make_source
from .version import is_newer


CURRENT_VERSION = "3.0.0"  # ← should match your app version


class UpdateChecker(QThread):
    """
    Signals:
        update_available(UpdateInfo)
        up_to_date()
        check_failed(str reason)
    """
    update_available = pyqtSignal(object)  # UpdateInfo
    up_to_date = pyqtSignal()
    check_failed = pyqtSignal(str)

    def __init__(self, config: UpdateConfig, current_version: str = CURRENT_VERSION,
                 force: bool = False, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_version = current_version
        self.force = force  # if True, ignore interval + skipped_version

    def run(self):
        try:
            # ✅ Respect interval (unless forced)
            if not self.force and not self._should_check_now():
                self.up_to_date.emit()
                return

            source = make_source(self.config)
            info = source.fetch_latest(include_prerelease=self.config.include_prerelease)

            if info is None:
                self.check_failed.emit("fetch_failed")
                return

            # ✅ Update last_check timestamp
            self.config.last_check = datetime.now().isoformat()

            # ✅ Compare
            if not is_newer(self.current_version, info.version):
                self.up_to_date.emit()
                return

            # ✅ Respect "skip this version" (unless forced or critical)
            if (not self.force and not info.is_critical
                    and info.version == self.config.skipped_version):
                self.up_to_date.emit()
                return

            self.update_available.emit(info)

        except Exception as e:
            print(f"⚠️ UpdateChecker failed: {e}")
            self.check_failed.emit(str(e))

    def _should_check_now(self) -> bool:
        """Return True if enough time has passed since last check."""
        if not self.config.last_check:
            return True
        try:
            last = datetime.fromisoformat(self.config.last_check)
            interval = timedelta(hours=max(1, self.config.check_interval_hours))
            return datetime.now() >= last + interval
        except Exception:
            return True