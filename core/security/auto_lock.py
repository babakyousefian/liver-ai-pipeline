# core/security/auto_lock.py
# -*- coding: utf-8 -*-
"""
Auto-lock after inactivity.

Design:
- QTimer (single-shot) resets on every real user activity
- Distinguishes user-activity from programmatic events
- Pauses during modal dialogs (but NOT lock screen)
- Configurable timeout (5 min - 60 min, or never)
- Emits signal on timeout → MainWindow shows LockScreen
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

from PyQt5.QtCore import QObject, QTimer, QEvent, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication


# ============================================================
# Config (persisted to settings.json)
# ============================================================

@dataclass
class AutoLockConfig:
    enabled: bool = True
    timeout_minutes: int = 2           # 5 min default
    lock_on_minimize: bool = False     # lock when window minimized
    lock_on_sleep: bool = False        # lock on system sleep (best-effort)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AutoLockConfig":
        obj = cls()
        for k, v in (data or {}).items():
            if hasattr(obj, k) and v is not None:
                setattr(obj, k, v)
        return obj


def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))


def load_config() -> AutoLockConfig:
    path = os.path.join(_project_root(), "settings.json")
    if not os.path.isfile(path):
        cfg = AutoLockConfig()
        cfg.timeout_minutes = 2
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AutoLockConfig.from_dict(data.get("auto_lock", {}))

        # ✅ Validate timeout: must be 1-60 min, else default to 2
        try:
            t = int(cfg.timeout_minutes)
            if t < 1 or t > 60:
                t = 2
            cfg.timeout_minutes = t
        except Exception:
            cfg.timeout_minutes = 2

        return cfg
    except Exception:
        cfg = AutoLockConfig()
        cfg.timeout_minutes = 2
        return cfg


def save_config(cfg: AutoLockConfig) -> bool:
    path = os.path.join(_project_root(), "settings.json")
    try:
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["auto_lock"] = cfg.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"⚠️ save auto_lock config failed: {e}")
        return False


# ============================================================
# Manager
# ============================================================

class AutoLockManager(QObject):
    """
    Manages inactivity detection + lock trigger.

    Signals:
        timeout_warning(seconds_left): called at 60s, 30s, 10s remaining
        timeout_reached(): called when timeout expires
    """

    timeout_warning = pyqtSignal(int)   # seconds_left
    timeout_reached = pyqtSignal()

    # ✅ Events that count as activity
    ACTIVITY_EVENTS = {
        QEvent.MouseButtonPress,
        QEvent.MouseButtonRelease,
        QEvent.MouseButtonDblClick,
        QEvent.KeyPress,
        QEvent.KeyRelease,
        QEvent.Wheel,
    }

    WARNING_THRESHOLDS = [60, 30, 10]   # seconds before timeout

    def __init__(self, config: AutoLockConfig = None, parent=None):
        super().__init__(parent)
        self._config = config or AutoLockConfig()
        self._is_active = False
        self._is_paused = False
        self._lock_screen_open = False
        self._last_activity = 0.0
        self._warned = set()

        # Main timer — single-shot
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        # Warning timer — fires every 1s to check remaining
        self._warn_timer = QTimer(self)
        self._warn_timer.setInterval(1000)
        self._warn_timer.timeout.connect(self._check_warnings)

        # Install global event filter
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ============================================================
    # Public API
    # ============================================================
    def start(self):
        """Start monitoring (call on login)."""
        if not self._config.enabled:
            return
        if self._is_active:
            return
        self._is_active = True
        self._is_paused = False
        self._last_activity = self._now()
        self._warned.clear()
        timeout_ms = max(60_000, self._config.timeout_minutes * 60_000)
        self._timer.start(timeout_ms)
        self._warn_timer.start()
        print(f"✅ AutoLock started ({self._config.timeout_minutes} min)")

    def stop(self):
        """Stop monitoring (call on logout)."""
        self._is_active = False
        self._is_paused = False
        self._timer.stop()
        self._warn_timer.stop()
        self._warned.clear()

    def pause(self):
        """Pause (e.g., while lock screen is open)."""
        if not self._is_active or self._is_paused:
            return
        self._is_paused = True
        self._timer.stop()
        self._warn_timer.stop()

    def resume(self):
        """Resume after pause."""
        if not self._is_active or not self._is_paused:
            return
        self._is_paused = False
        self._last_activity = self._now()
        self._warned.clear()
        timeout_ms = max(60_000, self._config.timeout_minutes * 60_000)
        self._timer.start(timeout_ms)
        self._warn_timer.start()

    def reset(self):
        """Reset timer (called on user activity)."""
        if not self._is_active or self._is_paused:
            return
        self._last_activity = self._now()
        self._warned.clear()
        timeout_ms = max(60_000, self._config.timeout_minutes * 60_000)
        self._timer.start(timeout_ms)

    def set_config(self, config: AutoLockConfig):
        """Update config (e.g., from Settings)."""
        self._config = config
        if self._is_active and not self._is_paused:
            self.reset()

    def set_lock_screen_open(self, open_: bool):
        """Called by MainWindow when lock screen opens/closes."""
        self._lock_screen_open = open_
        if open_:
            self.pause()
        else:
            # Re-activate after unlock
            if self._is_active:
                self.resume()

    def remaining_seconds(self) -> int:
        """Remaining seconds until lock (or -1 if inactive)."""
        if not self._is_active or self._is_paused:
            return -1
        ms = self._timer.remainingTime()
        if ms < 0:
            return -1
        return ms // 1000

    def is_active(self) -> bool:
        return self._is_active and not self._is_paused

    # ============================================================
    # Internals
    # ============================================================
    @staticmethod
    def _now() -> float:
        return time.time()

    def _on_timeout(self):
        if not self._is_active or self._is_paused:
            return
        self._warn_timer.stop()
        print("⏰ AutoLock timeout reached")
        self.timeout_reached.emit()

    def _check_warnings(self):
        """Check if we need to emit warning."""
        if not self._is_active or self._is_paused:
            return
        secs = self.remaining_seconds()
        if secs < 0:
            return
        for threshold in self.WARNING_THRESHOLDS:
            if secs <= threshold and threshold not in self._warned:
                self._warned.add(threshold)
                self.timeout_warning.emit(secs)
                break

    def eventFilter(self, obj, event):
        """Global event filter — resets timer on real user activity."""
        if not self._is_active or self._is_paused:
            return False
        if self._lock_screen_open:
            return False

        et = event.type()

        if et in self.ACTIVITY_EVENTS:
            self.reset()
            return False

        # MouseMove only counts if dragging (button pressed)
        if et == QEvent.MouseMove:
            try:
                if event.buttons() != Qt.NoButton:
                    self.reset()
            except Exception:
                pass
            return False

        return False
