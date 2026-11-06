# gui/widgets/session_timeout.py
# -*- coding: utf-8 -*-
"""
Session timeout manager.

Detects user inactivity and triggers a lock screen after N seconds.

Architecture:
    - QTimer (single-shot, 2 min) resets on every user activity
    - Event filter on QApplication captures user events
    - Pauses while a modal dialog is open
    - Only active when a user is logged in
"""

from PyQt5.QtCore import QObject, QTimer, QEvent, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication


class SessionTimeoutManager(QObject):
    """
    Manages session timeout detection.

    Signals:
        timeout_reached(): emitted when user has been idle for `timeout_ms`

    Usage:
        manager = SessionTimeoutManager(timeout_ms=120_000)
        manager.timeout_reached.connect(self._lock_session)
        manager.start()  # called after login
        manager.stop()   # called on logout
    """

    timeout_reached = pyqtSignal()

    # Events that count as user activity
    _ACTIVITY_EVENTS = {
        QEvent.MouseButtonPress,
        QEvent.MouseButtonRelease,
        QEvent.MouseButtonDblClick,
        QEvent.KeyPress,
        QEvent.KeyRelease,
        QEvent.Wheel,
        # MouseMove is special — only counts if dragging
    }

    def __init__(self, timeout_ms: int = 120_000, parent=None):
        super().__init__(parent)
        self._timeout_ms = timeout_ms
        self._is_active = False
        self._is_paused = False
        self._last_activity_time = 0.0

        # ✅ Single-shot timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self._timeout_ms)
        self._timer.timeout.connect(self._on_timeout)

        # ✅ Install global event filter
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ============================================================
    # Public API
    # ============================================================
    def start(self):
        """Start the session timeout timer (call after login)."""
        if self._is_active:
            return
        self._is_active = True
        self._is_paused = False
        self._last_activity_time = self._now()
        self._timer.start(self._timeout_ms)
        print(f"✅ Session timeout started ({self._timeout_ms / 1000:.0f}s)")

    def stop(self):
        """Stop the session timeout timer (call on logout)."""
        if not self._is_active:
            return
        self._is_active = False
        self._is_paused = False
        self._timer.stop()
        print("⏹️ Session timeout stopped")

    def pause(self):
        """Pause the timer (e.g., when a modal dialog is open)."""
        if not self._is_active or self._is_paused:
            return
        self._is_paused = True
        self._timer.stop()
        print("⏸️ Session timeout paused")

    def resume(self):
        """Resume the timer after a pause."""
        if not self._is_active or not self._is_paused:
            return
        self._is_paused = False
        self._last_activity_time = self._now()
        self._timer.start(self._timeout_ms)
        print("▶️ Session timeout resumed")

    def reset(self):
        """Reset the timer (called on any user activity)."""
        if not self._is_active or self._is_paused:
            return
        self._last_activity_time = self._now()
        self._timer.start(self._timeout_ms)

    def is_active(self) -> bool:
        return self._is_active

    def is_paused(self) -> bool:
        return self._is_paused

    def set_timeout(self, timeout_ms: int):
        """Change the timeout interval."""
        self._timeout_ms = max(10_000, timeout_ms)  # min 10s
        if self._is_active and not self._is_paused:
            self._timer.start(self._timeout_ms)

    def remaining_ms(self) -> int:
        """Return the remaining time in ms (or -1 if inactive)."""
        if not self._is_active:
            return -1
        if self._is_paused:
            return self._timeout_ms
        return self._timer.remainingTime()

    # ============================================================
    # Internals
    # ============================================================
    @staticmethod
    def _now() -> float:
        import time
        return time.time()

    def _on_timeout(self):
        """Timer fired → emit signal."""
        if not self._is_active or self._is_paused:
            return
        print("⏰ Session timeout reached!")
        self.timeout_reached.emit()

    def eventFilter(self, obj, event):
        """Global event filter — captures user activity."""
        if not self._is_active or self._is_paused:
            return False

        et = event.type()

        # ✅ Regular activity events
        if et in self._ACTIVITY_EVENTS:
            self.reset()
            return False

        # ✅ MouseMove only counts as activity if dragging
        if et == QEvent.MouseMove:
            try:
                # event.buttons() returns Qt.MouseButtons
                # If any button is held, it's a drag
                if event.buttons() != Qt.NoButton:
                    self.reset()
            except Exception:
                pass
            return False

        return False  # don't consume events


# ============================================================
# ✅ Modal detection helper
# ============================================================

class ModalWatcher(QObject):
    """
    Watches for modal dialogs opening/closing and pauses/resumes
    the session timeout manager accordingly.

    QApplication has no direct signal for modal state changes,
    so we use a lightweight polling timer (200ms).
    """

    def __init__(self, manager: SessionTimeoutManager, poll_ms: int = 200,
                 parent=None):
        super().__init__(parent)
        self._manager = manager
        self._last_modal_state = False

        self._poll = QTimer(self)
        self._poll.setInterval(poll_ms)
        self._poll.timeout.connect(self._check)
        self._poll.start()

    def _check(self):
        try:
            app = QApplication.instance()
            if app is None:
                return
            modal = app.activeModalWidget()
            is_modal = modal is not None and modal is not app.activeWindow()
            # If a modal is open AND it's not the lock screen itself
            # (lock screen is handled separately, not by manager)
            if is_modal != self._last_modal_state:
                self._last_modal_state = is_modal
                if is_modal:
                    self._manager.pause()
                else:
                    self._manager.resume()
        except Exception as e:
            print(f"⚠️ ModalWatcher._check failed: {e}")

    def stop(self):
        if self._poll.isActive():
            self._poll.stop()
