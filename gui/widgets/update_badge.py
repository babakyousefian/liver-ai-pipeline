# gui/widgets/update_badge.py
# -*- coding: utf-8 -*-
"""
Small 🔴 badge overlay for the update button.
"""

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor, QGraphicsOpacityEffect


class UpdateBadge(QLabel):
    """Animated red dot badge."""

    def __init__(self, parent, theme: dict):
        super().__init__(parent)
        self._theme = theme
        self.setFixedSize(12, 12)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._apply_style()
        self.setVisible(False)

        # Pulse animation
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(1.0)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.35)
        self._anim.setEasingCurve(QEasingCurve.InOutSine)
        self._anim.setLoopCount(-1)

    def _apply_style(self):
        danger = self._theme.get("danger", "#f87171")
        self.setStyleSheet(f"""
            QLabel {{
                background: {danger};
                border-radius: 6px;
                border: 2px solid rgba(255,255,255,0.9);
            }}
        """)

    def set_theme(self, theme: dict):
        self._theme = theme
        self._apply_style()

    def show_badge(self):
        self.setVisible(True)
        self.raise_()
        if self._anim.state() != QPropertyAnimation.Running:
            self._anim.start()

    def hide_badge(self):
        self._anim.stop()
        self.setVisible(False)

    def reposition(self, parent_widget):
        """Call this after parent resizes."""
        try:
            x = parent_widget.width() - self.width() - 2
            y = 2
            self.move(x, y)
        except Exception:
            pass