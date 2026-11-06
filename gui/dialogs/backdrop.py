# gui/dialogs/backdrop.py
# -*- coding: utf-8 -*-
"""
Opaque backdrop for dialogs.

Prevents background content from showing through translucent dialogs.
"""

from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QColor, QPainter


class DialogBackdrop(QWidget):
    """
    A fullscreen-ish opaque backdrop that appears behind a dialog.

    - Covers the parent widget (or entire screen)
    - Darkened + slightly blurred look
    - Fade-in / fade-out animation
    - Auto-closes when dialog closes

    Usage:
        backdrop = DialogBackdrop(parent_widget, theme, opacity=0.85)
        backdrop.show_backdrop()
        dialog = MyDialog(parent=backdrop)  # dialog on top of backdrop
        dialog.exec_()
        backdrop.hide_backdrop()
    """

    def __init__(self, parent: QWidget, theme: dict, opacity: float = 0.85):
        super().__init__(parent)

        # ✅ Make sure backdrop is on top of everything in the parent
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setVisible(False)

        self._theme = theme
        self._opacity = opacity

        # ✅ Fade effect
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(
            self._opacity_effect, b"opacity", self
        )
        self._fade_anim.setDuration(220)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        # ✅ Track parent resize
        if parent is not None:
            parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Keep backdrop full-size when parent resizes."""
        from PyQt5.QtCore import QEvent
        if obj is self.parent() and event.type() == QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def _composite_color(self) -> QColor:
        """
        Build the opaque backdrop color.

        Strategy:
        - Take theme's bg_gradient_end (darkest color) as base
        - Blend with black to make it more opaque/darker
        - Use `opacity` factor for alpha
        """
        t = self._theme
        try:
            base_hex = t.get("bg_gradient_start", "#0f172a")
            base = QColor(base_hex)
            # Darken it a bit more
            r = int(base.red() * 0.6)
            g = int(base.green() * 0.6)
            b = int(base.blue() * 0.6)
            return QColor(r, g, b, int(self._opacity * 255))
        except Exception:
            return QColor(10, 10, 20, int(self._opacity * 255))

    def paintEvent(self, event):
        if not self.isVisible():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self._composite_color())
        painter.end()

    def show_backdrop(self):
        """Show with fade-in animation."""
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())
        self.setVisible(True)
        self.raise_()

        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def hide_backdrop(self, on_finished=None):
        """Hide with fade-out animation."""
        if not self.isVisible():
            if on_finished:
                try:
                    on_finished()
                except Exception:
                    pass
            return

        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except TypeError:
            pass
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)

        def _on_done():
            try:
                self._fade_anim.finished.disconnect(_on_done)
            except TypeError:
                pass
            self.setVisible(False)
            if on_finished:
                try:
                    on_finished()
                except Exception:
                    pass

        self._fade_anim.finished.connect(_on_done)
        self._fade_anim.start()


def center_on_parent(dialog: QWidget, parent: QWidget):
    """Center a dialog on its parent widget."""
    try:
        if parent is None:
            return
        pg = parent.geometry()
        dg = dialog.geometry()
        x = pg.x() + (pg.width() - dg.width()) // 2
        y = pg.y() + (pg.height() - dg.height()) // 2
        dialog.move(x, y)
    except Exception:
        pass


def center_on_screen(dialog: QWidget):
    """Center a dialog on the primary screen."""
    try:
        screen = QApplication.primaryScreen().geometry()
        dg = dialog.geometry()
        x = (screen.width() - dg.width()) // 2
        y = (screen.height() - dg.height()) // 2
        dialog.move(x, y)
    except Exception:
        pass
