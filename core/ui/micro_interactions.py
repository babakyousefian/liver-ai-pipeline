# core/ui/micro_interactions.py
# -*- coding: utf-8 -*-
"""
Micro-interactions — small, polished animations that give
users feedback on every action.

All functions respect `QApplication`'s reduced-motion setting
(if the user has disabled animations system-wide, they are skipped).

Design principles:
- Fast (150-400ms)
- Subtle (not distracting)
- Respect user preferences
- Fail gracefully (never crash the app)
"""

import math
from typing import Optional, Callable

from PyQt5.QtWidgets import (
    QWidget, QLineEdit, QLabel, QPushButton, QScrollArea,
    QGraphicsOpacityEffect, QApplication,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve,
    QPoint, QPointF, QRect, QParallelAnimationGroup,
    QSequentialAnimationGroup, QObject, pyqtSignal, QSize,
)
from PyQt5.QtGui import QColor


# ============================================================
# ✅ Reduced motion support
# ============================================================

class MotionSettings:
    """
    Tracks whether the user wants animations.

    Sources (checked in order):
    1. Explicit override (from Settings)
    2. OS-level "reduce motion" (best-effort)
    """
    _instance = None
    _reduced = False
    _explicit: Optional[bool] = None

    @classmethod
    def get(cls) -> "MotionSettings":
        if cls._instance is None:
            cls._instance = MotionSettings()
            cls._instance._detect_os_preference()
        return cls._instance

    def _detect_os_preference(self):
        """Try to detect OS-level reduce-motion."""
        try:
            import sys
            if sys.platform.startswith("win"):
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Control Panel\Accessibility",
                )
                try:
                    val, _ = winreg.QueryValueEx(key, "Animation")
                    # 0 = disabled, 1 = enabled
                    self._reduced = (val == "0" or val == 0)
                except Exception:
                    pass
                winreg.CloseKey(key)
        except Exception:
            pass

    def set_reduced(self, reduced: bool):
        self._explicit = reduced
        self._reduced = reduced

    def is_reduced(self) -> bool:
        if self._explicit is not None:
            return self._explicit
        return self._reduced


def _motion_ok() -> bool:
    """True if animations should play."""
    try:
        return not MotionSettings.get().is_reduced()
    except Exception:
        return True


# ============================================================
# ✅ 1. SHAKE ON ERROR
# ============================================================

def shake_widget(
    widget: QWidget,
    duration_ms: int = 400,
    intensity: int = 8,
    oscillations: int = 4,
):
    """
    Shake a widget horizontally — used for invalid input.

    Example: wrong password → field shakes
    """
    if not _motion_ok():
        return
    try:
        original_pos = widget.pos()

        # ✅ Build a key-frame sequence
        group = QSequentialAnimationGroup(widget)

        for i in range(oscillations):
            offset = intensity if (i % 2 == 0) else -intensity
            offset *= (1.0 - i / max(1, oscillations))

            anim = QPropertyAnimation(widget, b"pos", widget)
            anim.setDuration(duration_ms // oscillations)
            anim.setStartValue(widget.pos())
            anim.setEndValue(QPoint(
                original_pos.x() + int(offset),
                original_pos.y(),
            ))
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            group.addAnimation(anim)

        # ✅ Return to original
        anim_back = QPropertyAnimation(widget, b"pos", widget)
        anim_back.setDuration(duration_ms // oscillations)
        anim_back.setStartValue(widget.pos())
        anim_back.setEndValue(original_pos)
        anim_back.setEasingCurve(QEasingCurve.OutQuad)
        group.addAnimation(anim_back)

        widget._shake_group = group
        group.start()
    except Exception as e:
        print(f"⚠️ shake_widget failed: {e}")


# ============================================================
# ✅ 2. FIELD FOCUS GLOW
# ============================================================

def add_focus_glow(widget: QLineEdit, theme: dict):
    """
    Add a subtle glow animation on focus.

    This changes the border color smoothly (via QSS is not
    animatable, so we use a QGraphicsDropShadowEffect).
    """
    try:
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect

        if not hasattr(widget, "_focus_glow_effect"):
            effect = QGraphicsDropShadowEffect(widget)
            effect.setBlurRadius(0)
            effect.setColor(QColor(0, 0, 0, 0))
            effect.setOffset(0, 0)
            widget.setGraphicsEffect(effect)
            widget._focus_glow_effect = effect

            anim = QPropertyAnimation(effect, b"blurRadius", widget)
            anim.setDuration(250)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            widget._focus_glow_anim = anim

        primary = QColor(theme.get("primary", "#00e5ff"))
        effect = widget._focus_glow_effect
        anim = widget._focus_glow_anim

        def _on_focus_in(event):
            if _motion_ok():
                effect.setColor(QColor(
                    primary.red(), primary.green(), primary.blue(), 180
                ))
                anim.stop()
                anim.setStartValue(effect.blurRadius())
                anim.setEndValue(16)
                anim.start()
            QLineEdit.focusInEvent(widget, event)

        def _on_focus_out(event):
            if _motion_ok():
                anim.stop()
                anim.setStartValue(effect.blurRadius())
                anim.setEndValue(0)
                anim.start()
            QLineEdit.focusOutEvent(widget, event)

        widget.focusInEvent = _on_focus_in
        widget.focusOutEvent = _on_focus_out
    except Exception as e:
        print(f"⚠️ add_focus_glow failed: {e}")


# ============================================================
# ✅ 3. SUCCESS CHECK OVERLAY
# ============================================================

def show_success_check(
    parent: QWidget,
    color: str = "#34d399",
    duration_ms: int = 1200,
    size: int = 64,
):
    """
    Show a green ✓ overlay in the center of `parent`.

    Example: after saving account info, show a ✓.
    """
    if not _motion_ok():
        return
    try:
        check = QLabel(parent)
        check.setText("✓")
        check.setAlignment(Qt.AlignCenter)
        check.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: {int(size * 0.6)}px;
                font-weight: bold;
                background: transparent;
                border: 3px solid {color};
                border-radius: {size // 2}px;
            }}
        """)
        check.setFixedSize(size, size)
        check.move(
            (parent.width() - size) // 2,
            (parent.height() - size) // 2,
        )
        check.raise_()

        # ✅ Pop-in
        effect = QGraphicsOpacityEffect(check)
        effect.setOpacity(0.0)
        check.setGraphicsEffect(effect)

        anim_in = QPropertyAnimation(effect, b"opacity", check)
        anim_in.setDuration(180)
        anim_in.setStartValue(0.0)
        anim_in.setEndValue(1.0)
        anim_in.setEasingCurve(QEasingCurve.OutBack)

        # ✅ Scale-in
        scale_anim = QPropertyAnimation(check, b"geometry", check)
        start_rect = QRect(
            (parent.width() - size // 2) // 2,
            (parent.height() - size // 2) // 2,
            size // 2, size // 2,
        )
        end_rect = QRect(
            (parent.width() - size) // 2,
            (parent.height() - size) // 2,
            size, size,
        )
        scale_anim.setDuration(250)
        scale_anim.setStartValue(start_rect)
        scale_anim.setEndValue(end_rect)
        scale_anim.setEasingCurve(QEasingCurve.OutBack)

        check.setGeometry(start_rect)

        group = QParallelAnimationGroup(check)
        group.addAnimation(anim_in)
        group.addAnimation(scale_anim)
        check._group = group
        check._effect = effect
        group.start()

        # ✅ Fade out + delete
        def _fade_out():
            try:
                anim_out = QPropertyAnimation(effect, b"opacity", check)
                anim_out.setDuration(300)
                anim_out.setStartValue(1.0)
                anim_out.setEndValue(0.0)
                anim_out.finished.connect(check.deleteLater)
                check._anim_out = anim_out
                anim_out.start()
            except Exception:
                try:
                    check.deleteLater()
                except Exception:
                    pass

        QTimer.singleShot(duration_ms, _fade_out)
    except Exception as e:
        print(f"⚠️ show_success_check failed: {e}")


# ============================================================
# ✅ 4. COUNT-UP (animated number)
# ============================================================

def animate_count_up(
    label: QLabel,
    from_value: float,
    to_value: float,
    duration_ms: int = 600,
    decimals: int = 0,
    suffix: str = "",
):
    """
    Animate a number from `from_value` to `to_value` in `label`.

    Example: dashboard "Total: 42" → "Total: 45" counts up smoothly.
    """
    if not _motion_ok():
        try:
            fmt = f"{{:.{decimals}f}}"
            label.setText(fmt.format(to_value) + suffix)
        except Exception:
            pass
        return
    try:
        # ✅ Use a QObject holder with a dynamic property
        #    (simpler: use QTimer at 16ms with easing math)
        start_time = [0]  # mutable for closure
        steps = max(1, duration_ms // 16)
        current_step = [0]

        def _tick():
            current_step[0] += 1
            t = min(1.0, current_step[0] / steps)
            # ✅ Ease-out cubic
            eased = 1.0 - (1.0 - t) ** 3
            val = from_value + (to_value - from_value) * eased
            try:
                fmt = f"{{:.{decimals}f}}"
                label.setText(fmt.format(val) + suffix)
            except RuntimeError:
                timer.stop()
                return
            if t >= 1.0:
                timer.stop()
                try:
                    fmt = f"{{:.{decimals}f}}"
                    label.setText(fmt.format(to_value) + suffix)
                except Exception:
                    pass

        timer = QTimer(label)
        timer.setInterval(16)
        timer.timeout.connect(_tick)
        label._count_up_timer = timer
        timer.start()
    except Exception as e:
        print(f"⚠️ animate_count_up failed: {e}")
        try:
            fmt = f"{{:.{decimals}f}}"
            label.setText(fmt.format(to_value) + suffix)
        except Exception:
            pass


# ============================================================
# ✅ 5. BUTTON PRESS SCALE
# ============================================================

def button_press_scale(
    button: QPushButton,
    pressed_scale: float = 0.96,
    duration_ms: int = 100,
):
    """
    Scale the button down when pressed, back to 1.0 on release.

    This is like iOS buttons — subtle feedback.
    """
    if not _motion_ok():
        return
    try:
        if hasattr(button, "_press_scale_setup"):
            return
        button._press_scale_setup = True

        original_geom = [None]  # mutable for closure

        def _on_press(event):
            original_geom[0] = QRect(button.geometry())
            g = button.geometry()
            new_w = int(g.width() * pressed_scale)
            new_h = int(g.height() * pressed_scale)
            offset_x = (g.width() - new_w) // 2
            offset_y = (g.height() - new_h) // 2

            anim = QPropertyAnimation(button, b"geometry", button)
            anim.setDuration(duration_ms)
            anim.setStartValue(g)
            anim.setEndValue(QRect(
                g.x() + offset_x,
                g.y() + offset_y,
                new_w, new_h,
            ))
            anim.setEasingCurve(QEasingCurve.OutQuad)
            anim.start()
            button._press_anim = anim
            QPushButton.mousePressEvent(button, event)

        def _on_release(event):
            if original_geom[0] is not None:
                anim = QPropertyAnimation(button, b"geometry", button)
                anim.setDuration(duration_ms + 50)
                anim.setStartValue(button.geometry())
                anim.setEndValue(original_geom[0])
                anim.setEasingCurve(QEasingCurve.OutBack)
                anim.start()
                button._release_anim = anim
                original_geom[0] = None
            QPushButton.mouseReleaseEvent(button, event)

        button.mousePressEvent = _on_press
        button.mouseReleaseEvent = _on_release
    except Exception as e:
        print(f"⚠️ button_press_scale failed: {e}")


# ============================================================
# ✅ 6. SMOOTH SCROLL
# ============================================================

def smooth_scroll_to(
    scroll_area: QScrollArea,
    target_value: int,
    duration_ms: int = 300,
):
    """Smoothly scroll a QScrollArea to a value."""
    if not _motion_ok():
        try:
            scroll_area.verticalScrollBar().setValue(target_value)
        except Exception:
            pass
        return
    try:
        sb = scroll_area.verticalScrollBar()
        if sb is None:
            return
        current = sb.value()
        if current == target_value:
            return

        anim = QPropertyAnimation(sb, b"value", scroll_area)
        anim.setDuration(duration_ms)
        anim.setStartValue(current)
        anim.setEndValue(target_value)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        scroll_area._smooth_scroll_anim = anim
    except Exception as e:
        print(f"⚠️ smooth_scroll_to failed: {e}")


# ============================================================
# ✅ 7. BOUNCE IN (for toasts)
# ============================================================

def bounce_in(
    widget: QWidget,
    start_pos: QPoint,
    end_pos: QPoint,
    duration_ms: int = 350,
):
    """Bounce-in animation for toasts — overshoot then settle."""
    if not _motion_ok():
        try:
            widget.move(end_pos)
        except Exception:
            pass
        return
    try:
        widget.setVisible(True)
        widget.move(start_pos)
        widget.raise_()

        anim = QPropertyAnimation(widget, b"pos", widget)
        anim.setDuration(duration_ms)
        anim.setStartValue(start_pos)
        anim.setEndValue(end_pos)
        anim.setEasingCurve(QEasingCurve.OutBack)
        anim.start()
        widget._bounce_in_anim = anim
    except Exception as e:
        print(f"⚠️ bounce_in failed: {e}")


# ============================================================
# ✅ 8. COLOR PULSE (for auto-lock countdown)
# ============================================================

def color_pulse(
    widget: QWidget,
    base_style_fn: Callable[[float], str],
    duration_ms: int = 600,
    repeats: int = 2,
):
    """
    Pulse the widget's color.

    `base_style_fn(t)` should return a stylesheet string,
    where `t` is a phase from 0.0 to 1.0.

    Example:
        color_pulse(
            my_label,
            lambda t: f"color: rgba(255, {int(180 + 75*t)}, 100, 1);"
        )
    """
    if not _motion_ok():
        return
    try:
        step_ms = 30
        total_steps = max(1, duration_ms // step_ms)
        current = [0]

        def _tick():
            current[0] += 1
            t = (current[0] / total_steps) % 1.0
            # ✅ Sine wave
            phase = (1 + math.sin(t * 2 * math.pi - math.pi / 2)) / 2
            try:
                widget.setStyleSheet(base_style_fn(phase))
            except RuntimeError:
                timer.stop()
                return
            if current[0] >= total_steps * repeats:
                timer.stop()
                # Reset to base
                try:
                    widget.setStyleSheet(base_style_fn(1.0))
                except Exception:
                    pass

        timer = QTimer(widget)
        timer.setInterval(step_ms)
        timer.timeout.connect(_tick)
        widget._pulse_timer = timer
        timer.start()
    except Exception as e:
        print(f"⚠️ color_pulse failed: {e}")


# ============================================================
# ✅ 9. PAGE TRANSITION (fade + slide + scale)
# ============================================================

def page_transition_in(
    widget: QWidget,
    direction: str = "up",   # "up" | "down" | "left" | "right"
    duration_ms: int = 300,
):
    """
    Animate a widget entering: fade + slide from `direction`.
    """
    if not _motion_ok():
        widget.setVisible(True)
        return
    try:
        final_geom = QRect(widget.geometry())

        offset_map = {
            "up": QPoint(0, 30),
            "down": QPoint(0, -30),
            "left": QPoint(30, 0),
            "right": QPoint(-30, 0),
        }
        offset = offset_map.get(direction, QPoint(0, 30))

        start_geom = QRect(
            final_geom.x() + offset.x(),
            final_geom.y() + offset.y(),
            final_geom.width(),
            final_geom.height(),
        )

        # ✅ Opacity
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        anim_opacity = QPropertyAnimation(effect, b"opacity", widget)
        anim_opacity.setDuration(duration_ms)
        anim_opacity.setStartValue(0.0)
        anim_opacity.setEndValue(1.0)
        anim_opacity.setEasingCurve(QEasingCurve.OutCubic)

        anim_pos = QPropertyAnimation(widget, b"geometry", widget)
        anim_pos.setDuration(duration_ms)
        anim_pos.setStartValue(start_geom)
        anim_pos.setEndValue(final_geom)
        anim_pos.setEasingCurve(QEasingCurve.OutCubic)

        widget.setGeometry(start_geom)
        widget.setVisible(True)

        group = QParallelAnimationGroup(widget)
        group.addAnimation(anim_opacity)
        group.addAnimation(anim_pos)
        group.start()

        # ✅ Remove effect when done
        def _cleanup():
            try:
                widget.setGraphicsEffect(None)
            except Exception:
                pass
        anim_opacity.finished.connect(_cleanup)

        widget._page_transition = group
        widget._page_transition_opacity = anim_opacity
    except Exception as e:
        print(f"⚠️ page_transition_in failed: {e}")
        widget.setVisible(True)


# ============================================================
# ✅ 10. HELPERS: decorate input / button
# ============================================================

def decorate_input(widget: QLineEdit, theme: dict):
    """Add focus glow to a QLineEdit."""
    add_focus_glow(widget, theme)


def decorate_button_press(widget: QPushButton):
    """Add press scale to any QPushButton."""
    button_press_scale(widget)


def highlight_widget_error(widget: QWidget, theme: dict):
    """Shake + red border on error."""
    shake_widget(widget)
    try:
        error_color = theme.get("danger", "#f87171")
        widget.setStyleSheet(
            widget.styleSheet()
            + f"\nQLineEdit {{ border: 2px solid {error_color}; }}"
        )
        # Revert after 2s
        def _revert():
            try:
                # Remove the injected border
                s = widget.styleSheet()
                s = s.replace(f"\nQLineEdit {{ border: 2px solid {error_color}; }}", "")
                widget.setStyleSheet(s)
            except Exception:
                pass
        QTimer.singleShot(2000, _revert)
    except Exception as e:
        print(f"⚠️ highlight_widget_error failed: {e}")


def highlight_widget_success(widget: QWidget, parent_for_check: QWidget = None, theme: dict = None):
    """Show a ✓ overlay in the parent widget."""
    parent = parent_for_check or widget.window()
    color = "#34d399"
    if theme:
        color = theme.get("success", "#34d399")
    show_success_check(parent, color=color)

