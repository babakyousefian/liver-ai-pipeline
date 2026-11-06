# gui/widgets/password_strength_meter.py
# -*- coding: utf-8 -*-
"""
PasswordStrengthMeter — real-time password strength widget.

Features:
- Animated progress bar (color changes with strength)
- Level label (Very Weak / Weak / ...)
- Rules checklist (✓ / ✗ per rule)
- Feedback messages (e.g. "Add 4 more digits")
- Theme-aware
- RTL-aware
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsOpacityEffect,
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QColor


def _get_helpers():
    from GUI_update import tr, THEMES
    from core.security import calculate_strength, LEVEL_EMOJI, LEVEL_PROGRESS
    return {
        "tr": tr,
        "THEMES": THEMES,
        "calculate_strength": calculate_strength,
        "LEVEL_EMOJI": LEVEL_EMOJI,
        "LEVEL_PROGRESS": LEVEL_PROGRESS,
    }


# ============================================================
# Constants
# ============================================================

# ✅ Level → color key in theme (or literal hex)
LEVEL_COLOR_KEYS = {
    "very_weak": "danger",     # red
    "weak": "warning",         # yellow/orange
    "medium": "#fbbf24",       # literal hex (zard)
    "strong": "success",       # green
    "very_strong": "primary",  # theme primary (neon)
}

RTL_LANGS = {"Persian", "Arabic", "Morocco"}


class PasswordStrengthMeter(QWidget):
    """
    A widget that shows password strength in real-time.

    Usage:
        meter = PasswordStrengthMeter(theme=THEMES["theme_dark_aqua"], lang="English")
        meter.set_password("MyPass123!@#")
        # or connect to QLineEdit:
        line_edit.textChanged.connect(meter.set_password)
    """

    ANIMATION_MS = 250

    def __init__(self, theme: dict = None, lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._h = h
        self._tr = h["tr"]
        self._theme = theme or h["THEMES"].get("theme_dark_aqua", {})
        self._lang = lang
        self._current_password = ""
        self._current_result = None

        # ✅ Animation state
        self._current_progress = 0.0
        self._target_progress = 0.0
        self._current_color = QColor("#00e5ff")

        self._anim = QPropertyAnimation(self, b"dummy")  # placeholder
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(16)  # ~60 FPS
        self._tick_timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_theme()
        self._refresh()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(6)

        # ---------- Progress bar (track + fill) ----------
        self._track = QFrame()
        self._track.setObjectName("psm_track")
        self._track.setFixedHeight(8)

        self._fill = QFrame(self._track)
        self._fill.setObjectName("psm_fill")
        self._fill.setGeometry(0, 0, 0, 8)

        main.addWidget(self._track)

        # ---------- Status row (emoji + level + feedback) ----------
        status_row = QHBoxLayout()
        status_row.setSpacing(6)

        self._emoji_label = QLabel("")
        self._emoji_label.setObjectName("psm_emoji")
        self._emoji_label.setFixedWidth(20)
        status_row.addWidget(self._emoji_label)

        self._level_label = QLabel("")
        self._level_label.setObjectName("psm_level")
        status_row.addWidget(self._level_label)

        status_row.addStretch()

        self._feedback_label = QLabel("")
        self._feedback_label.setObjectName("psm_feedback")
        self._feedback_label.setWordWrap(True)
        self._feedback_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self._feedback_label, 1)

        main.addLayout(status_row)

        # ---------- Rules checklist ----------
        self._rules_container = QWidget()
        self._rules_layout = QVBoxLayout(self._rules_container)
        self._rules_layout.setContentsMargins(0, 4, 0, 0)
        self._rules_layout.setSpacing(2)

        self._rule_labels = {}  # key -> QLabel
        for key in ("length", "lowercase", "uppercase", "digits", "special"):
            lbl = QLabel("")
            lbl.setObjectName("psm_rule")
            self._rule_labels[key] = lbl
            self._rules_layout.addWidget(lbl)

        main.addWidget(self._rules_container)

    def _apply_theme(self):
        t = self._theme
        # ✅ Colors
        track_bg = t.get("glass_highlight", "rgba(0, 229, 255, 0.10)")
        if "theme_solarized_white" == t.get("name", "").lower().replace(" ", "_"):
            track_bg = "rgba(38, 139, 210, 0.18)"

        self._track.setStyleSheet(f"""
            QFrame#psm_track {{
                background: {track_bg};
                border: 1px solid {t.get('glass_border', '#00e5ff')};
                border-radius: 4px;
            }}
        """)
        self._emoji_label.setStyleSheet("background: transparent; font-size: 14px;")
        self._level_label.setStyleSheet(f"""
            QLabel#psm_level {{
                color: {t.get('text', '#f1f5f9')};
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            }}
        """)
        self._feedback_label.setStyleSheet(f"""
            QLabel#psm_feedback {{
                color: {t.get('text_dim', '#94a3b8')};
                font-size: 10px;
                background: transparent;
            }}
        """)
        for lbl in self._rule_labels.values():
            lbl.setStyleSheet(f"""
                QLabel#psm_rule {{
                    color: {t.get('text_dim', '#94a3b8')};
                    font-size: 10px;
                    background: transparent;
                }}
            """)

    # ============================================================
    # Public API
    # ============================================================
    def set_theme(self, theme: dict):
        self._theme = theme
        self._apply_theme()
        self._refresh()

    def set_lang(self, lang: str):
        self._lang = lang
        self._refresh()

    def set_password(self, password: str):
        """Update meter with new password."""
        self._current_password = password or ""
        self._refresh()

    def clear(self):
        self.set_password("")

    def is_valid(self) -> bool:
        """True if all rules pass."""
        return bool(self._current_result and self._current_result.is_valid)

    # ============================================================
    # Internals
    # ============================================================
    def _refresh(self):
        try:
            # ✅ Calculate strength
            result = self._h["calculate_strength"](self._current_password)
            self._current_result = result

            # ✅ Determine target color
            color = self._get_level_color(result.level)

            # ✅ Set emoji + level text
            self._emoji_label.setText(result.emoji)
            self._level_label.setText(self._tr(self._lang, f"strength_{result.level}"))

            # ✅ Feedback
            feedback_text = self._format_feedback(result.feedback)
            self._feedback_label.setText(feedback_text)
            self._feedback_label.setVisible(bool(feedback_text))

            # ✅ Rules checklist
            self._update_rules(result)

            # ✅ Animate progress
            self._target_progress = result.progress / 100.0
            self._current_color = color
            self._tick_timer.start()
        except Exception as e:
            print(f"⚠️ PasswordStrengthMeter._refresh failed: {e}")

    def _get_level_color(self, level: str) -> QColor:
        color_key = LEVEL_COLOR_KEYS.get(level, "primary")
        t = self._theme
        if color_key.startswith("#"):
            return QColor(color_key)
        hex_color = t.get(color_key, "#00e5ff")
        return QColor(hex_color)

    def _format_feedback(self, feedback: list) -> str:
        """Convert feedback list to human-readable string."""
        if not feedback:
            return ""
        parts = []
        for item in feedback[:2]:  # max 2 items
            try:
                if ":" in item:
                    key, arg = item.split(":", 1)
                    if key == "length_short":
                        parts.append(
                            self._tr(self._lang, "strength_add_chars").format(n=arg)
                        )
                    elif key == "length_long":
                        parts.append(
                            self._tr(self._lang, "strength_remove_chars").format(n=arg)
                        )
                    elif key == "need_digits":
                        parts.append(
                            self._tr(self._lang, "strength_need_digits").format(n=arg)
                        )
                    elif key == "need_special":
                        parts.append(
                            self._tr(self._lang, "strength_need_special").format(n=arg)
                        )
                    elif key == "common_pattern":
                        parts.append(
                            self._tr(self._lang, "strength_common_pattern")
                        )
                    elif key == "common_year":
                        parts.append(
                            self._tr(self._lang, "strength_common_year")
                        )
                else:
                    tr_key = f"strength_{item}"
                    translated = self._tr(self._lang, tr_key)
                    if translated != tr_key:
                        parts.append(translated)
            except Exception:
                continue
        return " • ".join(parts)

    def _update_rules(self, result):
        """Update rules checklist labels."""
        try:
            tr = self._tr
            lang = self._lang
            t = self._theme
            success_color = t.get("success", "#34d399")
            danger_color = t.get("danger", "#f87171")

            # ✅ Map rule keys to translation keys
            rule_keys_map = {
                "length": "strength_rule_length",
                "lowercase": "strength_rule_lowercase",
                "uppercase": "strength_rule_uppercase",
                "digits": "strength_rule_digits",
                "special": "strength_rule_special",
            }

            for rule in result.rules:
                lbl = self._rule_labels.get(rule.key)
                if lbl is None:
                    continue

                # ✅ Display text
                tr_key = rule_keys_map.get(rule.key, rule.key)
                base_text = tr(lang, tr_key)
                if base_text == tr_key:
                    base_text = rule.key.title()

                # ✅ Show count if required > 1
                if rule.required > 1:
                    display = f"{base_text} ({rule.current}/{rule.required})"
                else:
                    display = base_text

                # ✅ Check mark
                mark = "✓" if rule.passed else "✗"
                color = success_color if rule.passed else danger_color
                lbl.setText(f"{mark}  {display}")
                lbl.setStyleSheet(f"""
                    QLabel#psm_rule {{
                        color: {color};
                        font-size: 10px;
                        background: transparent;
                    }}
                """)

            # ✅ If all rules pass, collapse checklist (less noise)
            self._rules_container.setVisible(not result.is_valid)
        except Exception as e:
            print(f"⚠️ _update_rules failed: {e}")

    def _on_tick(self):
        """Animation tick — move current_progress toward target."""
        try:
            diff = self._target_progress - self._current_progress
            if abs(diff) < 0.005:
                self._current_progress = self._target_progress
                self._tick_timer.stop()
            else:
                self._current_progress += diff * 0.2  # easing
            self._update_fill()
        except Exception:
            self._tick_timer.stop()

    def _update_fill(self):
        """Redraw fill bar."""
        try:
            w = self._track.width()
            h = self._track.height()
            fill_w = max(0, int(w * self._current_progress))
            self._fill.setGeometry(0, 0, fill_w, h)
            c = self._current_color
            self._fill.setStyleSheet(f"""
                QFrame#psm_fill {{
                    background: {c.name()};
                    border-radius: 3px;
                }}
            """)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_fill()

    # ✅ Dummy property (for QPropertyAnimation compat, if needed later)
    def _dummy(self):
        return 0

    def _set_dummy(self, v):
        pass

    dummy = property(_dummy, _set_dummy)
