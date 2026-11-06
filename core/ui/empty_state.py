# core/ui/empty_state.py
# -*- coding: utf-8 -*-
"""
Beautiful Empty States — وقتی چیزی نیست، پیام زیبا نشون بده.

Features:
- Animated emoji (pulse)
- Multi-line HTML (title + subtitle + hint)
- Theme-aware colors
- RTL aware
- Optional CTA button

Usage:
    from core.ui.empty_state import make_empty_state_html, EmptyStateWidget

    # For QLabel
    label.setText(make_empty_state_html("history", theme, lang))

    # For full widget with animation
    widget = EmptyStateWidget("history", theme, lang)
    widget.clicked.connect(...)  # if has CTA
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Callable, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame,
    QGraphicsOpacityEffect, QSizePolicy,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal, QRect,
)
from PyQt5.QtGui import QFont, QColor


# ============================================================
# ✅ Emoji + Style presets for each empty state
# ============================================================

@dataclass
class EmptyStateStyle:
    """Style for an empty state."""
    emoji: str  # big emoji
    pulse: bool = True  # animate emoji
    accent_key: str = "primary"  # color key from theme
    emoji_size: int = 64  # px
    title_size: int = 18
    subtitle_size: int = 12
    hint_size: int = 11


EMPTY_STATE_STYLES: Dict[str, EmptyStateStyle] = {
    # ============================================================
    # History Panel — No history yet
    # ============================================================
    "history_empty": EmptyStateStyle(
        emoji="🕐",
        pulse=True,
        accent_key="primary",
        emoji_size=72,
        title_size=18,
    ),
    # ============================================================
    # History Panel — Not logged in
    # ============================================================
    "history_no_user": EmptyStateStyle(
        emoji="🔐",
        pulse=False,
        accent_key="secondary",
        emoji_size=72,
        title_size=16,
    ),
    # ============================================================
    # Dashboard — No results yet
    # ============================================================
    "dash_no_results": EmptyStateStyle(
        emoji="📊",
        pulse=True,
        accent_key="primary",
        emoji_size=56,
        title_size=14,
    ),
    # ============================================================
    # Dashboard — No distribution data
    # ============================================================
    "dash_no_dist": EmptyStateStyle(
        emoji="📈",
        pulse=False,
        accent_key="text_dim",
        emoji_size=40,
        title_size=12,
    ),
    # ============================================================
    # Image View — No image inserted
    # ============================================================
    "image_empty": EmptyStateStyle(
        emoji="🖼️",
        pulse=True,
        accent_key="primary",
        emoji_size=96,
        title_size=22,
    ),
    # ============================================================
    # Image View — Drop prompt
    # ============================================================
    "image_drop_hint": EmptyStateStyle(
        emoji="📥",
        pulse=True,
        accent_key="secondary",
        emoji_size=64,
        title_size=16,
    ),
    # ============================================================
    # Processing — Waiting
    # ============================================================
    "processing_wait": EmptyStateStyle(
        emoji="⏳",
        pulse=True,
        accent_key="warning",
        emoji_size=48,
        title_size=14,
    ),
}


# ============================================================
# ✅ Translation helper (lazy import)
# ============================================================

def _tr(lang: str, key: str) -> str:
    """Lazy translation."""
    try:
        from GUI_update import tr as _tr_fn
        return _tr_fn(lang, key)
    except Exception:
        return key


def _get_theme(theme: Optional[Dict] = None) -> Dict:
    """Lazy theme getter."""
    if theme is not None:
        return theme
    try:
        from GUI_update import THEMES
        return THEMES.get("theme_dark_aqua", {})
    except Exception:
        return {
            "primary": "#00e5ff",
            "secondary": "#c084fc",
            "text": "#f1f5f9",
            "text_dim": "#94a3b8",
            "success": "#34d399",
            "warning": "#fbbf24",
            "danger": "#f87171",
        }


# ============================================================
# ✅ HTML builder (for QLabel)
# ============================================================

def make_empty_state_html(
        key: str,
        theme: Dict,
        lang: str = "English",
        custom_title: str = "",
        custom_subtitle: str = "",
        custom_hint: str = "",
        show_emoji: bool = True,
) -> str:
    """
    Build an HTML string for an empty state.

    Args:
        key: empty state key (e.g. "history_empty", "dash_no_results")
        theme: THEMES dict
        lang: current language
        custom_title: override title (optional)
        custom_subtitle: override subtitle (optional)
        custom_hint: override hint (optional)
        show_emoji: include emoji in HTML

    Returns:
        HTML string suitable for QLabel.setText() with RichText
    """
    try:
        style = EMPTY_STATE_STYLES.get(key, EmptyStateStyle(emoji="✨"))

        # ✅ رنگ accent
        accent = theme.get(style.accent_key, theme.get("primary", "#00e5ff"))
        text = theme.get("text", "#f1f5f9")
        text_dim = theme.get("text_dim", "#94a3b8")

        # ✅ عنوان
        if custom_title:
            title = custom_title
        else:
            # Translation key: "empty_<key>_title"
            title = _tr(lang, f"empty_{key}_title")
            if title == f"empty_{key}_title" or not title:
                title = _tr(lang, f"{key}")

        # ✅ زیرنویس
        if custom_subtitle:
            subtitle = custom_subtitle
        else:
            subtitle = _tr(lang, f"empty_{key}_subtitle")
            if subtitle == f"empty_{key}_subtitle":
                subtitle = ""

        # ✅ hint
        if custom_hint:
            hint = custom_hint
        else:
            hint = _tr(lang, f"empty_{key}_hint")
            if hint == f"empty_{key}_hint":
                hint = ""

        # ✅ ساخت HTML
        parts = []

        # 1) Emoji
        if show_emoji and style.emoji:
            parts.append(
                f'<div style="font-size: {style.emoji_size}px; '
                f'color: {accent}; line-height: 1.2;">'
                f'{style.emoji}</div>'
            )

        # 2) Title
        if title:
            parts.append(
                f'<div style="font-size: {style.title_size}px; '
                f'font-weight: bold; color: {accent}; '
                f'margin-top: 8px;">{title}</div>'
            )

        # 3) Subtitle
        if subtitle:
            parts.append(
                f'<div style="font-size: {style.subtitle_size}px; '
                f'color: {text_dim}; margin-top: 4px;">{subtitle}</div>'
            )

        # 4) Hint (italic, smaller)
        if hint:
            parts.append(
                f'<div style="font-size: {style.hint_size}px; '
                f'color: {text_dim}; margin-top: 6px; '
                f'font-style: italic;">💡 {hint}</div>'
            )

        return "".join(parts) if parts else ""
    except Exception as e:
        print(f"⚠️ make_empty_state_html failed: {e}")
        return ""


# ============================================================
# ✅ EmptyStateWidget (full widget with animation + optional CTA)
# ============================================================

class EmptyStateWidget(QWidget):
    """
    A complete empty state widget with:
    - Animated emoji (pulse)
    - Title
    - Subtitle
    - Hint
    - Optional CTA button

    Signals:
        cta_clicked()
    """

    cta_clicked = pyqtSignal()

    def __init__(
            self,
            key: str,
            theme: Dict,
            lang: str = "English",
            cta_text: str = "",
            parent=None,
    ):
        super().__init__(parent)
        self._key = key
        self._theme = theme
        self._lang = lang
        self._cta_text = cta_text
        self._style = EMPTY_STATE_STYLES.get(key, EmptyStateStyle(emoji="✨"))

        self._build_ui()
        self._apply_theme()
        self._start_animation()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignCenter)

        # ✅ Emoji label (big, animated)
        self._emoji_label = QLabel(self._style.emoji)
        self._emoji_label.setAlignment(Qt.AlignCenter)
        self._emoji_label.setObjectName("empty_emoji")
        self._emoji_label.setStyleSheet(
            f"font-size: {self._style.emoji_size}px;"
            "background: transparent;"
        )
        layout.addWidget(self._emoji_label, 0, Qt.AlignCenter)

        # ✅ Title
        self._title_label = QLabel()
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setObjectName("empty_title")
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        # ✅ Subtitle
        self._subtitle_label = QLabel()
        self._subtitle_label.setAlignment(Qt.AlignCenter)
        self._subtitle_label.setObjectName("empty_subtitle")
        self._subtitle_label.setWordWrap(True)
        layout.addWidget(self._subtitle_label)

        # ✅ Hint
        self._hint_label = QLabel()
        self._hint_label.setAlignment(Qt.AlignCenter)
        self._hint_label.setObjectName("empty_hint")
        self._hint_label.setWordWrap(True)
        layout.addWidget(self._hint_label)

        # ✅ CTA button (اختیاری)
        self._cta_btn = QPushButton(self._cta_text or "")
        self._cta_btn.setObjectName("empty_cta")
        self._cta_btn.setCursor(Qt.PointingHandCursor)
        self._cta_btn.setMinimumHeight(38)
        self._cta_btn.clicked.connect(self._on_cta_clicked)
        self._cta_btn.setVisible(bool(self._cta_text))
        layout.addWidget(self._cta_btn, 0, Qt.AlignCenter)

        layout.addStretch()

        # ✅ متن‌ها رو ست کن
        self._update_texts()

    def _update_texts(self):
        """ترجمه‌ها رو ست کن."""
        try:
            title = _tr(self._lang, f"empty_{self._key}_title")
            if title == f"empty_{self._key}_title" or not title:
                title = _tr(self._lang, self._key)

            subtitle = _tr(self._lang, f"empty_{self._key}_subtitle")
            if subtitle == f"empty_{self._key}_subtitle":
                subtitle = ""

            hint = _tr(self._lang, f"empty_{self._key}_hint")
            if hint == f"empty_{self._key}_hint":
                hint = ""

            self._title_label.setText(title)
            self._subtitle_label.setText(subtitle)
            self._subtitle_label.setVisible(bool(subtitle))
            self._hint_label.setText(f"💡 {hint}" if hint else "")
            self._hint_label.setVisible(bool(hint))
        except Exception as e:
            print(f"⚠️ EmptyStateWidget._update_texts failed: {e}")

    def _apply_theme(self):
        t = self._theme
        accent = t.get(self._style.accent_key, t.get("primary", "#00e5ff"))
        text_dim = t.get("text_dim", "#94a3b8")

        self._title_label.setStyleSheet(
            f"color: {accent};"
            f"font-size: {self._style.title_size}px;"
            "font-weight: bold;"
            "background: transparent;"
        )
        self._subtitle_label.setStyleSheet(
            f"color: {text_dim};"
            f"font-size: {self._style.subtitle_size}px;"
            "background: transparent;"
        )
        self._hint_label.setStyleSheet(
            f"color: {text_dim};"
            f"font-size: {self._style.hint_size}px;"
            "font-style: italic;"
            "background: transparent;"
        )
        # CTA button
        self._cta_btn.setStyleSheet(f"""
            QPushButton#empty_cta {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t.get('primary', '#00e5ff')},
                    stop:0.5 {t.get('accent', '#f472b6')},
                    stop:1 {t.get('secondary', '#c084fc')}
                );
                color: {t.get('button_text', '#0a0a14')};
                border: none;
                border-radius: 10px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#empty_cta:hover {{
                background: {t.get('button_hover', '#67e8f9')};
            }}
        """)

    def _start_animation(self):
        """پالس ملایم روی emoji."""
        if not self._style.pulse:
            return
        try:
            self._emoji_effect = QGraphicsOpacityEffect(self._emoji_label)
            self._emoji_effect.setOpacity(1.0)
            self._emoji_label.setGraphicsEffect(self._emoji_effect)

            self._pulse_anim = QPropertyAnimation(
                self._emoji_effect, b"opacity", self
            )
            self._pulse_anim.setDuration(1800)
            self._pulse_anim.setStartValue(1.0)
            self._pulse_anim.setEndValue(0.45)
            self._pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
            self._pulse_anim.setLoopCount(-1)
            self._pulse_anim.start()
        except Exception as e:
            print(f"⚠️ EmptyStateWidget._start_animation failed: {e}")

    def stop_animation(self):
        try:
            if hasattr(self, "_pulse_anim") and self._pulse_anim is not None:
                self._pulse_anim.stop()
        except Exception:
            pass

    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_theme()

    def set_lang(self, lang: str):
        self._lang = lang
        self._update_texts()

    def _on_cta_clicked(self):
        self.cta_clicked.emit()
