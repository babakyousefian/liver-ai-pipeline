# gui/widgets/password_match_indicator.py
# -*- coding: utf-8 -*-
"""
PasswordMatchIndicator — shows whether repeat_password matches password.

Usage:
    indicator = PasswordMatchIndicator(theme, lang)
    indicator.set_passwords(pw1, pw2)
"""

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


def _get_helpers():
    from GUI_update import tr, THEMES
    return {"tr": tr, "THEMES": THEMES}


class PasswordMatchIndicator(QLabel):
    """Simple label showing match status."""

    def __init__(self, theme: dict = None, lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme or h["THEMES"].get("theme_dark_aqua", {})
        self._lang = lang
        self._pw1 = ""
        self._pw2 = ""

        self.setObjectName("pmi_label")
        self.setVisible(False)
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QLabel#pmi_label {{
                font-size: 10px;
                background: transparent;
                padding: 2px 4px;
            }}
        """)

    def set_theme(self, theme: dict):
        self._theme = theme
        self._apply_theme()
        self._refresh()

    def set_lang(self, lang: str):
        self._lang = lang
        self._refresh()

    def set_passwords(self, pw1: str, pw2: str):
        self._pw1 = pw1 or ""
        self._pw2 = pw2 or ""
        self._refresh()

    def _refresh(self):
        if not self._pw1 or not self._pw2:
            self.setVisible(False)
            return

        self.setVisible(True)
        t = self._theme

        if self._pw1 == self._pw2:
            color = t.get("success", "#34d399")
            self.setText("✓ " + self._tr(self._lang, "passwords_match"))
            self.setStyleSheet(f"""
                QLabel#pmi_label {{
                    color: {color};
                    font-size: 10px;
                    background: transparent;
                    padding: 2px 4px;
                }}
            """)
        else:
            color = t.get("danger", "#f87171")
            self.setText("✗ " + self._tr(self._lang, "passwords_dont_match"))
            self.setStyleSheet(f"""
                QLabel#pmi_label {{
                    color: {color};
                    font-size: 10px;
                    background: transparent;
                    padding: 2px 4px;
                }}
            """)
