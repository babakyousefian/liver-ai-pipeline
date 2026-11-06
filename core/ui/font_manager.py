# core/ui/font_manager.py
# -*- coding: utf-8 -*-
"""
Font Manager — manage global application font (family + size + scale).

Three layers:
  1. font-family  → applied via QApplication.setStyleSheet("* { font-family: ... }")
  2. font-size    → applied via QApplication.setFont(QFont(family, size))
  3. scale        → applied via fontMetrics scaling on top of size

Persistence:
  - Stored in settings.json under "font" key
  - Loaded on app startup (before MainWindow)
  - Applied instantly via FontManager.apply()
"""

import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase


# ============================================================
# Constants
# ============================================================

DEFAULT_FAMILY = "Noto Sans"      # built-in if available
DEFAULT_SIZE = 10                  # base size
DEFAULT_SCALE = 1.0                # 1.0 = normal

MIN_SIZE = 7
MAX_SIZE = 20

MIN_SCALE = 0.85
MAX_SCALE = 1.5

# ✅ Friendly names for common fonts (for display)
FONT_DISPLAY_NAMES = {
    "Noto Sans": "Noto Sans (recommended — multilingual)",
    "Vazirmatn": "Vazirmatn (Persian/Arabic)",
    "Segoe UI": "Segoe UI (Windows default)",
    "Tahoma": "Tahoma (classic)",
    "Arial": "Arial (universal)",
    "Calibri": "Calibri (Office)",
    "Roboto": "Roboto (Android)",
    "Cairo": "Cairo (Arabic)",
    "Iran Sans": "Iran Sans",
    "B Nazanin": "B Nazanin",
    "Courier New": "Courier New (monospace)",
    "Consolas": "Consolas (monospace)",
}


# ============================================================
# Settings
# ============================================================

@dataclass
class FontSettings:
    """Font configuration persisted to settings.json."""
    family: str = DEFAULT_FAMILY
    size: int = DEFAULT_SIZE
    scale: float = DEFAULT_SCALE
    follow_system: bool = False     # if True, use system default font

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FontSettings":
        obj = cls()
        for k, v in (data or {}).items():
            if hasattr(obj, k) and v is not None:
                setattr(obj, k, v)
        # ✅ Validate ranges
        try:
            obj.size = max(MIN_SIZE, min(MAX_SIZE, int(obj.size)))
        except Exception:
            obj.size = DEFAULT_SIZE
        try:
            obj.scale = max(MIN_SCALE, min(MAX_SCALE, float(obj.scale)))
        except Exception:
            obj.scale = DEFAULT_SCALE
        return obj


def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))


def load_font_settings() -> FontSettings:
    """Load font settings from settings.json."""
    path = os.path.join(_project_root(), "settings.json")
    if not os.path.isfile(path):
        return FontSettings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FontSettings.from_dict(data.get("font", {}))
    except Exception:
        return FontSettings()


def save_font_settings(settings: FontSettings) -> bool:
    """Save font settings to settings.json (merge)."""
    path = os.path.join(_project_root(), "settings.json")
    try:
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["font"] = settings.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"⚠️ save_font_settings failed: {e}")
        return False


# ============================================================
# Font listing
# ============================================================

def list_available_fonts() -> List[str]:
    """
    Return list of font families that:
    - are installed on the system
    - support at least Latin OR Persian characters
    """
    families = QFontDatabase().families()
    # ✅ Sort: recommended fonts first, then alphabetical
    priority = [
        "Noto Sans", "Vazirmatn", "Segoe UI", "Tahoma", "Arial",
        "Calibri", "Roboto", "Cairo", "Iran Sans", "B Nazanin",
    ]
    result = []
    for name in priority:
        if name in families and name not in result:
            result.append(name)
    for name in sorted(families):
        if name not in result:
            result.append(name)
    return result


def display_name_for(family: str) -> str:
    """Return a friendly display name for a font family."""
    return FONT_DISPLAY_NAMES.get(family, family)


# ============================================================
# Manager
# ============================================================

class FontManager:
    """
    Singleton-style font manager.

    Usage:
        FontManager.get().apply(settings)
        FontManager.get().reset()
    """

    _instance = None

    def __init__(self):
        self._settings = FontSettings()
        self._applied = False

    @classmethod
    def get(cls) -> "FontManager":
        if cls._instance is None:
            cls._instance = FontManager()
        return cls._instance

    # ============================================================
    # Public API
    # ============================================================
    def settings(self) -> FontSettings:
        return self._settings

    def apply(self, settings: FontSettings) -> bool:
        """
        Apply font settings globally.

        This:
          1. Sets QApplication font (family + size)
          2. Sets a global QSS with font-family
          3. Sets a global QSS variable for scaling (not used directly,
             but individual dialogs may pick it up via fontMetrics)
        """
        try:
            app = QApplication.instance()
            if app is None:
                print("⚠️ FontManager.apply: QApplication not ready")
                return False

            self._settings = settings

            # ✅ Determine effective family
            family = settings.family or DEFAULT_FAMILY
            if settings.follow_system:
                family = app.font().family()

            # ✅ Determine effective size (with scale)
            base_size = settings.size or DEFAULT_SIZE
            scale = settings.scale or DEFAULT_SCALE
            effective_size = max(6, int(round(base_size * scale)))

            # ✅ Set QApplication font
            new_font = QFont(family, effective_size)
            new_font.setStyleStrategy(QFont.PreferAntialias)
            app.setFont(new_font)

            # ✅ Set global QSS — font-family on EVERY widget
            app.setStyleSheet(app.styleSheet() or "")
            global_qss = (
                f"* {{"
                f"  font-family: '{family}', 'Noto Sans', 'Segoe UI', sans-serif;"
                f"}}"
            )
            app.setStyleSheet(global_qss)

            self._applied = True
            print(f"🎨 Font applied: {family} {effective_size}pt (scale {scale})")
            return True
        except Exception as e:
            print(f"⚠️ FontManager.apply failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reset(self) -> bool:
        """Reset to default font."""
        return self.apply(FontSettings())

    def effective_size(self) -> int:
        """Effective size (base × scale) — useful for dialogs."""
        try:
            base = self._settings.size or DEFAULT_SIZE
            scale = self._settings.scale or DEFAULT_SCALE
            return max(6, int(round(base * scale)))
        except Exception:
            return DEFAULT_SIZE

    def effective_family(self) -> str:
        """Effective family — useful for dialogs."""
        try:
            return self._settings.family or DEFAULT_FAMILY
        except Exception:
            return DEFAULT_FAMILY
