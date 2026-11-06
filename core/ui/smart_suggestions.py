# core/ui/smart_suggestions.py
# -*- coding: utf-8 -*-
"""
Smart Suggestions — rule-based recommendations that guide the user.

Each suggestion appears as a slide-up bar at the bottom of the main window
with a clear action button. Suggestions are shown ONCE per user.

Rules (8 total):
    1. image_ready      — image loaded, not processed (5s)
    2. pdf_ready        — result ready, no PDF (5s)
    3. gradcam_hint     — result ready, no Grad-CAM (10s)
    4. repeated_grid    — same grid 2x in a row
    5. history_hint     — never opened History (after 2 results)
    6. settings_hint    — never opened Settings (after 2 min)
    7. dashboard_hint   — never opened Dashboard (after 2 results)
    8. idle_help        — idle for 20s

Storage:
    smart_suggestions.json next to users_db.json
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Callable

from PyQt5.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGraphicsOpacityEffect,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve,
    pyqtSignal, QObject,
)
from PyQt5.QtGui import QColor


# ============================================================
# ✅ Constants
# ============================================================

SUGGESTIONS_FILENAME = "smart_suggestions.json"
AUTO_HIDE_MS = 15000          # خودکار بعد از ۱۵ ثانیه مخفی
ANIMATION_DURATION = 320      # انیمیشن slide
CHECK_INTERVAL_MS = 2000      # هر ۲ ثانیه چک کن

# ✅ آستانه‌های زمانی (طبق جدول جدید)
IMAGE_READY_DELAY_SEC    = 5.0     # ۵ ثانیه بعد از لود تصویر
PDF_READY_DELAY_SEC      = 5.0     # ۵ ثانیه بعد از نتیجه
GRADCAM_HINT_DELAY_SEC   = 10.0    # ۱۰ ثانیه بعد از نتیجه
SETTINGS_HINT_DELAY_SEC  = 120.0   # ۲ دقیقه از ورود کاربر
IDLE_HELP_DELAY_SEC      = 20.0    # ۲۰ ثانیه بی‌تحرکی

# ✅ آستانه‌های شمارشی
REPEATED_GRID_THRESHOLD  = 2       # ۲ بار پشت‌سرهم
HISTORY_MIN_RESULTS      = 2       # حداقل ۲ نتیجه
DASHBOARD_MIN_RESULTS    = 2       # حداقل ۲ نتیجه


# ============================================================
# ✅ Project root
# ============================================================

def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))


# ============================================================
# ✅ Storage (per-user)
# ============================================================

class SuggestionStorage:
    """مدیریت فایل smart_suggestions.json."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(_project_root(), SUGGESTIONS_FILENAME)
        self.path = path
        self._data = self._load()

    def _load(self) -> Dict:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return {"users": {}}
                data.setdefault("users", {})
                return data
            except Exception:
                return {"users": {}}
        return {"users": {}}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"⚠️ SuggestionStorage save failed: {e}")

    def _get_user(self, username: str) -> Dict:
        users = self._data.setdefault("users", {})
        if username not in users:
            users[username] = {
                "shown": [],
                "ignored": [],
                "last_seen": "",
            }
        u = users[username]
        u.setdefault("shown", [])
        u.setdefault("ignored", [])
        u.setdefault("last_seen", "")
        return u

    def is_shown(self, username: str, rule_id: str) -> bool:
        if not username:
            return True
        u = self._get_user(username)
        return rule_id in u.get("shown", [])

    def mark_shown(self, username: str, rule_id: str):
        if not username:
            return
        u = self._get_user(username)
        if rule_id not in u["shown"]:
            u["shown"].append(rule_id)
        u["last_seen"] = datetime.now().isoformat()
        self._save()

    def is_ignored(self, username: str, rule_id: str) -> bool:
        if not username:
            return True
        u = self._get_user(username)
        return rule_id in u.get("ignored", [])

    def mark_ignored(self, username: str, rule_id: str):
        if not username:
            return
        u = self._get_user(username)
        if rule_id not in u["ignored"]:
            u["ignored"].append(rule_id)
        self._save()

    def clear_user(self, username: str):
        if not username:
            return
        users = self._data.get("users", {})
        if username in users:
            del users[username]
            self._save()


# ============================================================
# ✅ Suggestion data
# ============================================================

@dataclass
class Suggestion:
    """یک پیشنهاد آماده‌ی نمایش."""
    rule_id: str
    title: str
    subtitle: str
    action_label: str
    action: Optional[Callable] = None
    icon: str = "💡"


@dataclass
class SuggestionContext:
    """وضعیت فعلی برنامه که موتور ازش استفاده می‌کنه."""
    username: str = ""
    has_image: bool = False
    image_loaded_at: Optional[float] = None       # timestamp
    has_result: bool = False
    result_at: Optional[float] = None             # timestamp
    has_pdf: bool = False
    has_gradcam: bool = False
    last_grid: Optional[int] = None
    consecutive_same_grid: int = 0
    results_count: int = 0
    history_opened: bool = False
    dashboard_opened: bool = False
    settings_opened: bool = False
    session_start_at: float = 0.0                 # ✅ زمان شروع session (برای rule 6)
    last_activity_at: float = 0.0
    current_time: float = 0.0


# ============================================================
# ✅ Suggestion Engine
# ============================================================

class SuggestionEngine(QObject):
    """
    موتور Smart Suggestions.

    - با QTimer هر CHECK_INTERVAL_MS چک می‌کنه
    - اگه یه rule match شد و قبلاً نشون داده نشده، emit می‌کنه
    - هر پیشنهاد فقط یک بار per-user نشون داده می‌شه
    """

    suggestion_ready = pyqtSignal(object)  # Suggestion

    def __init__(self, storage: SuggestionStorage, lang: str = "English", parent=None):
        super().__init__(parent)
        self._storage = storage
        self._lang = lang
        self._context = SuggestionContext()
        self._current_suggestion: Optional[Suggestion] = None
        self._enabled = True

        # ✅ تایمر چک
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._check)
        self._timer.setInterval(CHECK_INTERVAL_MS)

        # ✅ cooldown بین پیشنهادها (۱۵ ثانیه)
        self._last_emit_at = 0.0
        self._cooldown_sec = 15.0

    # ============================================================
    # Public API
    # ============================================================
    def start(self):
        if self._enabled:
            self._timer.start()

    def stop(self):
        self._timer.stop()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.start()
        else:
            self.stop()

    def set_lang(self, lang: str):
        """تغییر زبان — متن‌ها با tr ترجمه می‌شن."""
        self._lang = lang

    def update_context(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self._context, k):
                setattr(self._context, k, v)

    def set_username(self, username: str):
        self._context.username = username
        self._last_emit_at = 0.0

    def dismiss_current(self):
        self._current_suggestion = None

    # ============================================================
    # Translation helper
    # ============================================================
    def _t(self, key: str, fallback: str = "") -> str:
        """ترجمه‌ی امن با tr از GUI_update."""
        try:
            from GUI_update import tr
            val = tr(self._lang, key)
            if val and val != key:
                return val
            return fallback or key
        except Exception:
            return fallback or key

    # ============================================================
    # Internals
    # ============================================================
    def _check(self):
        if not self._enabled:
            return
        try:
            if not self._context.username:
                return

            # اگه یه پیشنهاد فعلاً بازه، چک نکن
            if self._current_suggestion is not None:
                return

            # cooldown
            now = self._context.current_time or datetime.now().timestamp()
            if (now - self._last_emit_at) < self._cooldown_sec:
                return

            # ✅ ترتیب اولویت (طبق جدول)
            for rule_fn in (
                self._rule_image_ready,       # 1
                self._rule_pdf_ready,         # 2
                self._rule_gradcam_hint,      # 3
                self._rule_repeated_grid,     # 4
                self._rule_history_hint,      # 5
                self._rule_settings_hint,     # 6
                self._rule_dashboard_hint,    # 7
                self._rule_idle_help,         # 8
            ):
                suggestion = rule_fn()
                if suggestion is not None:
                    self._current_suggestion = suggestion
                    self._last_emit_at = now
                    self.suggestion_ready.emit(suggestion)
                    return
        except Exception as e:
            print(f"⚠️ SuggestionEngine._check failed: {e}")

    def _can_show(self, rule_id: str) -> bool:
        username = self._context.username
        if self._storage.is_shown(username, rule_id):
            return False
        if self._storage.is_ignored(username, rule_id):
            return False
        return True

    def _mark(self, rule_id: str):
        self._storage.mark_shown(self._context.username, rule_id)

    def _grid_short(self, grid: Optional[int]) -> str:
        return {0: "NL", 1: "G1", 2: "G2", 3: "G3"}.get(grid, "G?")

    # ============================================================
    # Rules
    # ============================================================

    # ---------- 1. image_ready ----------
    def _rule_image_ready(self) -> Optional[Suggestion]:
        """تصویر لود شده، پردازش نشده — ۵ ثانیه."""
        ctx = self._context
        if not ctx.has_image or ctx.has_result:
            return None
        if ctx.image_loaded_at is None:
            return None
        if (ctx.current_time - ctx.image_loaded_at) < IMAGE_READY_DELAY_SEC:
            return None
        if not self._can_show("image_ready"):
            return None
        self._mark("image_ready")
        return Suggestion(
            rule_id="image_ready",
            title=self._t("suggestion_image_ready", "Image loaded — ready to process"),
            subtitle=self._t("suggestion_image_ready_sub",
                             "Click Process to run the AI analysis."),
            action_label="⚡ " + self._t("process", "Process"),
            icon="💡",
        )

    # ---------- 2. pdf_ready ----------
    def _rule_pdf_ready(self) -> Optional[Suggestion]:
        """نتیجه گرفته، PDF نگرفته — ۵ ثانیه."""
        ctx = self._context
        if not ctx.has_result or ctx.has_pdf:
            return None
        if ctx.result_at is None:
            return None
        if (ctx.current_time - ctx.result_at) < PDF_READY_DELAY_SEC:
            return None
        if not self._can_show("pdf_ready"):
            return None
        self._mark("pdf_ready")
        return Suggestion(
            rule_id="pdf_ready",
            title=self._t("suggestion_pdf_ready", "Result ready — export as PDF?"),
            subtitle=self._t("suggestion_pdf_ready_sub",
                             "Save a professional medical report in one click."),
            action_label="📄 " + self._t("export_pdf", "PDF"),
            icon="📄",
        )

    # ---------- 3. gradcam_hint ----------
    def _rule_gradcam_hint(self) -> Optional[Suggestion]:
        """نتیجه گرفته، Grad-CAM نگرفته — ۱۰ ثانیه."""
        ctx = self._context
        if not ctx.has_result or ctx.has_gradcam:
            return None
        if ctx.result_at is None:
            return None
        if (ctx.current_time - ctx.result_at) < GRADCAM_HINT_DELAY_SEC:
            return None
        if not self._can_show("gradcam_hint"):
            return None
        self._mark("gradcam_hint")
        return Suggestion(
            rule_id="gradcam_hint",
            title=self._t("suggestion_gradcam_hint",
                          "Curious what the AI saw?"),
            subtitle=self._t("suggestion_gradcam_hint_sub",
                             "See the attention heatmap over the image."),
            action_label="🔥 " + self._t("gradcam_show", "Show Map"),
            icon="🔥",
        )

    # ---------- 4. repeated_grid ----------
    def _rule_repeated_grid(self) -> Optional[Suggestion]:
        """۲ بار پشت‌سرهم همون Grid."""
        ctx = self._context
        if ctx.consecutive_same_grid < REPEATED_GRID_THRESHOLD:
            return None
        if ctx.last_grid is None:
            return None
        if not self._can_show("repeated_grid"):
            return None
        self._mark("repeated_grid")
        grid_short = self._grid_short(ctx.last_grid)
        title_tpl = self._t(
            "suggestion_repeated_grid",
            "You keep getting {grid}",
        )
        # اگه کلید ترجمه نداره، پیش‌فرض
        if title_tpl == "suggestion_repeated_grid":
            title_tpl = "You keep getting {grid}"
        return Suggestion(
            rule_id="repeated_grid",
            title=title_tpl.format(grid=grid_short),
            subtitle=self._t("suggestion_repeated_grid_sub",
                             "Open the dashboard to see the distribution."),
            action_label="📊 " + self._t("dashboard_title", "Dashboard"),
            icon="📊",
        )

    # ---------- 5. history_hint ----------
    def _rule_history_hint(self) -> Optional[Suggestion]:
        """هیچ‌وقت History باز نکرده — بعد از ۲ نتیجه."""
        ctx = self._context
        if ctx.history_opened:
            return None
        if ctx.results_count < HISTORY_MIN_RESULTS:
            return None
        if not self._can_show("history_hint"):
            return None
        self._mark("history_hint")
        return Suggestion(
            rule_id="history_hint",
            title=self._t("suggestion_history_hint",
                          "Your history is building up"),
            subtitle=self._t("suggestion_history_hint_sub",
                             "Browse your last 20 analyzed images."),
            action_label="🕐 " + self._t("history_title", "History"),
            icon="🕐",
        )

    # ---------- 6. settings_hint ----------
    def _rule_settings_hint(self) -> Optional[Suggestion]:
        """هیچ‌وقت Settings باز نکرده — ۲ دقیقه از ورود."""
        ctx = self._context
        if ctx.settings_opened:
            return None
        if ctx.session_start_at <= 0:
            return None
        if (ctx.current_time - ctx.session_start_at) < SETTINGS_HINT_DELAY_SEC:
            return None
        if not self._can_show("settings_hint"):
            return None
        self._mark("settings_hint")
        return Suggestion(
            rule_id="settings_hint",
            title=self._t("suggestion_settings_hint",
                          "Personalize the app"),
            subtitle=self._t("suggestion_settings_hint_sub",
                             "Change language, theme, font, and enable 2FA."),
            action_label="⚙️ " + self._t("settings", "Settings"),
            icon="⚙️",
        )

    # ---------- 7. dashboard_hint ----------
    def _rule_dashboard_hint(self) -> Optional[Suggestion]:
        """هیچ‌وقت Dashboard باز نکرده — بعد از ۲ نتیجه."""
        ctx = self._context
        if ctx.dashboard_opened:
            return None
        if ctx.results_count < DASHBOARD_MIN_RESULTS:
            return None
        if not self._can_show("dashboard_hint"):
            return None
        self._mark("dashboard_hint")
        return Suggestion(
            rule_id="dashboard_hint",
            title=self._t("suggestion_dashboard_hint",
                          "Your dashboard has interesting stats"),
            subtitle=self._t("suggestion_dashboard_hint_sub",
                             "See totals, averages, and distribution at a glance."),
            action_label="📊 " + self._t("dashboard_title", "Dashboard"),
            icon="📊",
        )

    # ---------- 8. idle_help ----------
    def _rule_idle_help(self) -> Optional[Suggestion]:
        """۲۰ ثانیه هیچ کاری نکرده."""
        ctx = self._context
        if ctx.last_activity_at <= 0:
            return None
        if (ctx.current_time - ctx.last_activity_at) < IDLE_HELP_DELAY_SEC:
            return None
        if not self._can_show("idle_help"):
            return None
        self._mark("idle_help")
        return Suggestion(
            rule_id="idle_help",
            title=self._t("suggestion_idle_help", "Need a quick guide?"),
            subtitle=self._t("suggestion_idle_help_sub",
                             "Replay the onboarding tour anytime."),
            action_label="🎓 " + self._t("tour_show_again", "Show Tour"),
            icon="🎓",
        )


# ============================================================
# ✅ Suggestion Bar (UI) — بدون تغییر نسبت به نسخه‌ی قبل
# ============================================================

class SmartSuggestionBar(QFrame):
    """
    نوار پیشنهاد که از پایین slide می‌شه.

    Signals:
        action_clicked(rule_id)
        dismissed(rule_id)
    """

    action_clicked = pyqtSignal(str)
    dismissed = pyqtSignal(str)

    HEIGHT = 64

    def __init__(self, parent: QWidget, theme: Dict, lang: str = "English"):
        super().__init__(parent)
        self._theme = theme
        self._lang = lang
        self._current: Optional[Suggestion] = None
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide)

        self.setObjectName("smart_suggestion_bar")
        self.setFixedHeight(self.HEIGHT)
        self.setVisible(False)

        self._build_ui()
        self._apply_theme()

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._slide_anim = QPropertyAnimation(self, b"pos", self)
        self._slide_anim.setDuration(ANIMATION_DURATION)
        self._slide_anim.setEasingCurve(QEasingCurve.OutCubic)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 10, 14, 10)
        layout.setSpacing(12)

        self._icon_label = QLabel("💡")
        self._icon_label.setObjectName("ss_icon")
        self._icon_label.setFixedSize(32, 32)
        self._icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._icon_label)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(2)

        self._title_label = QLabel()
        self._title_label.setObjectName("ss_title")

        self._subtitle_label = QLabel()
        self._subtitle_label.setObjectName("ss_subtitle")

        text_box.addWidget(self._title_label)
        text_box.addWidget(self._subtitle_label)
        layout.addLayout(text_box, 1)

        self._action_btn = QPushButton()
        self._action_btn.setObjectName("ss_action")
        self._action_btn.setCursor(Qt.PointingHandCursor)
        self._action_btn.setMinimumHeight(34)
        self._action_btn.clicked.connect(self._on_action_clicked)
        layout.addWidget(self._action_btn)

        self._dismiss_btn = QPushButton("✕")
        self._dismiss_btn.setObjectName("ss_dismiss")
        self._dismiss_btn.setFixedSize(28, 28)
        self._dismiss_btn.setCursor(Qt.PointingHandCursor)
        self._dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        layout.addWidget(self._dismiss_btn)

    def _apply_theme(self):
        t = self._theme
        primary = t.get("primary", "#00e5ff")
        secondary = t.get("secondary", "#c084fc")
        accent = t.get("accent", "#f472b6")
        text = t.get("text", "#f1f5f9")
        text_dim = t.get("text_dim", "#94a3b8")
        input_bg = t.get("input_bg", "rgba(30, 41, 59, 0.95)")
        button_text = t.get("button_text", "#0a0a14")

        self.setStyleSheet(f"""
            QFrame#smart_suggestion_bar {{
                background: {input_bg};
                border: 2px solid {primary};
                border-radius: 14px;
            }}
            QLabel#ss_icon {{
                font-size: 22px;
                background: transparent;
            }}
            QLabel#ss_title {{
                font-size: 13px;
                font-weight: bold;
                color: {text};
                background: transparent;
            }}
            QLabel#ss_subtitle {{
                font-size: 11px;
                color: {text_dim};
                background: transparent;
            }}
            QPushButton#ss_action {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {primary}, stop:0.5 {accent}, stop:1 {secondary}
                );
                color: {button_text};
                border: none;
                border-radius: 10px;
                padding: 8px 18px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#ss_action:hover {{
                background: {t.get('button_hover', '#67e8f9')};
            }}
            QPushButton#ss_dismiss {{
                background: transparent;
                color: {text_dim};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#ss_dismiss:hover {{
                color: {t.get('danger', '#f87171')};
            }}
        """)

    def set_theme(self, theme: Dict):
        self._theme = theme
        self._apply_theme()

    def set_lang(self, lang: str):
        self._lang = lang

    def show_suggestion(self, suggestion: Suggestion):
        try:
            self._current = suggestion
            self._icon_label.setText(suggestion.icon or "💡")
            self._title_label.setText(suggestion.title)
            self._subtitle_label.setText(suggestion.subtitle)
            self._action_btn.setText(suggestion.action_label)

            parent_h = self.parent().height() if self.parent() else 800
            parent_w = self.parent().width() if self.parent() else 1200
            bar_w = min(720, parent_w - 40)
            bar_h = self.HEIGHT
            margin = 20
            target_x = (parent_w - bar_w) // 2
            target_y = parent_h - bar_h - margin

            start_y = parent_h + 10

            self.setFixedWidth(bar_w)
            self.setGeometry(target_x, start_y, bar_w, bar_h)
            self.setVisible(True)
            self.raise_()

            self._slide_anim.stop()
            self._slide_anim.setStartValue(QPoint(target_x, start_y))
            self._slide_anim.setEndValue(QPoint(target_x, target_y))
            self._slide_anim.start()

            fade_in = QPropertyAnimation(self._opacity_effect, b"opacity", self)
            fade_in.setDuration(ANIMATION_DURATION)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.start()
            self._fade_in = fade_in

            self._auto_hide_timer.start(AUTO_HIDE_MS)
        except Exception as e:
            print(f"⚠️ SmartSuggestionBar.show_suggestion failed: {e}")

    def hide_suggestion(self):
        try:
            if not self.isVisible():
                return
            self._auto_hide_timer.stop()

            parent_h = self.parent().height() if self.parent() else 800
            current_pos = self.pos()
            end_y = parent_h + 10

            self._slide_anim.stop()
            self._slide_anim.setStartValue(current_pos)
            self._slide_anim.setEndValue(QPoint(current_pos.x(), end_y))
            self._slide_anim.finished.connect(self._on_slide_out_finished)
            self._slide_anim.start()

            fade_out = QPropertyAnimation(self._opacity_effect, b"opacity", self)
            fade_out.setDuration(ANIMATION_DURATION)
            fade_out.setStartValue(self._opacity_effect.opacity())
            fade_out.setEndValue(0.0)
            fade_out.start()
            self._fade_out = fade_out
        except Exception as e:
            print(f"⚠️ SmartSuggestionBar.hide_suggestion failed: {e}")
            self.setVisible(False)

    def _on_slide_out_finished(self):
        try:
            self._slide_anim.finished.disconnect(self._on_slide_out_finished)
        except TypeError:
            pass
        self.setVisible(False)

    def reposition(self):
        try:
            if not self.isVisible():
                return
            parent_h = self.parent().height() if self.parent() else 800
            parent_w = self.parent().width() if self.parent() else 1200
            bar_w = min(720, parent_w - 40)
            bar_h = self.HEIGHT
            margin = 20
            target_x = (parent_w - bar_w) // 2
            target_y = parent_h - bar_h - margin
            self.setFixedWidth(bar_w)
            self.move(target_x, target_y)
        except Exception:
            pass

    def _on_action_clicked(self):
        try:
            if self._current:
                self.action_clicked.emit(self._current.rule_id)
                self.hide_suggestion()
        except Exception as e:
            print(f"⚠️ _on_action_clicked failed: {e}")

    def _on_dismiss_clicked(self):
        try:
            if self._current:
                self.dismissed.emit(self._current.rule_id)
                self.hide_suggestion()
        except Exception as e:
            print(f"⚠️ _on_dismiss_clicked failed: {e}")

    def _on_auto_hide(self):
        self.hide_suggestion()

