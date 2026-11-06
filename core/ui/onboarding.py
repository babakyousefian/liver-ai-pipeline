# core/ui/onboarding.py
# -*- coding: utf-8 -*-
"""
Onboarding Tour — interactive step-by-step guide.

Two tour types:
- GUEST: 2 steps (Login + Signup) — for non-logged-in users
- FULL: ~15 steps covering every major feature — after first login

Storage:
    onboarding.json next to users_db.json
    {
        "users": {
            "babak": {
                "tour_completed": true,
                "completed_at": "2026-11-05T...",
                "version": "1.0"
            }
        }
    }
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable

from PyQt5.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGraphicsOpacityEffect, QApplication,
)
from PyQt5.QtCore import (
    Qt, QTimer, QRect, QRectF, QPoint, QPropertyAnimation, QEasingCurve,
    pyqtSignal, QObject, QEvent,
)
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QPainterPath, QFont, QRegion,
    QPolygonF, QRadialGradient,
)


# ============================================================
# ✅ Constants
# ============================================================

ONBOARDING_FILENAME = "onboarding.json"
TOUR_VERSION = "1.0"


# ============================================================
# ✅ Project root helper
# ============================================================

def _project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))


# ============================================================
# ✅ Storage
# ============================================================

class OnboardingStorage:
    """
    مدیریت فایل onboarding.json.

    Structure:
    {
        "users": {
            "username": {
                "tour_completed": true,
                "completed_at": "ISO",
                "version": "1.0",
                "skipped": false
            }
        }
    }
    """

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(_project_root(), ONBOARDING_FILENAME)
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
            except Exception as e:
                print(f"⚠️ OnboardingStorage load failed: {e}")
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
            print(f"⚠️ OnboardingStorage save failed: {e}")

    def has_completed(self, username: str) -> bool:
        """آیا این کاربر Tour رو دیده؟"""
        if not username:
            return False
        u = self._data.get("users", {}).get(username, {})
        return bool(u.get("tour_completed", False)) and \
               u.get("version") == TOUR_VERSION

    def mark_completed(self, username: str, skipped: bool = False):
        """ثبت اینکه کاربر Tour رو دید (یا skip کرد)."""
        if not username:
            return
        self._data.setdefault("users", {})[username] = {
            "tour_completed": True,
            "completed_at": datetime.now().isoformat(),
            "version": TOUR_VERSION,
            "skipped": bool(skipped),
        }
        self._save()

    def reset(self, username: str):
        """پاک کردن وضعیت Tour (برای دیدن مجدد)."""
        if not username:
            return
        users = self._data.get("users", {})
        if username in users:
            del users[username]
            self._save()


# ============================================================
# ✅ Step definition
# ============================================================

@dataclass
class OnboardingStep:
    """
    یه مرحله‌ی Tour.

    - widget_attr: نام attribute توی MainWindow (مثلاً "login_btn")
    - title_key / subtitle_key: کلیدهای ترجمه
    - position: "top" | "bottom" | "left" | "right" | "auto"
    - action: اقدام اختیاری قبل از نمایش (مثلاً باز کردن یه منو)
    """
    widget_attr: str
    title_key: str
    subtitle_key: str
    position: str = "auto"
    action: Optional[str] = None   # "open_account_menu", "open_dashboard", ...


# ✅ Tour برای مهمان (۲ مرحله)
TOUR_STEPS_GUEST: List[OnboardingStep] = [
    OnboardingStep(
        widget_attr="login_btn",
        title_key="tour_guest_login_title",
        subtitle_key="tour_guest_login_subtitle",
        position="bottom",
    ),
    OnboardingStep(
        widget_attr="signup_btn",
        title_key="tour_guest_signup_title",
        subtitle_key="tour_guest_signup_subtitle",
        position="bottom",
    ),
]


# ✅ Tour کامل برای کاربر لاگین‌شده
TOUR_STEPS_FULL: List[OnboardingStep] = [
    # 1) Insert Image
    OnboardingStep(
        widget_attr="insert_btn",
        title_key="tour_insert_title",
        subtitle_key="tour_insert_subtitle",
        position="top",
    ),
    # 2) Process
    OnboardingStep(
        widget_attr="process_btn",
        title_key="tour_process_title",
        subtitle_key="tour_process_subtitle",
        position="top",
    ),
    # 3) Export PDF
    OnboardingStep(
        widget_attr="export_pdf_btn",
        title_key="tour_export_title",
        subtitle_key="tour_export_subtitle",
        position="top",
    ),
    # 4) Grad-CAM
    OnboardingStep(
        widget_attr="gradcam_btn",
        title_key="tour_gradcam_title",
        subtitle_key="tour_gradcam_subtitle",
        position="top",
    ),
    # 5) Dashboard toggle
    OnboardingStep(
        widget_attr="dashboard_btn",
        title_key="tour_dashboard_title",
        subtitle_key="tour_dashboard_subtitle",
        position="bottom",
    ),
    # 6) History toggle
    OnboardingStep(
        widget_attr="history_btn",
        title_key="tour_history_title",
        subtitle_key="tour_history_subtitle",
        position="bottom",
    ),
    # 7) Visit Webpage
    OnboardingStep(
        widget_attr="visit_btn",
        title_key="tour_visit_title",
        subtitle_key="tour_visit_subtitle",
        position="top",
    ),
    # 8) Mute
    OnboardingStep(
        widget_attr="mute_btn",
        title_key="tour_mute_title",
        subtitle_key="tour_mute_subtitle",
        position="top",
    ),
    # 9) Zoom controls
    OnboardingStep(
        widget_attr="image_view",
        title_key="tour_zoom_title",
        subtitle_key="tour_zoom_subtitle",
        position="bottom",
    ),
    # 10) Account Menu
    OnboardingStep(
        widget_attr="account_btn",
        title_key="tour_account_title",
        subtitle_key="tour_account_subtitle",
        position="bottom",
    ),
]


# ============================================================
# ✅ Spotlight overlay
# ============================================================

class OnboardingOverlay(QWidget):
    """
    Backdrop تاریک + سوراخ روی widget هدف.

    - کل parent رو می‌پوشونه
    - یه مستطیل (دور widget هدف) رو روشن نگه می‌داره
    - کلیک روی backdrop → Tour رو نمی‌بنده (کاربر باید از دکمه‌ها استفاده کنه)
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_StyledBackground, False)

        self._target_rect = QRect()
        self._padding = 6
        self._corner_radius = 12
        self._overlay_alpha = 170

        self.setVisible(False)

    def set_target_rect(self, rect: QRect, padding: int = 6, corner_radius: int = 12):
        self._target_rect = QRect(rect).adjusted(-padding, -padding, padding, padding)
        self._padding = padding
        self._corner_radius = corner_radius
        self.update()

    def clear_target(self):
        self._target_rect = QRect()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)

            w = self.width()
            h = self.height()
            if w <= 0 or h <= 0:
                return

            # ✅ کل صفحه تاریک
            full_path = QPainterPath()
            full_path.addRect(QRectF(0, 0, w, h))

            # ✅ اگه widget هدف داریم، سوراخش کن
            if self._target_rect.isValid() and not self._target_rect.isEmpty():
                # ✅ QRectF از QRect بساز
                tr = self._target_rect
                target_rectf = QRectF(
                    float(tr.x()),
                    float(tr.y()),
                    float(tr.width()),
                    float(tr.height()),
                )

                hole = QPainterPath()
                hole.addRoundedRect(
                    target_rectf,
                    float(self._corner_radius),
                    float(self._corner_radius),
                )

                cut = full_path.subtracted(hole)
                painter.fillPath(cut, QColor(0, 0, 0, self._overlay_alpha))

                # ✅ حلقه‌ی دور سوراخ (رنگ primary)
                pen = QPen(QColor(255, 255, 255, 220), 2)
                pen.setStyle(Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(
                    target_rectf,
                    float(self._corner_radius),
                    float(self._corner_radius),
                )
            else:
                painter.fillPath(full_path, QColor(0, 0, 0, self._overlay_alpha))
        except Exception as e:
            print(f"⚠️ OnboardingOverlay.paintEvent failed: {e}")
        finally:
            # ✅ همیشه end() صدا زده بشه
            try:
                painter.end()
            except Exception:
                pass


# ============================================================
# ✅ Tooltip Card (راهنما)
# ============================================================

class OnboardingCard(QFrame):
    """
    کارت راهنما که کنار widget هدف ظاهر می‌شه.
    """

    next_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()

    def __init__(self, theme: Dict, lang: str = "English", parent=None):
        super().__init__(parent)
        self._theme = theme
        self._lang = lang
        self.setObjectName("onboarding_card")
        self.setFixedWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # ✅ Title
        self._title = QLabel()
        self._title.setObjectName("ob_title")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        # ✅ Subtitle
        self._subtitle = QLabel()
        self._subtitle.setObjectName("ob_subtitle")
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._subtitle)

        layout.addSpacing(6)

        # ✅ Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._skip_btn = QPushButton()
        self._skip_btn.setObjectName("ob_skip")
        self._skip_btn.setCursor(Qt.PointingHandCursor)
        self._skip_btn.clicked.connect(self.skip_clicked.emit)

        self._prev_btn = QPushButton()
        self._prev_btn.setObjectName("ob_prev")
        self._prev_btn.setCursor(Qt.PointingHandCursor)
        self._prev_btn.clicked.connect(self.prev_clicked.emit)

        self._next_btn = QPushButton()
        self._next_btn.setObjectName("ob_next")
        self._next_btn.setCursor(Qt.PointingHandCursor)
        self._next_btn.clicked.connect(self.next_clicked.emit)

        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._prev_btn)
        btn_row.addWidget(self._next_btn)
        layout.addLayout(btn_row)

        self._apply_theme()

    def _apply_theme(self):
        t = self._theme
        self.setStyleSheet(f"""
            QFrame#onboarding_card {{
                background: {t.get('input_bg', 'rgba(30, 41, 59, 0.95)')};
                border: 2px solid {t.get('primary', '#00e5ff')};
                border-radius: 14px;
            }}
            QLabel#ob_title {{
                font-size: 16px;
                font-weight: bold;
                color: {t.get('primary', '#00e5ff')};
                background: transparent;
            }}
            QLabel#ob_subtitle {{
                font-size: 12px;
                color: {t.get('text', '#f1f5f9')};
                background: transparent;
                line-height: 1.4;
            }}
            QPushButton#ob_skip {{
                background: transparent;
                color: {t.get('text_dim', '#94a3b8')};
                border: none;
                font-size: 11px;
                text-decoration: underline;
            }}
            QPushButton#ob_skip:hover {{
                color: {t.get('danger', '#f87171')};
            }}
            QPushButton#ob_prev {{
                background: {t.get('input_bg', 'rgba(30, 41, 59, 0.85)')};
                color: {t.get('text', '#f1f5f9')};
                border: 1px solid {t.get('glass_border', '#00e5ff')};
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#ob_prev:hover {{
                background: {t.get('primary', '#00e5ff')};
                color: {t.get('button_text', '#0a0a14')};
            }}
            QPushButton#ob_next {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t.get('primary', '#00e5ff')},
                    stop:1 {t.get('secondary', '#c084fc')}
                );
                color: {t.get('button_text', '#0a0a14')};
                border: none;
                border-radius: 8px;
                padding: 6px 18px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#ob_next:hover {{
                background: {t.get('button_hover', '#67e8f9')};
            }}
        """)

    def set_content(
        self,
        title: str,
        subtitle: str,
        counter: str,
        is_first: bool,
        is_last: bool,
        next_text: str = "",
        prev_text: str = "",
        skip_text: str = "",
        finish_text: str = "",
    ):
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        self._skip_btn.setText(skip_text or "Skip")
        self._prev_btn.setText(prev_text or "←")
        self._prev_btn.setVisible(not is_first)
        if is_last:
            self._next_btn.setText(finish_text or "Finish ✓")
        else:
            self._next_btn.setText(f"{next_text or 'Next'} ({counter})")


# ============================================================
# ✅ Onboarding Tour (main class)
# ============================================================

class OnboardingTour(QObject):
    """
    Tour کامل.

    Usage:
        tour = OnboardingTour(main_window, steps, theme, lang)
        tour.finished.connect(lambda skipped: ...)
        tour.start()
    """

    finished = pyqtSignal(bool)  # True if skipped

    def __init__(
        self,
        main_window: QWidget,
        steps: List[OnboardingStep],
        theme: Dict,
        lang: str = "English",
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent or main_window)
        self._mw = main_window
        self._steps = steps
        self._theme = theme
        self._lang = lang
        self._index = 0
        self._skipped = False

        # ✅ Overlay
        self._overlay = OnboardingOverlay(main_window)
        self._overlay.setGeometry(main_window.rect())

        # ✅ Card
        self._card = OnboardingCard(theme, lang, main_window)
        self._card.next_clicked.connect(self._on_next)
        self._card.prev_clicked.connect(self._on_prev)
        self._card.skip_clicked.connect(self._on_skip)

        # ✅ نصب event filter برای resize
        main_window.installEventFilter(self)

    # ============================================================
    # Public API
    # ============================================================
    def start(self):
        """شروع Tour از مرحله اول."""
        if not self._steps:
            self.finished.emit(False)
            return
        self._index = 0
        self._skipped = False
        QTimer.singleShot(300, self._show_step)

    # ============================================================
    # Internals
    # ============================================================
    def _tr(self, key: str) -> str:
        try:
            from GUI_update import tr
            return tr(self._lang, key)
        except Exception:
            return key

    def _find_widget(self, attr: str) -> Optional[QWidget]:
        try:
            w = getattr(self._mw, attr, None)
            if w is None:
                return None
            if not isinstance(w, QWidget):
                return None
            return w
        except Exception:
            return None

    def _show_step(self):
        """نمایش مرحله فعلی."""
        try:
            if self._index < 0:
                self._index = 0
            if self._index >= len(self._steps):
                self._finish(skipped=False)
                return

            step = self._steps[self._index]

            # ✅ اجرای action (اگه لازمه)
            if step.action:
                self._run_action(step.action)

            # ✅ پیدا کردن widget
            target = self._find_widget(step.widget_attr)
            if target is None:
                # ✅ اگه widget اصلاً وجود نداره، برو مرحله بعد
                print(f"⚠️ Tour: widget '{step.widget_attr}' not found, skipping")
                self._index += 1
                QTimer.singleShot(100, self._show_step)
                return

            # ✅ اگه widget پنهانه، موقتاً نشونش بده
            was_hidden = not target.isVisible()
            if was_hidden:
                try:
                    target.setVisible(True)
                    # ✅ ذخیره کن که بعداً دوباره مخفیش کنیم
                    if not hasattr(self, "_temporarily_shown"):
                        self._temporarily_shown = set()
                    self._temporarily_shown.add(step.widget_attr)
                    print(f"👁️ Tour: temporarily showing '{step.widget_attr}'")
                except Exception as e:
                    print(f"⚠️ Could not show widget '{step.widget_attr}': {e}")
                    self._index += 1
                    QTimer.singleShot(100, self._show_step)
                    return

            # ✅ اگه widget هنوز layout نشده، یه بار دیگه امتحان کن
            if target.width() <= 0 or target.height() <= 0:
                QTimer.singleShot(150, self._show_step)
                return

            # ✅ محاسبه‌ی موقعیت روی صفحه
            try:
                top_left_global = target.mapTo(self._mw, QPoint(0, 0))
                target_rect = QRect(top_left_global, target.size())
            except Exception:
                target_rect = target.geometry()

            # ✅ overlay
            self._overlay.setGeometry(self._mw.rect())
            self._overlay.set_target_rect(target_rect, padding=6, corner_radius=12)
            self._overlay.setVisible(True)
            self._overlay.raise_()

            # ✅ محتوای card
            title = self._tr(step.title_key)
            subtitle = self._tr(step.subtitle_key)
            counter = f"{self._index + 1} / {len(self._steps)}"
            is_first = (self._index == 0)
            is_last = (self._index == len(self._steps) - 1)

            next_text = self._tr("tour_btn_next") if self._tr("tour_btn_next") != "tour_btn_next" else "Next"
            prev_text = self._tr("tour_btn_prev") if self._tr("tour_btn_prev") != "tour_btn_prev" else "Back"
            skip_text = self._tr("tour_btn_skip") if self._tr("tour_btn_skip") != "tour_btn_skip" else "Skip"
            finish_text = self._tr("tour_btn_finish") if self._tr(
                "tour_btn_finish") != "tour_btn_finish" else "Finish ✓"

            self._card.set_content(
                title=title,
                subtitle=subtitle,
                counter=counter,
                is_first=is_first,
                is_last=is_last,
                next_text=next_text,
                prev_text=prev_text,
                skip_text=skip_text,
                finish_text=finish_text,
            )

            # ✅ موقعیت card
            self._position_card(target_rect, step.position)

            self._card.setVisible(True)
            self._card.raise_()
            self._card.adjustSize()

            # ✅ اطمینان از اینکه card بیرون صفحه نیفته
            self._clamp_card()
        except Exception as e:
            print(f"⚠️ OnboardingTour._show_step failed: {e}")
            import traceback
            traceback.print_exc()
            self._finish(skipped=True)

    def _position_card(self, target_rect: QRect, position: str = "auto"):
        """محاسبه‌ی موقعیت card کنار widget."""
        try:
            self._card.adjustSize()
            cw = self._card.width()
            ch = self._card.height()

            mw_w = self._mw.width()
            mw_h = self._mw.height()

            gap = 14

            if position == "auto":
                # اگه widget پایین صفحه‌ست، card بالا؛ وگرنه پایین
                if target_rect.center().y() > mw_h * 0.6:
                    position = "top"
                else:
                    position = "bottom"

            if position == "top":
                x = target_rect.center().x() - cw // 2
                y = target_rect.top() - ch - gap
            elif position == "bottom":
                x = target_rect.center().x() - cw // 2
                y = target_rect.bottom() + gap
            elif position == "left":
                x = target_rect.left() - cw - gap
                y = target_rect.center().y() - ch // 2
            elif position == "right":
                x = target_rect.right() + gap
                y = target_rect.center().y() - ch // 2
            else:
                x = target_rect.center().x() - cw // 2
                y = target_rect.bottom() + gap

            self._card.move(int(x), int(y))
            self._card._preferred_position = position
            self._card._target_rect = target_rect
        except Exception as e:
            print(f"⚠️ _position_card failed: {e}")

    def _clamp_card(self):
        """اگه card بیرون صفحه افتاد، جابه‌جاش کن."""
        try:
            mw_rect = self._mw.rect()
            card_geo = self._card.geometry()
            preferred = getattr(self._card, "_preferred_position", "bottom")
            target_rect = getattr(self._card, "_target_rect", QRect())

            gap = 14
            cw = self._card.width()
            ch = self._card.height()

            x = card_geo.x()
            y = card_geo.y()

            # اگه از راست زد بیرون
            if x + cw > mw_rect.right() - 8:
                x = mw_rect.right() - cw - 8
            # اگه از چپ زد بیرون
            if x < mw_rect.left() + 8:
                x = mw_rect.left() + 8

            # اگه از پایین زد بیرون
            if y + ch > mw_rect.bottom() - 8:
                # اگه می‌شه بالا بذار
                if target_rect.isValid():
                    new_y = target_rect.top() - ch - gap
                    if new_y >= mw_rect.top() + 8:
                        y = new_y
                    else:
                        y = mw_rect.bottom() - ch - 8
                else:
                    y = mw_rect.bottom() - ch - 8

            # اگه از بالا زد بیرون
            if y < mw_rect.top() + 8:
                if target_rect.isValid():
                    new_y = target_rect.bottom() + gap
                    if new_y + ch <= mw_rect.bottom() - 8:
                        y = new_y
                    else:
                        y = mw_rect.top() + 8
                else:
                    y = mw_rect.top() + 8

            self._card.move(int(x), int(y))
        except Exception as e:
            print(f"⚠️ _clamp_card failed: {e}")

    def _run_action(self, action: str):
        """اجرای action قبل از نمایش مرحله."""
        try:
            if action == "open_account_menu":
                # ✅ باز کردن منوی account
                btn = getattr(self._mw, "account_btn", None)
                if btn is not None:
                    # شبیه‌سازی کلیک
                    try:
                        btn.click()
                    except Exception:
                        pass
        except Exception as e:
            print(f"⚠️ _run_action failed: {e}")

    def _on_next(self):
        """Next."""
        self._index += 1
        if self._index >= len(self._steps):
            self._finish(skipped=False)
        else:
            self._show_step()

    def _on_prev(self):
        """Previous."""
        self._index = max(0, self._index - 1)
        self._show_step()

    def _on_skip(self):
        """Skip."""
        self._finish(skipped=True)

    def _finish(self, skipped: bool):
        """پایان Tour."""
        try:
            # ✅ مخفی کردن widgetهایی که موقتاً نشون دادیم
            if hasattr(self, "_temporarily_shown"):
                for attr in self._temporarily_shown:
                    try:
                        w = getattr(self._mw, attr, None)
                        if w is not None and isinstance(w, QWidget):
                            w.setVisible(False)
                            print(f"👁️ Tour: hiding '{attr}' again")
                    except Exception:
                        pass
                self._temporarily_shown.clear()

            self._overlay.setVisible(False)
            self._card.setVisible(False)
            try:
                self._mw.removeEventFilter(self)
            except Exception:
                pass
            self.finished.emit(skipped)
        except Exception as e:
            print(f"⚠️ OnboardingTour._finish failed: {e}")

    # ============================================================
    # Event filter (resize)
    # ============================================================
    def eventFilter(self, obj, event):
        try:
            if obj is self._mw and event.type() == QEvent.Resize:
                self._overlay.setGeometry(self._mw.rect())
                if self._card.isVisible():
                    self._show_step()
        except Exception:
            pass
        return False
