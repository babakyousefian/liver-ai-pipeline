# gui/widgets/lock_screen.py
# -*- coding: utf-8 -*-
"""
Lock screen dialog shown after session timeout.

- Full opaque backdrop covering parent window
- Glassmorphism card centered
- Real-time countdown display
- Enter to unlock, Escape disabled
- Logout button (clears session, keeps account)
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QWidget, QApplication,
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QRect,
)
from PyQt5.QtGui import QColor, QPainter


def _get_helpers():
    from GUI_update import (
        tr, THEMES, attach_sound, SoundManager,
        animate_dialog_open, animate_dialog_close,
    )
    return {
        "tr": tr,
        "THEMES": THEMES,
        "attach_sound": attach_sound,
        "SoundManager": SoundManager,
        "animate_dialog_open": animate_dialog_open,
        "animate_dialog_close": animate_dialog_close,
    }


class LockScreen(QDialog):
    """
    Lock screen — blocks the entire app until password is entered.

    Signals:
        unlock_successful(): password correct
        logout_requested(): user chose to logout
    """

    unlock_successful = pyqtSignal()
    logout_requested = pyqtSignal()

    def __init__(self, username: str, db, theme: dict,
                 lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._username = username
        self._db = db
        self._attempts = []

        self.setWindowTitle(self._tr(lang, "session_locked"))
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        # Size to parent (or screen if no parent)
        if parent is not None:
            tl = parent.mapToGlobal(parent.rect().topLeft())
            self.setGeometry(tl.x(), tl.y(), parent.width(), parent.height())
        else:
            self.setGeometry(QApplication.primaryScreen().geometry())

        self._build_ui()
        self._apply_theme()

        try:
            h["SoundManager"].get().play("logoutSound")
        except Exception:
            pass

        # Focus password after show
        QTimer.singleShot(300, self.password_input.setFocus)

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        tr = self._tr
        lang = self._lang

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setAlignment(Qt.AlignCenter)

        self.backdrop = _OpaqueBackdrop(self._theme, parent=self)
        self.backdrop.setGeometry(self.rect())
        self.backdrop.lower()

        self.card = QFrame()
        self.card.setObjectName("lock_card")
        self.card.setFixedSize(440, 560)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(36, 32, 36, 32)
        card_layout.setSpacing(14)

        self.icon_label = QLabel("🔒")
        self.icon_label.setObjectName("lock_icon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedHeight(80)
        card_layout.addWidget(self.icon_label)

        title = QLabel(tr(lang, "session_locked"))
        title.setObjectName("lock_title")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        user_label = QLabel(
            tr(lang, "locked_for_user").format(user=self._username)
        )
        user_label.setObjectName("lock_username")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setWordWrap(True)
        card_layout.addWidget(user_label)

        card_layout.addSpacing(6)

        pass_label = QLabel(tr(lang, "password"))
        pass_label.setObjectName("lock_label")
        card_layout.addWidget(pass_label)

        self.password_input = QLineEdit()
        self.password_input.setObjectName("lock_input")
        self.password_input.setPlaceholderText(tr(lang, "password"))
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(46)
        self.password_input.returnPressed.connect(self._on_unlock)
        card_layout.addWidget(self.password_input)

        self.error_label = QLabel("")
        self.error_label.setObjectName("lock_error")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setVisible(False)
        card_layout.addWidget(self.error_label)

        card_layout.addStretch()

        self.unlock_btn = QPushButton("🔓  " + tr(lang, "unlock"))
        self.unlock_btn.setObjectName("lock_primary_btn")
        self.unlock_btn.setMinimumHeight(46)
        self.unlock_btn.setCursor(Qt.PointingHandCursor)
        self.unlock_btn.clicked.connect(self._on_unlock)
        card_layout.addWidget(self.unlock_btn)

        self.logout_btn = QPushButton("🚪  " + tr(lang, "logout"))
        self.logout_btn.setObjectName("lock_secondary_btn")
        self.logout_btn.setMinimumHeight(42)
        self.logout_btn.setCursor(Qt.PointingHandCursor)
        self.logout_btn.clicked.connect(self._on_logout)
        card_layout.addWidget(self.logout_btn)

        main.addWidget(self.card, 0, Qt.AlignCenter)

    def _apply_theme(self):
        t = self._theme
        self.card.setStyleSheet(f"""
            QFrame#lock_card {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 24px;
            }}
            QLabel#lock_icon {{
                font-size: 56px;
                background: transparent;
            }}
            QLabel#lock_title {{
                font-size: 22px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#lock_username {{
                font-size: 12px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QLabel#lock_label {{
                font-size: 12px;
                font-weight: bold;
                color: {t['text']};
                background: transparent;
            }}
            QLabel#lock_error {{
                font-size: 12px;
                font-weight: bold;
                color: {t['danger']};
                background: transparent;
                padding: 4px;
            }}
            QLineEdit#lock_input {{
                background: {t['input_bg']};
                border: 2px solid {t['input_border']};
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 14px;
                color: {t['text']};
            }}
            QLineEdit#lock_input:focus {{
                border: 2px solid {t['primary']};
            }}
            QPushButton#lock_primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#lock_primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#lock_secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 12px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#lock_secondary_btn:hover {{
                background: {t['danger']};
                color: {t['button_text']};
                border-color: {t['danger']};
            }}
        """)

    # ============================================================
    # Actions
    # ============================================================
    def _on_unlock(self):
        h = _get_helpers()
        password = self.password_input.text()

        if not password:
            self._show_error(self._tr(self._lang, "empty_fields"))
            return

        import time
        now = time.time()
        self._attempts = [t for t in self._attempts if now - t < 300]
        if len(self._attempts) >= 5:
            self._show_error(self._tr(self._lang, "too_many_attempts"))
            return

        try:
            ok = self._db.verify(self._username, password)
        except Exception as e:
            print(f"⚠️ verify failed: {e}")
            ok = False

        if ok:
            h["SoundManager"].get().play("ok_sound")
            self.password_input.clear()
            self._attempts = []
            self.unlock_successful.emit()
            h["animate_dialog_close"](self, on_finished=self.accept)
        else:
            self._attempts.append(now)
            h["SoundManager"].get().play("warningSound")
            self._show_error(self._tr(self._lang, "invalid_credentials"))
            self.password_input.clear()
            self.password_input.setFocus()

    def _on_logout(self):
        h = _get_helpers()
        h["SoundManager"].get().play("logoutSound")
        self.logout_requested.emit()
        h["animate_dialog_close"](self, on_finished=self.reject)

    def _show_error(self, msg: str):
        def _hide():
            try:
                self.error_label.setVisible(False)
            except RuntimeError:
                pass
        try:
            self.error_label.setText(f"⚠️  {msg}")
            self.error_label.setVisible(True)
            QTimer.singleShot(4000, _hide)
        except RuntimeError:
            pass

    # ============================================================
    # Layout & Events
    # ============================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "backdrop") and self.backdrop is not None:
            try:
                self.backdrop.setGeometry(self.rect())
                self.backdrop.lower()
            except RuntimeError:
                pass

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "backdrop") and self.backdrop is not None:
            try:
                self.backdrop.setGeometry(self.rect())
                self.backdrop.lower()
                self.card.raise_()
            except RuntimeError:
                pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        event.ignore()


# ============================================================
# Opaque backdrop (theme-colored)
# ============================================================

class _OpaqueBackdrop(QWidget):
    def __init__(self, theme: dict, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._bg_color = self._compute_bg_color(theme)

    @staticmethod
    def _compute_bg_color(theme: dict) -> QColor:
        try:
            base_hex = theme.get("bg_gradient_mid", "#302b63")
            base = QColor(base_hex)
            r = int(base.red() * 0.65)
            g = int(base.green() * 0.65)
            b = int(base.blue() * 0.65)
            return QColor(r, g, b, 255)
        except Exception:
            return QColor(25, 22, 50, 255)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), self._bg_color)
        painter.end()
