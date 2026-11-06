# gui/dialogs/update_dialogs.py
# -*- coding: utf-8 -*-
"""
Beautiful, theme-aware update dialogs.

Replaces the default QMessageBox/QProgressDialog with
custom glassmorphism dialogs that match the app theme.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QTextBrowser, QWidget, QScrollArea,
    QGraphicsOpacityEffect, QApplication,
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QRect,
)
from PyQt5.QtGui import QFont, QColor, QPixmap

# ============================================================
# ✅ Backdrop helper — attaches backdrop to any dialog
# ============================================================

def _with_backdrop(dialog, parent, theme):
    """
    Show an opaque backdrop behind the dialog.
    Backdrop is automatically removed when dialog closes.

    Usage:
        _with_backdrop(my_dialog, parent, theme)
    """
    try:
        from gui.dialogs.backdrop import DialogBackdrop, center_on_parent

        # ✅ Create backdrop on parent (or on screen if no parent)
        backdrop_parent = parent if parent is not None else None

        backdrop = DialogBackdrop(backdrop_parent, theme, opacity=0.88)
        backdrop.show_backdrop()

        # ✅ If dialog has a parent, center on parent; else center on screen
        if parent is not None:
            center_on_parent(dialog, parent)
        else:
            from gui.dialogs.backdrop import center_on_screen
            center_on_screen(dialog)

        # ✅ Store reference so it doesn't get garbage collected
        dialog._backdrop = backdrop

        # ✅ When dialog closes, hide backdrop
        def _on_dialog_finished(*args, **kwargs):
            try:
                backdrop.hide_backdrop()
                # Cleanup after animation
                QTimer.singleShot(300, lambda: (
                    backdrop.deleteLater() if backdrop else None
                ))
            except Exception:
                pass

        dialog.finished.connect(_on_dialog_finished)

        return backdrop
    except Exception as e:
        print(f"⚠️ _with_backdrop failed: {e}")
        return None


# ============================================================
# ✅ Helpers (imported lazily to avoid circular import)
# ============================================================

def _get_helpers():
    from GUI_update import (
        tr, THEMES, make_tooltip,
        animate_dialog_open, animate_dialog_close,
        attach_sound, SoundManager,
    )
    return {
        "tr": tr,
        "THEMES": THEMES,
        "make_tooltip": make_tooltip,
        "animate_dialog_open": animate_dialog_open,
        "animate_dialog_close": animate_dialog_close,
        "attach_sound": attach_sound,
        "SoundManager": SoundManager,
    }


# ============================================================
# ✅ UpdateAvailableDialog — replaces ask_question()
# ============================================================

class UpdateAvailableDialog(QDialog):
    """
    Beautiful dialog showing update info with Download / Cancel buttons.

    Usage:
        dlg = UpdateAvailableDialog(info, current_version, theme, lang, parent)
        if dlg.exec_() == QDialog.Accepted:
            # user clicked Download
    """

    def __init__(self, info, current_version: str, theme: dict,
                 lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._info = info
        self._current_version = current_version

        self.setWindowTitle(self._tr(lang, "update_available"))
        self.setFixedSize(560, 620)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._build_ui()
        self._apply_theme()

        # ✅ Add backdrop
        _with_backdrop(self, parent, theme)

        # ✅ Animate open
        QTimer.singleShot(0, lambda: h["animate_dialog_open"](self))

        # ✅ Play sound
        try:
            h["SoundManager"].get().play("ok_sound")
        except Exception:
            pass


    def _build_ui(self):
        tr = self._tr
        lang = self._lang

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # ============================================================
        # Header: icon + title + version
        # ============================================================
        header = QHBoxLayout()
        header.setSpacing(14)

        self.icon_label = QLabel("🚀")
        self.icon_label.setFixedSize(64, 64)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setObjectName("update_icon")
        header.addWidget(self.icon_label)

        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        self.title_label = QLabel(tr(lang, "update_available"))
        self.title_label.setObjectName("update_title")

        self.version_label = QLabel(
            f"v{self._current_version}  →  v{self._info.version}"
        )
        self.version_label.setObjectName("update_version")

        title_box.addWidget(self.title_label)
        title_box.addWidget(self.version_label)
        header.addLayout(title_box, 1)

        layout.addLayout(header)

        # ============================================================
        # Meta row: size + critical badge
        # ============================================================
        meta = QHBoxLayout()
        meta.setSpacing(12)

        size_label = QLabel(f"📦 {self._info.human_size}")
        size_label.setObjectName("update_meta")
        meta.addWidget(size_label)

        if self._info.is_critical:
            badge = QLabel("⚠️ " + tr(lang, "critical_update"))
            badge.setObjectName("update_critical_badge")
            meta.addWidget(badge)

        meta.addStretch()
        layout.addLayout(meta)

        # ============================================================
        # Changelog
        # ============================================================
        cl_title = QLabel("✨ " + tr(lang, "whats_new"))
        cl_title.setObjectName("update_section")
        layout.addWidget(cl_title)

        self.changelog = QTextBrowser()
        self.changelog.setObjectName("update_changelog")
        self.changelog.setOpenExternalLinks(True)
        self.changelog.setMinimumHeight(180)
        self.changelog.setHtml(self._changelog_to_html(self._info.changelog))
        layout.addWidget(self.changelog, 1)

        # ============================================================
        # Buttons
        # ============================================================
        buttons = QHBoxLayout()
        buttons.setSpacing(12)

        self.cancel_btn = QPushButton("✕  " + tr(lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)

        self.download_btn = QPushButton(
            f"⬇️  {tr(lang, 'download_now')} ({self._info.human_size})"
        )
        self.download_btn.setObjectName("primary_btn")
        self.download_btn.setMinimumHeight(44)
        self.download_btn.setCursor(Qt.PointingHandCursor)
        self.download_btn.clicked.connect(self._on_download)

        buttons.addWidget(self.cancel_btn, 1)
        buttons.addWidget(self.download_btn, 2)
        layout.addLayout(buttons)

        # ✅ Close button (top-right)
        self._close_btn = QPushButton("✕")
        self._close_btn.setObjectName("close_btn")
        self._close_btn.setFixedSize(30, 30)
        self._close_btn.clicked.connect(self._on_cancel)
        self._close_btn.setParent(self.container)
        self._close_btn.move(520, 10)

    def _on_download(self):
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("ok_sound")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=self.accept)

    def _on_cancel(self):
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("simpleClick")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=self.reject)

    def _changelog_to_html(self, text: str) -> str:
        t = self._theme
        text_color = t.get("text", "#f1f5f9")
        primary = t.get("primary", "#00e5ff")
        if not text:
            return f"<p style='color:{text_color};'>{self._tr(self._lang, 'no_changelog')}</p>"

        import re
        html = text
        html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html)
        html = re.sub(r"`(.+?)`", rf"<code style='color:{primary};'>\1</code>", html)
        html = re.sub(r"^### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^- (.+)$", r"• \1", html, flags=re.MULTILINE)
        html = html.replace("\n", "<br>")
        return f"<div style='color:{text_color};font-size:12px;'>{html}</div>"

    def _apply_theme(self):
        t = self._theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#update_icon {{
                font-size: 42px;
                background: {t['input_bg']};
                border: 2px solid {t['primary']};
                border-radius: 16px;
            }}
            QLabel#update_title {{
                font-size: 22px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#update_version {{
                font-size: 13px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QLabel#update_meta {{
                font-size: 12px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QLabel#update_critical_badge {{
                font-size: 11px;
                font-weight: bold;
                color: {t['danger']};
                background: rgba(248, 113, 113, 0.15);
                border: 1px solid {t['danger']};
                border-radius: 6px;
                padding: 3px 10px;
            }}
            QLabel#update_section {{
                font-size: 13px;
                font-weight: bold;
                color: {t['text']};
                background: transparent;
                margin-top: 4px;
            }}
            QTextBrowser#update_changelog {{
                background: {t['input_bg']};
                border: 1px solid {t['input_border']};
                border-radius: 10px;
                color: {t['text']};
                padding: 10px;
                font-size: 12px;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)


# ============================================================
# ✅ UpdateDownloadDialog — replaces QProgressDialog
# ============================================================

class UpdateDownloadDialog(QDialog):
    """
    Beautiful download dialog with animated progress bar.
    Handles download internally and emits signals.

    Signals:
        download_ready(str path) — when download completes
        download_failed(str reason) — when download fails
        canceled() — when user cancels
    """

    download_ready = pyqtSignal(str)
    download_failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(self, info, theme: dict, lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._info = info
        self._downloader = None
        self._is_done = False

        self.setWindowTitle(self._tr(lang, "downloading"))
        self.setFixedSize(520, 320)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._build_ui()
        self._apply_theme()

        # ✅ Add backdrop
        _with_backdrop(self, parent, theme)

        QTimer.singleShot(0, lambda: h["animate_dialog_open"](self))

    def _build_ui(self):
        tr = self._tr
        lang = self._lang

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(18)

        # Header
        header = QHBoxLayout()
        header.setSpacing(14)

        self.icon_label = QLabel("⬇️")
        self.icon_label.setFixedSize(56, 56)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setObjectName("dl_icon")
        header.addWidget(self.icon_label)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        self.title_label = QLabel(tr(lang, "downloading"))
        self.title_label.setObjectName("dl_title")
        self.version_label = QLabel(f"v{self._info.version}")
        self.version_label.setObjectName("dl_version")
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.version_label)
        header.addLayout(title_box, 1)

        layout.addLayout(header)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("dl_progress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        # Stats row
        stats = QHBoxLayout()
        stats.setSpacing(8)

        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("dl_percent")

        self.size_label = QLabel(f"0 KB / {self._info.human_size}")
        self.size_label.setObjectName("dl_size")
        self.size_label.setAlignment(Qt.AlignRight)

        stats.addWidget(self.percent_label)
        stats.addStretch()
        stats.addWidget(self.size_label)
        layout.addLayout(stats)

        # Cancel button
        buttons = QHBoxLayout()
        buttons.addStretch()

        self.cancel_btn = QPushButton("✕  " + tr(lang, "cancel"))
        self.cancel_btn.setObjectName("secondary_btn")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._on_cancel)
        buttons.addWidget(self.cancel_btn)

        layout.addLayout(buttons)

    def start_download(self):
        """Begin the download."""
        from core.updater import UpdateDownloader

        self._downloader = UpdateDownloader(
            url=self._info.download_url,
            expected_size=self._info.download_size,
        )
        self._downloader.progress.connect(self._on_progress)
        self._downloader.finished_ok.connect(self._on_done)
        self._downloader.failed.connect(self._on_failed)
        self._downloader.start()

    def _on_progress(self, percent, downloaded, total):
        if self._is_done:
            return
        self.progress_bar.setValue(percent)
        self.percent_label.setText(f"{percent}%")

        if total > 0:
            mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self.size_label.setText(f"{mb:.1f} MB / {total_mb:.1f} MB")
        else:
            kb = downloaded / 1024
            self.size_label.setText(f"{kb:.0f} KB")

    def _on_done(self, path):
        if self._is_done:
            return
        self._is_done = True
        self.progress_bar.setValue(100)
        self.percent_label.setText("100%")
        self.title_label.setText("✅ " + self._tr(self._lang, "update_ready"))
        self._icon_finish()

        # Animate close and emit
        h = _get_helpers()
        QTimer.singleShot(600, lambda: self._emit_and_close(path))

    def _emit_and_close(self, path):
        h = _get_helpers()
        h["animate_dialog_close"](self, on_finished=lambda: (
            self.download_ready.emit(path),
            self.accept(),
        ))

    def _icon_finish(self):
        try:
            t = self._theme
            self.icon_label.setStyleSheet(f"""
                QLabel#dl_icon {{
                    font-size: 32px;
                    background: {t['success']};
                    border-radius: 14px;
                }}
            """)
        except Exception:
            pass

    def _on_failed(self, reason):
        if self._is_done:
            return
        self._is_done = True
        h = _get_helpers()
        h["animate_dialog_close"](self, on_finished=lambda: (
            self.download_failed.emit(reason),
            self.reject(),
        ))

    def _on_cancel(self):
        if self._is_done:
            return
        self._is_done = True
        if self._downloader is not None:
            self._downloader.cancel()
            self._downloader.wait(2000)
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("simpleClick")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=lambda: (
            self.canceled.emit(),
            self.reject(),
        ))

    def closeEvent(self, event):
        if not self._is_done and self._downloader is not None:
            self._downloader.cancel()
            self._downloader.wait(2000)
        super().closeEvent(event)

    def _apply_theme(self):
        t = self._theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#dl_icon {{
                font-size: 32px;
                background: {t['input_bg']};
                border: 2px solid {t['primary']};
                border-radius: 14px;
            }}
            QLabel#dl_title {{
                font-size: 20px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#dl_version {{
                font-size: 12px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QProgressBar#dl_progress {{
                background: {t['input_bg']};
                border: 1px solid {t['glass_border']};
                border-radius: 7px;
                text-align: center;
            }}
            QProgressBar#dl_progress::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t['primary']},
                    stop:1 {t['secondary']}
                );
                border-radius: 6px;
            }}
            QLabel#dl_percent {{
                font-size: 13px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#dl_size {{
                font-size: 12px;
                color: {t['text_dim']};
                background: transparent;
            }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 12px;
                padding: 10px 24px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['danger']};
                color: {t['button_text']};
                border-color: {t['danger']};
            }}
        """)


# ============================================================
# ✅ ConfirmInstallDialog — replaces ask_question() for install
# ============================================================

class ConfirmInstallDialog(QDialog):
    """Beautiful dialog asking user to confirm install."""

    def __init__(self, info, theme: dict, lang: str = "English", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._info = info

        self.setWindowTitle(self._tr(lang, "update_ready"))
        self.setFixedSize(520, 400)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._build_ui()
        self._apply_theme()

        # ✅ Add backdrop
        _with_backdrop(self, parent, theme)

        QTimer.singleShot(0, lambda: h["animate_dialog_open"](self))

        try:
            h["SoundManager"].get().play("ok_sound")
        except Exception:
            pass

    def _build_ui(self):
        tr = self._tr
        lang = self._lang

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # Icon
        icon = QLabel("✨")
        icon.setFixedSize(72, 72)
        icon.setAlignment(Qt.AlignCenter)
        icon.setObjectName("install_icon")
        layout.addWidget(icon, 0, Qt.AlignCenter)

        # Title
        title = QLabel(tr(lang, "update_ready") + "!")
        title.setObjectName("install_title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Message
        msg = QLabel(
            f"v{self._info.version}\n\n" + tr(lang, "app_will_restart")
        )
        msg.setObjectName("install_msg")
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)
        layout.addWidget(msg)

        layout.addStretch()

        # Buttons
        buttons = QHBoxLayout()
        buttons.setSpacing(12)

        self.later_btn = QPushButton("⏰  " + tr(lang, "remind_later"))
        self.later_btn.setObjectName("secondary_btn")
        self.later_btn.setMinimumHeight(44)
        self.later_btn.setCursor(Qt.PointingHandCursor)
        self.later_btn.clicked.connect(self._on_later)

        self.install_btn = QPushButton("🚀  " + tr(lang, "install_now"))
        self.install_btn.setObjectName("primary_btn")
        self.install_btn.setMinimumHeight(44)
        self.install_btn.setCursor(Qt.PointingHandCursor)
        self.install_btn.clicked.connect(self._on_install)

        buttons.addWidget(self.later_btn, 1)
        buttons.addWidget(self.install_btn, 2)
        layout.addLayout(buttons)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self._on_later)
        close_btn.setParent(self.container)
        close_btn.move(480, 10)

    def _on_install(self):
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("ok_sound")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=self.accept)

    def _on_later(self):
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("simpleClick")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=self.reject)

    def _apply_theme(self):
        t = self._theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#install_icon {{
                font-size: 52px;
                background: {t['input_bg']};
                border: 2px solid {t['primary']};
                border-radius: 20px;
            }}
            QLabel#install_title {{
                font-size: 22px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#install_msg {{
                font-size: 13px;
                color: {t['text']};
                background: transparent;
                line-height: 1.5;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#secondary_btn {{
                background: {t['input_bg']};
                color: {t['text']};
                border: 2px solid {t['input_border']};
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#secondary_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)


# ============================================================
# ✅ SimpleInfoDialog — for "up to date" / "check failed"
# ============================================================

class SimpleInfoDialog(QDialog):
    """Beautiful dialog for simple info messages."""

    def __init__(self, title: str, message: str, theme: dict,
                 lang: str = "English", icon: str = "ℹ️",
                 parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._icon = icon

        self.setWindowTitle(title)
        self.setFixedSize(480, 320)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._build_ui(title, message)
        self._apply_theme()

        # ✅ Add backdrop
        _with_backdrop(self, parent, theme)

        QTimer.singleShot(0, lambda: h["animate_dialog_open"](self))

        try:
            h["SoundManager"].get().play("ok_sound" if icon == "ℹ️" else "warningSound")
        except Exception:
            pass

    def _build_ui(self, title: str, message: str):
        tr = self._tr

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # Icon
        icon_label = QLabel(self._icon)
        icon_label.setFixedSize(64, 64)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setObjectName("info_icon")
        layout.addWidget(icon_label, 0, Qt.AlignCenter)

        # Title
        title_label = QLabel(title)
        title_label.setObjectName("info_title")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Message
        msg_label = QLabel(message)
        msg_label.setObjectName("info_msg")
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)

        layout.addStretch()

        # OK button
        btn = QPushButton("OK")
        btn.setObjectName("primary_btn")
        btn.setMinimumHeight(42)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(self._on_ok)
        layout.addWidget(btn, 0, Qt.AlignCenter)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("close_btn")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self._on_ok)
        close_btn.setParent(self.container)
        close_btn.move(440, 10)

    def _on_ok(self):
        h = _get_helpers()
        try:
            h["SoundManager"].get().play("simpleClick")
        except Exception:
            pass
        h["animate_dialog_close"](self, on_finished=self.accept)

    def _apply_theme(self):
        t = self._theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#info_icon {{
                font-size: 44px;
                background: {t['input_bg']};
                border: 2px solid {t['primary']};
                border-radius: 18px;
            }}
            QLabel#info_title {{
                font-size: 20px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#info_msg {{
                font-size: 13px;
                color: {t['text']};
                background: transparent;
            }}
            QPushButton#primary_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']},
                    stop:0.5 {t['accent']},
                    stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 12px;
                padding: 10px 40px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }}
            QPushButton#primary_btn:hover {{
                background: {t['button_hover']};
            }}
            QPushButton#close_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: none;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton#close_btn:hover {{
                color: {t['danger']};
            }}
        """)
