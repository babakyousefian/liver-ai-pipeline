# gui/dialogs/audit_log_dialog.py
# -*- coding: utf-8 -*-
"""
Audit Log dialog — theme-aware, filterable, RTL-aware.
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QFileDialog,
    QWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont

from gui.dialogs.update_dialog import show_message


def _get_helpers():
    from GUI_update import (
        tr, THEMES, animate_dialog_open, animate_dialog_close,
        show_message, attach_sound, SoundManager,
    )
    from core.audit import LEVEL_COLORS, LEVEL_EMOJI, CATEGORY_EMOJI
    return {
        "tr": tr,
        "THEMES": THEMES,
        "animate_dialog_open": animate_dialog_open,
        "animate_dialog_close": animate_dialog_close,
        "show_message": show_message,
        "attach_sound": attach_sound,
        "SoundManager": SoundManager,
        "LEVEL_COLORS": LEVEL_COLORS,
        "LEVEL_EMOJI": LEVEL_EMOJI,
        "CATEGORY_EMOJI": CATEGORY_EMOJI,
    }


class AuditLogDialog(QDialog):
    def __init__(self, theme: dict, lang: str = "English",
                 current_user: str = "", parent=None):
        super().__init__(parent)
        h = _get_helpers()
        self._h = h
        self._tr = h["tr"]
        self._theme = theme
        self._lang = lang
        self._current_user = current_user
        self._all_entries = []

        self.setWindowTitle(self._tr(lang, "audit_log_title"))
        self.setMinimumSize(900, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._setup_ui()
        self._apply_theme()
        self._load_entries()

        # ✅ Add backdrop (same as LoginDialog / SignupDialog / UpdateAvailableDialog)
        try:
            from gui.dialogs.update_dialogs import _with_backdrop
            _with_backdrop(self, parent, theme)
        except Exception as e:
            print(f"⚠️ AuditLogDialog backdrop failed: {e}")

        QTimer.singleShot(0, lambda: h["animate_dialog_open"](self))

    # ============================================================
    # UI
    # ============================================================
    def _setup_ui(self):
        h = self._h                      # ✅ از self._h استفاده کن
        tr = self._tr
        lang = self._lang

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)

        self.container = QFrame()
        self.container.setObjectName("container")
        main.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # ---------- Header ----------
        header = QHBoxLayout()
        title = QLabel("📋 " + tr(lang, "audit_log_title"))
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()

        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setObjectName("icon_btn")
        self.refresh_btn.setFixedSize(32, 32)
        self.refresh_btn.clicked.connect(self._load_entries)
        header.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("💾 " + tr(lang, "audit_export"))
        self.export_btn.setObjectName("action_btn")
        self.export_btn.clicked.connect(self._export)
        header.addWidget(self.export_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("icon_btn")
        self.close_btn.setFixedSize(32, 32)
        self.close_btn.clicked.connect(self._close)
        header.addWidget(self.close_btn)

        layout.addLayout(header)

        # ---------- Filters ----------
        filters = QHBoxLayout()
        filters.setSpacing(8)

        # Level filter
        self.level_combo = QComboBox()
        self.level_combo.setObjectName("filter_combo")
        self.level_combo.addItem("All Levels", "")
        for lvl in ("INFO", "WARNING", "ERROR", "CRITICAL"):
            self.level_combo.addItem(
                f"{h['LEVEL_EMOJI'].get(lvl, '')} {lvl}", lvl
            )
        self.level_combo.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.level_combo)

        # Category filter
        self.cat_combo = QComboBox()
        self.cat_combo.setObjectName("filter_combo")
        self.cat_combo.addItem("All Categories", "")
        for cat in ("AUTH", "DATA", "SETTINGS", "SECURITY", "SYSTEM", "EXPORT"):
            self.cat_combo.addItem(
                f"{h['CATEGORY_EMOJI'].get(cat, '')} {cat}", cat
            )
        self.cat_combo.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.cat_combo)

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setObjectName("filter_input")
        self.search_input.setPlaceholderText(tr(lang, "audit_search_placeholder"))
        self.search_input.textChanged.connect(self._apply_filters)
        filters.addWidget(self.search_input, 1)

        layout.addLayout(filters)

        # ---------- Table ----------
        self.table = QTableWidget()
        self.table.setObjectName("audit_table")
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            tr(lang, "audit_col_time"),
            tr(lang, "audit_col_level"),
            tr(lang, "audit_col_category"),
            tr(lang, "audit_col_action"),
            tr(lang, "audit_col_user"),
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.Stretch)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.itemDoubleClicked.connect(self._show_details)
        layout.addWidget(self.table, 1)

        # ---------- Status ----------
        self.status_label = QLabel("")
        self.status_label.setObjectName("status_label")
        layout.addWidget(self.status_label)

    def _apply_theme(self):
        t = self._theme
        self.container.setStyleSheet(f"""
            QFrame#container {{
                background: {t['glass_bg']};
                border: 2px solid {t['glass_border']};
                border-radius: 20px;
            }}
            QLabel#title {{
                font-size: 22px;
                font-weight: bold;
                color: {t['primary']};
                background: transparent;
            }}
            QLabel#status_label {{
                font-size: 11px;
                color: {t['text_dim']};
                background: transparent;
                padding: 4px;
            }}
            QPushButton#icon_btn {{
                background: transparent;
                color: {t['text_dim']};
                border: 1px solid {t['glass_border']};
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#icon_btn:hover {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QPushButton#action_btn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 {t['primary']}, stop:0.5 {t['accent']}, stop:1 {t['secondary']}
                );
                color: {t['button_text']};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton#action_btn:hover {{
                background: {t['button_hover']};
            }}
            QComboBox#filter_combo {{
                background: {t['input_bg']};
                border: 1px solid {t['input_border']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                color: {t['text']};
                min-height: 20px;
            }}
            QComboBox#filter_combo::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox#filter_combo QAbstractItemView {{
                background: {t['input_bg']};
                color: {t['text']};
                selection-background-color: {t['primary']};
                border: 1px solid {t['input_border']};
            }}
            QLineEdit#filter_input {{
                background: {t['input_bg']};
                border: 1px solid {t['input_border']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                color: {t['text']};
                min-height: 20px;
            }}
            QTableWidget#audit_table {{
                background: {t['input_bg']};
                border: 1px solid {t['glass_border']};
                border-radius: 8px;
                color: {t['text']};
                font-size: 11px;
                gridline-color: {t['glass_border']};
            }}
            QTableWidget#audit_table::item {{
                padding: 4px;
            }}
            QTableWidget#audit_table::item:selected {{
                background: {t['primary']};
                color: {t['button_text']};
            }}
            QHeaderView::section {{
                background: {t['glass_highlight']};
                color: {t['text']};
                border: none;
                border-bottom: 1px solid {t['glass_border']};
                padding: 6px;
                font-size: 11px;
                font-weight: bold;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {t['glass_border']};
                border-radius: 3px;
                min-height: 20px;
            }}
        """)

    # ============================================================
    # Data
    # ============================================================
    def _load_entries(self):
        try:
            from core.audit import audit_log
            audit_log.flush()
            self._all_entries = audit_log.read_all()
            self._apply_filters()
        except Exception as e:
            print(f"⚠️ _load_entries failed: {e}")

    def _apply_filters(self):
        try:
            from core.audit.reader import read_filtered
            level = self.level_combo.currentData() or None
            category = self.cat_combo.currentData() or None
            search = self.search_input.text().strip() or None

            user_filter = None
            if self._current_user and self._current_user != "admin":
                user_filter = self._current_user

            filtered = read_filtered(
                self._all_entries,
                level=level,
                category=category,
                user=user_filter,
                search=search,
            )
            self._populate_table(filtered)
        except Exception as e:
            print(f"⚠️ _apply_filters failed: {e}")

    def _populate_table(self, entries):
        h = self._h                      # ✅ از self._h استفاده کن
        try:
            self.table.setRowCount(len(entries))
            for i, e in enumerate(entries):
                time_item = QTableWidgetItem(e.display_datetime)
                self.table.setItem(i, 0, time_item)

                level_text = f"{h['LEVEL_EMOJI'].get(e.level, '')} {e.level}"
                level_item = QTableWidgetItem(level_text)
                try:
                    color = QColor(h["LEVEL_COLORS"].get(e.level, "#00e5ff"))
                    level_item.setForeground(color)
                except Exception:
                    pass
                self.table.setItem(i, 1, level_item)

                cat_text = f"{h['CATEGORY_EMOJI'].get(e.category, '')} {e.category}"
                self.table.setItem(i, 2, QTableWidgetItem(cat_text))

                self.table.setItem(i, 3, QTableWidgetItem(e.action))
                self.table.setItem(i, 4, QTableWidgetItem(e.user or "—"))

            self.status_label.setText(
                self._tr(self._lang, "audit_status").format(
                    count=len(entries),
                    total=len(self._all_entries),
                )
            )
        except Exception as e:
            print(f"⚠️ _populate_table failed: {e}")

    def _show_details(self, item):
        try:
            from core.audit.reader import read_filtered
            row = item.row()
            level = self.level_combo.currentData() or None
            category = self.cat_combo.currentData() or None
            search = self.search_input.text().strip() or None
            user_filter = None
            if self._current_user and self._current_user != "admin":
                user_filter = self._current_user

            filtered = read_filtered(
                self._all_entries,
                level=level,
                category=category,
                user=user_filter,
                search=search,
            )
            if row < 0 or row >= len(filtered):
                return
            e = filtered[row]

            import json
            details_str = json.dumps(e.details, indent=2, ensure_ascii=False, default=str)
            show_message(
                self, "information",
                self._tr(self._lang, "audit_details_title"),
                (
                    f"Time:     {e.display_datetime}\n"
                    f"Level:    {e.level}\n"
                    f"Category: {e.category}\n"
                    f"Action:   {e.action}\n"
                    f"User:     {e.user or '—'}\n"
                    f"Session:  {e.session_id or '—'}\n\n"
                    f"Details:\n{details_str}"
                ),
            )
        except Exception as ex:
            print(f"⚠️ _show_details failed: {ex}")

    # ============================================================
    # Export
    # ============================================================
    def _export(self, show_message=None):
        try:
            tr = self._tr
            from core.audit import audit_log

            save_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            if not os.path.isdir(save_dir):
                save_dir = os.path.expanduser("~")

            from datetime import datetime
            default_name = f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            output_path, _ = QFileDialog.getSaveFileName(
                self,
                tr(self._lang, "audit_export"),
                os.path.join(save_dir, default_name),
                "CSV Files (*.csv);;JSON Files (*.json)",
            )
            if not output_path:
                return

            from core.audit.reader import read_filtered
            level = self.level_combo.currentData() or None
            category = self.cat_combo.currentData() or None
            search = self.search_input.text().strip() or None
            user_filter = None
            if self._current_user and self._current_user != "admin":
                user_filter = self._current_user

            filtered = read_filtered(
                self._all_entries,
                level=level,
                category=category,
                user=user_filter,
                search=search,
            )

            if output_path.lower().endswith(".json"):
                ok = audit_log.export_json(output_path, filtered)
            else:
                ok = audit_log.export_csv(output_path, filtered)

            if ok:
                show_message(
                    self, "information",
                    tr(self._lang, "success"),
                    f"{tr(self._lang, 'audit_exported')}:\n{output_path}",
                )
            else:
                from GUI_update import show_message
                show_message(
                    self, "critical",
                    tr(self._lang, "error"),
                    tr(self._lang, "audit_export_failed"),
                )
        except Exception as e:
            print(f"⚠️ _export failed: {e}")

    # ============================================================
    # Close
    # ============================================================
    def _close(self):
        try:
            self._h["animate_dialog_close"](self, on_finished=self.accept)
        except Exception:
            self.accept()
