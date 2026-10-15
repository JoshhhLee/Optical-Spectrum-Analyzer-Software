"""Dark Fusion theme for the application chrome.

Only the interactive pyqtgraph plots follow the dark theme. Matplotlib
previews stay white because they show the exported figure as it will look.
"""

from PyQt5.QtGui import QColor, QFont, QPalette

C = {
    "bg": "#1E1E1E", "panel": "#2B2B2B", "widget": "#383838", "hover": "#454545",
    "primary": "#1976D2", "primary_hover": "#1565C0", "accent": "#64B5F6",
    "success": "#43A047", "warning": "#FB8C00", "danger": "#E53935",
    "text": "#F0F0F0", "muted": "#A8A8A8", "border": "#555555",
}


def apply(app) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 9))
    pal = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"),
                      (QPalette.Base, "widget"), (QPalette.AlternateBase, "panel"),
                      (QPalette.Text, "text"), (QPalette.Button, "panel"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "primary"),
                      (QPalette.ToolTipBase, "panel"), (QPalette.ToolTipText, "text")):
        pal.setColor(role, QColor(C[key]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#777777"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#777777"))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QGroupBox {{
    font-weight: bold; border: 1px solid {C['border']}; border-radius: 6px;
    margin-top: 12px; padding: 8px 6px 6px 6px; background: {C['panel']};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {C['accent']}; }}
QPushButton {{
    background: {C['widget']}; color: {C['text']}; border: 1px solid {C['border']};
    border-radius: 5px; padding: 6px 10px; min-height: 22px;
}}
QPushButton:hover {{ background: {C['hover']}; border-color: {C['primary']}; }}
QPushButton:disabled {{ color: #777777; background: {C['panel']}; }}
QPushButton#primary {{
    background: {C['primary']}; color: white; font-weight: bold; border: none; min-height: 32px;
}}
QPushButton#primary:hover {{ background: {C['primary_hover']}; }}
QPushButton#primary:disabled {{ background: #3A4A5A; color: #9AA; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {C['widget']}; color: {C['text']}; border: 1px solid {C['border']};
    border-radius: 4px; padding: 3px 5px; min-height: 20px;
}}
QTabWidget::pane {{ border: 1px solid {C['border']}; background: {C['panel']}; }}
QTabBar::tab {{
    background: {C['widget']}; color: {C['muted']}; border: 1px solid {C['border']};
    padding: 7px 14px; margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {C['primary']}; color: white; }}
QTextEdit, QPlainTextEdit, QListWidget, QTableWidget {{
    background: {C['widget']}; color: {C['text']}; border: 1px solid {C['border']};
    border-radius: 4px; gridline-color: {C['border']};
}}
QHeaderView::section {{
    background: {C['panel']}; color: {C['text']}; padding: 4px; border: 1px solid {C['border']};
}}
QLabel#note {{ color: {C['muted']}; font-size: 9pt; }}
QLabel#warning {{ color: {C['warning']}; font-size: 9pt; }}
QProgressBar {{ border: 1px solid {C['border']}; border-radius: 4px; text-align: center; }}
QProgressBar::chunk {{ background: {C['primary']}; }}
"""
