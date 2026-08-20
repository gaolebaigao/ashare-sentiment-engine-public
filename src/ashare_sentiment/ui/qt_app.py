"""Qt application entry point with a friendly optional-dependency fallback."""

from __future__ import annotations

from pathlib import Path


def run_gui(config_path: str = "config/default.yaml", theme: str = "system") -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Desktop GUI requires PySide6. Install it with: pip install -e '.[desktop]'")
        return 2
    from ..config import load_config
    from ..application.service import AdvisoryService
    from .main_window import MainWindow

    config = load_config(config_path)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("MarketTemperature")
    from PySide6.QtGui import QFontDatabase

    app.setFont(QFontDatabase.systemFont(QFontDatabase.GeneralFont))
    window = MainWindow(AdvisoryService(config), str(Path(config_path)), theme=theme)
    window.show()
    return app.exec()
