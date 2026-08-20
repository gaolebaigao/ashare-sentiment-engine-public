"""PyInstaller entry point for the MarketTemperature macOS application."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ashare_sentiment.ui.qt_app import run_gui


def main() -> int:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    if getattr(sys, "frozen", False):
        app_root = Path.home() / "Library" / "Application Support" / "MarketTemperature"
        app_root.mkdir(parents=True, exist_ok=True)
        for directory in ("config", "data", "reports"):
            source = bundle_root / directory
            target = app_root / directory
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
        os.chdir(app_root)
        config_path = app_root / "config" / "default.yaml"
    else:
        os.chdir(bundle_root)
        config_path = bundle_root / "config" / "default.yaml"
    return run_gui(str(config_path), "system")


if __name__ == "__main__":
    raise SystemExit(main())
