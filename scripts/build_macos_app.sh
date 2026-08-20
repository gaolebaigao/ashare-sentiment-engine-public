#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m PyInstaller \
  --windowed \
  --name MarketTemperature \
  --paths src \
  --collect-all PySide6 \
  --add-data "config:config" \
  --add-data "data/processed:data/processed" \
  --add-data "reports:reports" \
  scripts/desktop_entry.py

echo "Built dist/MarketTemperature.app"
