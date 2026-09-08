#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Prefer python3.12 / python3.11 if available
PYTHON=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    VER=$("$c" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$("$c" -c 'import sys; print(sys.version_info.major)')
    MINOR=$("$c" -c 'import sys; print(sys.version_info.minor)')
    if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; }; then
      PYTHON="$c"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3.11+ is required. Install Python 3.11 or newer and retry."
  exit 1
fi

echo "Using $PYTHON ($($PYTHON --version))"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
echo ""
echo "Setup complete. Place PO workbooks in data/raw/ then run:"
echo "  source .venv/bin/activate"
echo "  python -m parts_inflation.cli run --input-dir data/raw --config config/model_config.xlsx --output-dir outputs --target-date 2027-07-09"
echo "Or double-click run_mac.command"
