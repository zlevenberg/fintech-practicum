#!/usr/bin/env bash
# Idempotent repository bootstrap for the parts-inflation Python project.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$REPO_ROOT/parts-inflation"
cd "$PROJECT_DIR"

# The default image ships python3.12 but not the venv/ensurepip module.
if ! python3.12 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3.12-venv
fi

if [ ! -x ".venv/bin/python" ]; then
  python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "parts-inflation environment ready."
echo "Activate with: source parts-inflation/.venv/bin/activate"
