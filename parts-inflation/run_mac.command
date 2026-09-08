#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run setup_mac.sh first."
  exit 1
fi

# Prefer python3.12/3.11 from PATH via venv
# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import parts_inflation" 2>/dev/null; then
  echo "ERROR: parts_inflation package not installed in .venv. Run setup_mac.sh."
  exit 1
fi

echo "Running parts inflation prototype..."
python -m parts_inflation.cli run --input-dir data/raw --config config/model_config.xlsx --output-dir outputs
STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "SUCCESS. See outputs/ for the Excel report and logs/."
else
  echo "FAILED with exit code $STATUS. Check outputs/logs/."
fi
echo "Press Enter to close..."
read -r _
exit $STATUS
