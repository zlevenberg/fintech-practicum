@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python launcher py not found. Install Python 3.11+ from python.org and retry.
  exit /b 1
)

py -3.11 -c "import sys" >nul 2>&1
if errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Python 3.11+ is required.
    exit /b 1
  )
  set PY=py -3.12
) else (
  set PY=py -3.11
)

echo Using %PY%
%PY% -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
echo.
echo Setup complete. Place PO workbooks in data\raw\ then run:
echo   .venv\Scripts\activate
echo   python -m parts_inflation.cli run --input-dir data/raw --config config/model_config.xlsx --output-dir outputs --target-date 2027-07-09
echo Or double-click run_windows.bat
