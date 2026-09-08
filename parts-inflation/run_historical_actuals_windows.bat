@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: .venv not found. Run setup_windows.bat first.
  exit /b 1
)

call .venv\Scripts\activate.bat
python -c "import parts_inflation" 2>nul
if errorlevel 1 (
  echo ERROR: parts_inflation package not installed in .venv. Run setup_windows.bat.
  exit /b 1
)

echo Running Historical Actual Inflation analysis...
python -m parts_inflation.cli historical-actuals --input-dir data\raw --config config\model_config.xlsx --output-dir outputs
set STATUS=%ERRORLEVEL%
echo.
if %STATUS%==0 (
  echo SUCCESS. See outputs\historical_actual_inflation_*.xlsx and historical_actual_inflation_summary.md
) else (
  echo FAILED with exit code %STATUS%. Check outputs\logs\.
)
pause
exit /b %STATUS%
