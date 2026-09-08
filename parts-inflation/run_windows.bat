@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run setup_windows.bat first.
  exit /b 1
)

.venv\Scripts\python.exe -c "import parts_inflation" >nul 2>&1
if errorlevel 1 (
  echo ERROR: parts_inflation package not installed. Run setup_windows.bat.
  exit /b 1
)

echo Running parts inflation prototype...
.venv\Scripts\python.exe -m parts_inflation.cli run --input-dir data/raw --config config/model_config.xlsx --output-dir outputs
set STATUS=%ERRORLEVEL%
echo.
if %STATUS%==0 (
  echo SUCCESS. See outputs\ for the Excel report and logs\.
) else (
  echo FAILED with exit code %STATUS%. Check outputs\logs\.
)
echo Press any key to close...
pause >nul
exit /b %STATUS%
