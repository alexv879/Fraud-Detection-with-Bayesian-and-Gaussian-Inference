@echo off
REM Private AI File Organiser - Dashboard Launcher
REM This script launches the web dashboard

echo ========================================
echo  AI File Organiser - Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Starting dashboard...
echo.
echo The dashboard will open at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python src\main.py dashboard

pause
