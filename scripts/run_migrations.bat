@echo off
REM Windows batch script to run database migrations

echo ============================================================
echo ShelfGuard - Database Migrations
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Running migration script...
echo.

python "%~dp0run_migrations.py"

echo.
echo ============================================================
echo Migration script completed
echo ============================================================
echo.
pause
