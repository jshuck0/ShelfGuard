#!/bin/bash
# Unix/Mac shell script to run database migrations

echo "============================================================"
echo "ShelfGuard - Database Migrations"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "Running migration script..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python migration script
python3 "$SCRIPT_DIR/run_migrations.py"

echo ""
echo "============================================================"
echo "Migration script completed"
echo "============================================================"
echo ""
