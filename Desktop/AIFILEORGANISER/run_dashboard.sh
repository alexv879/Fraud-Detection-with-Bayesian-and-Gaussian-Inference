#!/bin/bash
# Private AI File Organiser - Dashboard Launcher (Mac/Linux)

echo "========================================"
echo " AI File Organiser - Dashboard"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "Starting dashboard..."
echo ""
echo "The dashboard will open at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python3 src/main.py dashboard
