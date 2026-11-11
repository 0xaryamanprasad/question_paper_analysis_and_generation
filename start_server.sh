#!/usr/bin/env bash
set -e

# Change to project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo " Starting PaperMind Studio Local Server"
echo "=================================================="
echo " Project Directory: $PROJECT_DIR"
echo " Dashboard & API:   http://localhost:8000"
echo " Interactive Docs:  http://localhost:8000/docs"
echo "=================================================="

# Check virtual environment
if [ -d "$PROJECT_DIR/.venv" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    PYTHON_EXEC="python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "Error: Python 3 not found."
    exit 1
fi

# Run FastAPI backend with Uvicorn
exec "$PYTHON_EXEC" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
