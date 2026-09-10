#!/usr/bin/env bash
# Quickstart script for Bluff Bot (Local Development & Testing)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "===================================================="
echo "          Bluff Bot — Unified Local Launcher        "
echo "===================================================="

if [ "$1" == "--docker" ]; then
    echo "[*] Launching full stack with Docker Compose..."
    docker compose up --build
    exit 0
fi

echo "[1/3] Checking Python environment..."
python -c "import fastapi, uvicorn, torch; print('  ✓ Python packages ready')"

echo "[2/3] Starting FastAPI backend on port 8000..."
nohup python -m uvicorn server:app --host 127.0.0.1 --port 8000 > /tmp/bluff_backend.log 2>&1 &
BACKEND_PID=$!
echo "  ✓ Backend running (PID $BACKEND_PID, logs at /tmp/bluff_backend.log)"

echo "[3/3] Starting Next.js frontend on port 3000..."
cd frontend
nohup npm run dev -- --port 3000 > /tmp/bluff_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "  ✓ Frontend running (PID $FRONTEND_PID, logs at /tmp/bluff_frontend.log)"

echo ""
echo "===================================================="
echo "  Web UI:    http://localhost:3000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Stop with: kill $BACKEND_PID $FRONTEND_PID"
echo "===================================================="
