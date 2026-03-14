#!/bin/bash
# ===================================================================
# Gossip Handshake Protocol — POC Start Script
#
# Starts both the FastAPI backend and React frontend.
# Usage: ./start.sh [0.5B|1.5B]
# ===================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SCALE="${1:-0.5B}"

echo "🤝 Gossip Handshake Protocol — POC"
echo "==================================="
echo "Model scale: ${MODEL_SCALE}"
echo ""

# --- Check prerequisites ---
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not found."
    exit 1
fi

# --- Install backend dependencies ---
echo "📦 Installing backend dependencies..."
cd "$SCRIPT_DIR/backend"
pip install -q fastapi uvicorn pydantic torch transformers peft accelerate 2>/dev/null || \
pip install fastapi uvicorn pydantic torch transformers peft accelerate

# --- Install frontend dependencies ---
echo "📦 Installing frontend dependencies..."
cd "$SCRIPT_DIR/frontend"
npm install --silent 2>/dev/null || npm install

# --- Start backend ---
echo ""
echo "🚀 Starting backend (FastAPI on :8000)..."
cd "$SCRIPT_DIR/backend"
GH_MODEL_SCALE="$MODEL_SCALE" python3 main.py &
BACKEND_PID=$!

# Wait for backend to become ready
echo "⏳ Waiting for model to load..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "❌ Backend failed to start within 120 seconds."
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Start frontend ---
echo "🚀 Starting frontend (Vite on :5173)..."
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "==================================="
echo "🤝 POC is running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop."
echo "==================================="

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    echo "Done."
}

trap cleanup EXIT INT TERM

# Wait for either process
wait
