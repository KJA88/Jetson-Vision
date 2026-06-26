#!/usr/bin/env bash
# start.sh — launch vision_service + dashboard
# vision_service.py handles all cameras + MJPEG streams on port 8081
# dashboard.py serves the UI on port 8080

set -e

VENV=~/jetson_yolo_gpu/bin/activate
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

source "$VENV"

echo "[start] Launching vision_service.py (all cameras, YOLO GPU, streams → :8081)..."
python3 "$DIR/vision_service.py" >> "$LOG_DIR/vision_service.log" 2>&1 &
VSVC_PID=$!
echo "[start] vision_service PID=$VSVC_PID"

echo "[start] Waiting for streams to come up..."
sleep 6

echo "[start] Launching dashboard.py (port 8080)..."
python3 "$DIR/dashboard.py" >> "$LOG_DIR/dashboard.log" 2>&1 &
DASH_PID=$!
echo "[start] dashboard PID=$DASH_PID"

echo ""
echo "[start] All services started."
echo "  Frontyard stream  → http://192.168.1.17:8081/stream/frontyard"
echo "  Backyard stream   → http://192.168.1.17:8081/stream/backyard"
echo "  Indoor stream     → http://192.168.1.17:8081/stream/indoor"
echo "  Health check      → http://192.168.1.17:8081/health"
echo "  Dashboard         → http://192.168.1.17:8080"
echo "Logs in $LOG_DIR"
