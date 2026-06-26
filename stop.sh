#!/usr/bin/env bash
# stop.sh — stop DHRAS vision_service + dashboard

LOG_DIR="/home/KA_JET/robotics/jetson-vision/logs"

echo "[stop] Stopping DHRAS..."

if [ -f "$LOG_DIR/vision_service.pid" ]; then
    kill "$(cat "$LOG_DIR/vision_service.pid")" 2>/dev/null && echo "Stopped vision_service.py"
    rm -f "$LOG_DIR/vision_service.pid"
fi

if [ -f "$LOG_DIR/dashboard.pid" ]; then
    kill "$(cat "$LOG_DIR/dashboard.pid")" 2>/dev/null && echo "Stopped dashboard.py"
    rm -f "$LOG_DIR/dashboard.pid"
fi

pkill -f "vision_service.py" 2>/dev/null && echo "Stopped leftover vision_service.py" || true
pkill -f "dashboard.py" 2>/dev/null && echo "Stopped leftover dashboard.py" || true

echo "[stop] Done."
