"""
Frontyard_Logging.py
Jetson Orin Nano | ~/robotics/jetson-vision/

Moving vehicles — green box, trigger, snapshot, MQTT.
Parked vehicles — completely invisible, no box, no trigger, no snapshot.
People and animals — always green box, trigger, snapshot, MQTT.

Usage:
    source ~/jetson_yolo_gpu/bin/activate
    cd ~/robotics/jetson-vision
    python3 Frontyard_Logging.py

Debug stream:
    http://192.168.1.17:5000

Detections gallery:
    http://192.168.1.17:5000/detections
"""

import cv2
import time
import json
import logging
import os
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RTSP_URL         = "rtsp://admin:Mastercam101@192.168.1.82:554/Streaming/Channels/102"
MQTT_BROKER      = "192.168.1.18"
MQTT_PORT        = 1883
MQTT_TOPIC       = "outdoor/detection"
FLASK_PORT       = 5000
YOLO_MODEL       = "yolov8n.pt"
COOLDOWN_SEC     = 2
DEVICE           = "cpu"
MOTION_THRESHOLD = 50  # pixels a vehicle must move between frames to be considered moving

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

TOPIC_MAP = {
    "person":     "outdoor/person",
    "car":        "outdoor/vehicle",
    "truck":      "outdoor/vehicle",
    "bus":        "outdoor/vehicle",
    "motorcycle": "outdoor/vehicle",
    "bicycle":    "outdoor/vehicle",
    "dog":        "outdoor/animal",
    "cat":        "outdoor/animal",
    "bird":       "outdoor/animal",
    "horse":      "outdoor/animal",
    "bear":       "outdoor/animal",
}

# ─────────────────────────────────────────────
# LIVE CONFIG (hot-reload from dashboard)
# ─────────────────────────────────────────────

CAM_ID      = "frontyard"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras_config.json")
EVENTS_FILE = os.path.expanduser("~/robotics/jetson-vision/detections/events.jsonl")

_cfg_lock = Lock()
_cfg: dict = {}
_events_lock = Lock()


def _load_cam_config():
    global _cfg
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        cam = data["cameras"][CAM_ID]
        with _cfg_lock:
            _cfg = dict(cam)
    except Exception as exc:
        print(f"[config] load failed: {exc}")


def cfg(key, default=None):
    with _cfg_lock:
        return _cfg.get(key, default)


def update_cfg(data: dict):
    with _cfg_lock:
        _cfg.update(data)
    try:
        with open(CONFIG_FILE) as f:
            full = json.load(f)
        full["cameras"][CAM_ID].update(data)
        with open(CONFIG_FILE, "w") as f:
            json.dump(full, f, indent=2)
    except Exception as exc:
        print(f"[config] save failed: {exc}")


def log_event(label: str, confidence: float, image_path: str = None):
    event = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "camera":     CAM_ID,
        "class":      label,
        "confidence": round(confidence, 3),
        "image":      os.path.basename(image_path) if image_path else None,
    }
    with _events_lock:
        os.makedirs(os.path.dirname(EVENTS_FILE), exist_ok=True)
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")


_load_cam_config()

# Derive runtime constants from config (fallback defaults if config missing)
LOG_DIR = os.path.expanduser(
    "~/robotics/jetson-vision/" + (cfg("snapshot_dir") or "detections/frontyard")
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "detections.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

app           = Flask(__name__)
frame_lock    = Lock()
current_frame = None


# ─────────────────────────────────────────────
# CAMERA STREAM THREAD
# ─────────────────────────────────────────────

class CameraStream:
    def __init__(self, url):
        self.url   = url
        self.frame = None
        self.ok    = False
        self.lock  = Lock()
        self._stop = False
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("Camera stream could not open — retrying in 5s...")
                time.sleep(5)
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("Stream opened — %dx%d", w, h)
            while not self._stop:
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.ok    = True
                else:
                    log.warning("Frame read failed — reconnecting...")
                    with self.lock:
                        self.ok = False
                    break
            cap.release()
            if not self._stop:
                time.sleep(3)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def release(self):
        self._stop = True


# ─────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────

def build_mqtt_client():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    def on_connect(c, userdata, flags, rc, *args):
        if rc == 0:
            log.info("MQTT connected to broker at %s", MQTT_BROKER)
        else:
            log.warning("MQTT connection failed — rc=%d", rc)

    def on_disconnect(c, userdata, *args):
        log.warning("MQTT disconnected")

    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    return client


mqtt_client = build_mqtt_client()

def mqtt_connect():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        log.warning("MQTT connect failed: %s — detections will still log locally", e)

Thread(target=mqtt_connect, daemon=True).start()


# ─────────────────────────────────────────────
# DETECTION STATE
# ─────────────────────────────────────────────

last_trigger:   dict = {}
last_positions: dict = {}

def should_trigger(label: str) -> bool:
    now  = time.time()
    last = last_trigger.get(label, 0)
    if now - last >= COOLDOWN_SEC:
        last_trigger[label] = now
        return True
    return False

def is_moving(label: str, x1: int, y1: int, x2: int, y2: int) -> bool:
    cx     = (x1 + x2) // 2
    cy     = (y1 + y2) // 2
    region = (cx // 80, cy // 80)
    key    = (label, region)
    prev   = last_positions.get(key)
    last_positions[key] = (cx, cy)
    if prev is None:
        return False
    dx = abs(cx - prev[0])
    dy = abs(cy - prev[1])
    return (dx >= MOTION_THRESHOLD or dy >= MOTION_THRESHOLD)

def publish_event(label: str, confidence: float, image_path: str):
    topic   = TOPIC_MAP.get(label, MQTT_TOPIC)
    payload = json.dumps({
        "class":      label,
        "confidence": round(float(confidence), 3),
        "timestamp":  datetime.now().isoformat(),
        "image":      os.path.basename(image_path),
    })
    try:
        mqtt_client.publish(topic, payload)
        log.info("MQTT → %s | %s", topic, payload)
    except Exception as e:
        log.warning("MQTT publish failed: %s", e)


# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            frame = current_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(0.04)


@app.route("/config", methods=["GET", "POST"])
def config_route():
    if request.method == "GET":
        with _cfg_lock:
            return jsonify(dict(_cfg))
    update_cfg(request.get_json(force=True))
    return jsonify({"ok": True})


@app.route("/status")
def status():
    return jsonify({"camera": CAM_ID, "mode": "MONITOR" if cfg("monitor_only") else "ACTIVE", "online": True})


@app.route("/")
def index():
    return (
        "<html><body style='background:#111;margin:0;font-family:monospace'>"
        "<img src='/stream' style='width:100%;max-width:1280px;display:block;margin:auto'>"
        "<div style='text-align:center;margin-top:10px'>"
        "<a href='/detections' style='color:#666;font-size:0.8rem;letter-spacing:0.1em;text-decoration:none'>"
        "VIEW SNAPSHOTS →</a></div>"
        "</body></html>"
    )


@app.route("/stream")
def stream():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detections")
def detections_gallery():
    files = sorted(
        [f for f in os.listdir(LOG_DIR) if f.endswith(".jpg")],
        reverse=True
    )[:30]
    items = "".join(
        f"<div style='margin-bottom:20px'>"
        f"<p style='color:#666;font-size:0.75rem;margin-bottom:6px'>{f}</p>"
        f"<img src='/snapshot/{f}' style='max-width:100%;border:1px solid #222'>"
        f"</div>"
        for f in files
    )
    count = len(files)
    return (
        "<html><body style='background:#0a0a0a;color:#eee;"
        "font-family:Courier New,monospace;padding:16px;max-width:720px;margin:auto'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:16px'>"
        "<a href='/' style='color:#444;font-size:0.75rem;letter-spacing:0.1em;text-decoration:none'>← LIVE VIEW</a>"
        "<h2 style='letter-spacing:0.15em;color:#666'>FRONTYARD DETECTIONS</h2>"
        f"<span style='color:#444;font-size:0.75rem'>{count} snapshots</span>"
        "<button onclick=\"if(confirm('Delete all snapshots?')) fetch('/detections/clear',{method:'POST'}).then(()=>location.reload())\" "
        "style='background:#1a0000;border:1px solid #550000;color:#ff4444;padding:6px 14px;"
        "font-family:Courier New,monospace;font-size:0.75rem;letter-spacing:0.1em;cursor:pointer'>"
        "CLEAR ALL</button>"
        "</div>"
        f"{items or '<p style=color:#444>No snapshots yet.</p>'}"
        "</body></html>"
    )


@app.route("/detections/clear", methods=["POST"])
def clear_detections():
    removed = 0
    for f in os.listdir(LOG_DIR):
        if f.endswith(".jpg"):
            os.remove(os.path.join(LOG_DIR, f))
            removed += 1
    log.info("Cleared %d snapshots", removed)
    return jsonify({"ok": True, "removed": removed})


@app.route("/snapshot/<filename>")
def serve_snapshot(filename):
    filename = os.path.basename(filename)
    path     = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(path):
        return "Not found", 404
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

Thread(target=run_flask, daemon=True).start()


# ─────────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────────

def main():
    global current_frame

    log.info("Loading YOLO model: %s on GPU device %s", YOLO_MODEL, DEVICE)
    model = YOLO(YOLO_MODEL)
    model.to('cpu')
    log.info("Model loaded on GPU. Watching: %s", ", ".join(sorted(_cfg.get("watch_classes", []))))

    log.info("Opening RTSP stream: %s", RTSP_URL)
    cam = CameraStream(RTSP_URL)

    log.info("Waiting for camera...")
    for _ in range(30):
        ok, frame = cam.read()
        if ok and frame is not None:
            break
        time.sleep(0.5)
    else:
        log.error("Camera never produced a frame — check RTSP URL and credentials.")
        return

    frame_h = frame.shape[0]

    log.info("Debug stream:      http://192.168.1.17:%d", FLASK_PORT)
    log.info("Detections gallery: http://192.168.1.17:%d/detections", FLASK_PORT)
    log.info("Press Ctrl+C to stop.")

    fps_count = 0
    fps_timer = time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            fps_count += 1
            if fps_count >= 30:
                elapsed = time.time() - fps_timer
                log.debug("Inference FPS: %.1f", fps_count / elapsed)
                fps_count = 0
                fps_timer = time.time()

            watch  = set(cfg("watch_classes") or [])
            conf_t = cfg("confidence") or 0.25
            snap   = cfg("snapshots") if cfg("snapshots") is not None else True
            mqtt_e = cfg("mqtt_enabled") if cfg("mqtt_enabled") is not None else True
            mon    = cfg("monitor_only") or False

            results = model(frame, conf=conf_t, verbose=False, device=DEVICE)

            annotated            = frame.copy()
            triggered_this_frame = set()

            for r in results:
                for box in r.boxes:
                    cls_id     = int(box.cls[0])
                    label      = model.names[cls_id]
                    confidence = float(box.conf[0])

                    if label not in watch:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label in VEHICLE_CLASSES:
                        if not is_moving(label, x1, y1, x2, y2):
                            log.debug("PARKED  class=%-12s  conf=%.0f%%  (not moving)", label, confidence * 100)
                            continue

                    color = (0, 200, 80)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} {confidence:.0%}",
                                (x1, max(y1 - 8, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if not mon and label not in triggered_this_frame and should_trigger(label):
                        triggered_this_frame.add(label)

                        img_path = None
                        if snap:
                            snap_dir = os.path.expanduser(
                                "~/robotics/jetson-vision/" + (cfg("snapshot_dir") or "detections/frontyard")
                            )
                            os.makedirs(snap_dir, exist_ok=True)
                            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_name = f"{label}_{ts}.jpg"
                            img_path = os.path.join(snap_dir, img_name)
                            cv2.imwrite(img_path, annotated)

                        log.info(
                            "DETECTED  class=%-12s  conf=%.0f%%  image=%s",
                            label, confidence * 100, os.path.basename(img_path) if img_path else "–"
                        )

                        log_event(label, confidence, img_path)

                        if mqtt_e:
                            publish_event(label, confidence, img_path or "")

            ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            cv2.putText(annotated, ts_str, (10, frame_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            with frame_lock:
                current_frame = annotated

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        cam.release()
        mqtt_client.loop_stop()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
