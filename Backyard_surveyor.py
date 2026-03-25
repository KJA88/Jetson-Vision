"""
backyard_ptz_surveyor.py
Jetson Orin Nano | ~/robotics/jetson-vision/

Backyard PTZ camera — people and animals only (no vehicles).
Surveys automatically, locks on and tracks when target found.
Manual PTZ control via browser at http://192.168.1.17:5001
Taking manual control pauses auto mode; hands back after 10s of inactivity.

Usage:
    source ~/jetson_yolo_gpu/bin/activate
    cd ~/robotics/jetson-vision
    python3 backyard_ptz_surveyor.py

Control UI:
    http://192.168.1.17:5001
"""

import cv2
import time
import json
import logging
import os
import threading
from datetime import datetime
from ultralytics import YOLO
from onvif import ONVIFCamera
from flask import Flask, Response, request, jsonify
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────
# LIVE CONFIG (hot-reload from dashboard)
# ─────────────────────────────────────────────

CAM_ID      = "backyard"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras_config.json")
EVENTS_FILE = os.path.expanduser("~/robotics/jetson-vision/detections/events.jsonl")

_cfg_lock    = threading.Lock()
_cfg: dict   = {}
_events_lock = threading.Lock()


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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RTSP_URL    = "rtsp://admin:123456@192.168.1.83:554/stream1"
PTZ_IP      = "192.168.1.83"
PTZ_PORT    = 80
PTZ_USER    = "admin"
PTZ_PASS    = "123456"

MQTT_BROKER = "192.168.1.18"
MQTT_PORT   = 1883
FLASK_PORT  = 5001

YOLO_MODEL  = "yolov8s.pt"
DEVICE      = 0

FRAME_W     = 704
FRAME_H     = 576
CENTER_X    = FRAME_W // 2
CENTER_Y    = FRAME_H // 2

# ── PTZ tracking ────────────────────────────────────────
Kp_pan          = 1.0
Kp_tilt         = 1.0
MIN_SPEED       = 0.15
MAX_SPEED       = 1.0
DEAD_ZONE       = 25
PTZ_COOLDOWN    = 0.2
YOLO_EVERY      = 1

# ── Zoom ────────────────────────────────────────────────
ZOOM_SPEED        = 0.15
ZOOM_TARGET_RATIO = 0.45
ZOOM_TOLERANCE    = 0.08
ZOOM_COOLDOWN     = 1.5
ZOOM_OUT_DELAY    = 5.0

# ── Manual control ──────────────────────────────────────
MANUAL_TIMEOUT  = 10.0   # seconds of inactivity before auto resumes
MANUAL_SPEED    = 1.0    # full speed for manual button presses

# ── Survey sweep ────────────────────────────────────────
# 3-row rectangular sweep across ~240° at 15°/s.
# Each step: move 15° (1s at speed 0.5) then hold 5s.
# 16 steps per row covers ~240°. Rows go:
#   Row 1: tilt top    — pan left→right
#   Row 2: tilt middle — pan right→left
#   Row 3: tilt bottom — pan left→right
# Then tilt back to top and repeat. No snap — same speed throughout.

_M = 1.0   # move duration (seconds) per 15° step
_H = 5.0   # hold duration (seconds)
_S = 0.5   # pan speed (~15°/s)
_T = 0.06  # tilt speed between rows

def _row(direction, steps=14):
    legs = []
    for _ in range(steps):
        legs.append((direction * _S, 0.0, _M))
        legs.append((0.0, 0.0, _H))
    return legs

def _tilt(direction):
    return [(0.0, direction * _T, 0.8), (0.0, 0.0, 0.5)]

SURVEY_LEGS = (
    _tilt(1)    +   # tilt to top row
    _row(1)     +   # row 1: left → right
    _tilt(-1)   +   # tilt down to bottom
    _row(-1)    +   # row 2: right → left
    _tilt(1)        # tilt back up to top
)

# ── Detection classes — read live from cfg() at runtime ──

TOPIC_MAP = {
    "person": "backyard/person",
    "dog":    "backyard/animal",
    "cat":    "backyard/animal",
    "bird":   "backyard/animal",
    "horse":  "backyard/animal",
    "bear":   "backyard/animal",
}

COOLDOWN_SEC   = 3
CONFIRM_FRAMES = 2   # consecutive YOLO hits needed before reacting

LOG_DIR = os.path.expanduser("~/robotics/jetson-vision/detections")
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "backyard_ptz.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────

output_frame   = None
frame_lock     = threading.Lock()
last_manual    = 0.0          # timestamp of last manual PTZ command
manual_lock    = threading.Lock()

MODE_SURVEY = "SURVEY"
MODE_TRACK  = "TRACK"
MODE_SEARCH = "SEARCH"
MODE_MANUAL = "MANUAL"

current_mode = MODE_SURVEY
mode_lock    = threading.Lock()


def set_mode(m):
    global current_mode
    with mode_lock:
        current_mode = m


def get_mode():
    with mode_lock:
        return current_mode


def is_manual():
    with manual_lock:
        return (time.time() - last_manual) < MANUAL_TIMEOUT


def touch_manual():
    global last_manual
    with manual_lock:
        last_manual = time.time()

# ─────────────────────────────────────────────
# ONVIF PTZ
# ─────────────────────────────────────────────

log.info("Connecting PTZ at %s:%d ...", PTZ_IP, PTZ_PORT)
cam   = ONVIFCamera(PTZ_IP, PTZ_PORT, PTZ_USER, PTZ_PASS)
media = cam.create_media_service()
ptz   = cam.create_ptz_service()
token = media.GetProfiles()[0].token
ptz_lock = threading.Lock()
log.info("PTZ ready")


def move_camera(pan: float, tilt: float, zoom: float = 0.0):
    with ptz_lock:
        req = ptz.create_type("ContinuousMove")
        req.ProfileToken = token
        req.Velocity = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": {"x": zoom}}
        ptz.ContinuousMove(req)


def stop_camera():
    with ptz_lock:
        ptz.Stop({"ProfileToken": token})

# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────

app = Flask(__name__)


def generate_frames():
    while True:
        time.sleep(0.04)
        with frame_lock:
            if output_frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            data = buf.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"


@app.route("/config", methods=["GET", "POST"])
def config_route():
    if request.method == "GET":
        with _cfg_lock:
            return jsonify(dict(_cfg))
    update_cfg(request.get_json(force=True))
    return jsonify({"ok": True})


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backyard PTZ</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 16px;
  }
  h1 {
    font-size: 1rem;
    letter-spacing: 0.2em;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  #stream-wrap {
    position: relative;
    width: 100%;
    max-width: 704px;
  }
  #stream-wrap img {
    width: 100%;
    display: block;
    border: 1px solid #222;
  }
  #mode-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: bold;
    letter-spacing: 0.1em;
    background: #111;
    border: 1px solid #333;
    transition: color 0.3s, border-color 0.3s;
  }
  #countdown {
    position: absolute;
    top: 10px;
    left: 10px;
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 0.75rem;
    background: rgba(0,0,0,0.6);
    display: none;
  }

  /* Controls */
  #controls {
    margin-top: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    width: 100%;
    max-width: 320px;
  }
  .row {
    display: flex;
    gap: 6px;
    justify-content: center;
  }
  button {
    width: 64px;
    height: 64px;
    background: #161616;
    border: 1px solid #333;
    border-radius: 6px;
    color: #ccc;
    font-size: 1.4rem;
    cursor: pointer;
    user-select: none;
    transition: background 0.1s, border-color 0.1s;
    -webkit-tap-highlight-color: transparent;
  }
  button:active, button.held {
    background: #2a2a2a;
    border-color: #4cff90;
    color: #4cff90;
  }
  button.zoom {
    width: 80px;
    height: 48px;
    font-size: 1rem;
    letter-spacing: 0.05em;
  }
  #resume-btn {
    margin-top: 8px;
    width: 180px;
    height: 40px;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4cff90;
    border-color: #4cff90;
    display: none;
  }
  #status-row {
    margin-top: 10px;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 0.1em;
  }
</style>
</head>
<body>
<h1>Backyard Camera</h1>

<div id="stream-wrap">
  <img src="/stream" alt="live feed">
  <div id="mode-badge">–</div>
  <div id="countdown"></div>
</div>

<div id="controls">
  <div class="row">
    <button data-dir="up"    title="Tilt up">▲</button>
  </div>
  <div class="row">
    <button data-dir="left"  title="Pan left">◀</button>
    <button data-dir="stop"  title="Stop">■</button>
    <button data-dir="right" title="Pan right">▶</button>
  </div>
  <div class="row">
    <button data-dir="down"  title="Tilt down">▼</button>
  </div>
  <div class="row" style="margin-top:8px">
    <button class="zoom" data-dir="zoomin">＋ Zoom</button>
    <button class="zoom" data-dir="zoomout">－ Zoom</button>
  </div>
  <button id="resume-btn">↺ Resume Auto</button>
  <div id="status-row">AUTO MODE ACTIVE</div>
  <a href="/detections" style="margin-top:12px;font-size:0.7rem;color:#444;letter-spacing:0.1em;text-decoration:none">
    VIEW SNAPSHOTS →
  </a>
</div>

<script>
const badge     = document.getElementById('mode-badge');
const countdown = document.getElementById('countdown');
const statusRow = document.getElementById('status-row');
const resumeBtn = document.getElementById('resume-btn');

const modeColors = {
  SURVEY: '#888',
  TRACK:  '#4cff90',
  SEARCH: '#ff9800',
  MANUAL: '#4db8ff',
};

// ── Poll mode from server ─────────────────────────────
let countdownInterval = null;

async function pollMode() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const m = d.mode;
    badge.textContent   = m;
    badge.style.color         = modeColors[m] || '#ccc';
    badge.style.borderColor   = modeColors[m] || '#333';

    if (m === 'MANUAL') {
      const secs = Math.max(0, Math.ceil(d.resume_in));
      countdown.style.display = 'block';
      countdown.textContent   = `Auto resumes in ${secs}s`;
      resumeBtn.style.display = 'block';
      statusRow.textContent   = 'MANUAL CONTROL';
    } else {
      countdown.style.display = 'none';
      resumeBtn.style.display = 'none';
      statusRow.textContent   = 'AUTO MODE ACTIVE';
    }
  } catch(e) {}
}
setInterval(pollMode, 800);
pollMode();

// ── PTZ commands ─────────────────────────────────────
async function sendCmd(dir, action) {
  await fetch('/ptz', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({dir, action})
  });
}

// ── Button hold logic ─────────────────────────────────
const buttons = document.querySelectorAll('[data-dir]');
buttons.forEach(btn => {
  const dir = btn.dataset.dir;

  const start = (e) => {
    e.preventDefault();
    btn.classList.add('held');
    sendCmd(dir, 'start');
  };
  const end = (e) => {
    e.preventDefault();
    btn.classList.remove('held');
    if (dir !== 'stop') sendCmd(dir, 'stop');
  };

  btn.addEventListener('mousedown',  start);
  btn.addEventListener('touchstart', start, {passive: false});
  btn.addEventListener('mouseup',    end);
  btn.addEventListener('mouseleave', end);
  btn.addEventListener('touchend',   end);
});

resumeBtn.addEventListener('click', async () => {
  await fetch('/resume', {method: 'POST'});
});
</script>
</body>
</html>"""


@app.route("/stream")
def stream_route():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with manual_lock:
        resume_in = max(0.0, MANUAL_TIMEOUT - (time.time() - last_manual))
    return jsonify({"mode": get_mode(), "resume_in": round(resume_in, 1), "online": True, "camera": CAM_ID})


@app.route("/ptz", methods=["POST"])
def ptz_cmd():
    data   = request.get_json(force=True)
    dir_   = data.get("dir", "stop")
    action = data.get("action", "start")  # start | stop

    touch_manual()
    set_mode(MODE_MANUAL)

    if action == "stop" or dir_ == "stop":
        stop_camera()
    elif action == "start":
        pan, tilt, zoom = 0.0, 0.0, 0.0
        if dir_ == "left":    pan  = -MANUAL_SPEED
        elif dir_ == "right": pan  =  MANUAL_SPEED
        elif dir_ == "up":    tilt =  MANUAL_SPEED
        elif dir_ == "down":  tilt = -MANUAL_SPEED
        elif dir_ == "zoomin":  zoom =  ZOOM_SPEED
        elif dir_ == "zoomout": zoom = -ZOOM_SPEED
        move_camera(pan, tilt, zoom)
        log.info("Manual PTZ: dir=%s pan=%.2f tilt=%.2f zoom=%.2f", dir_, pan, tilt, zoom)

    return jsonify({"ok": True})


@app.route("/resume", methods=["POST"])
def resume():
    global last_manual
    with manual_lock:
        last_manual = 0.0   # force immediate handback
    log.info("Manual resume forced by user")
    return jsonify({"ok": True})


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
        "<h2 style='letter-spacing:0.15em;color:#666'>DETECTIONS</h2>"
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
    # Sanitise — no path traversal
    filename = os.path.basename(filename)
    path     = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(path):
        return "Not found", 404
    with open(path, "rb") as f:
        data = f.read()
    return Response(data, mimetype="image/jpeg")


threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False),
    daemon=True,
).start()

# ─────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────

def build_mqtt():
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        c = mqtt.Client()

    def on_connect(client, userdata, flags, rc, *args):
        log.info("MQTT connected" if rc == 0 else f"MQTT failed rc={rc}")

    c.on_connect = on_connect
    return c


mqtt_client = build_mqtt()


def mqtt_connect_thread():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        log.warning("MQTT connect failed: %s", e)


threading.Thread(target=mqtt_connect_thread, daemon=True).start()

last_trigger: dict = {}


def publish_event(label: str, confidence: float):
    now  = time.time()
    if now - last_trigger.get(label, 0) < COOLDOWN_SEC:
        return
    last_trigger[label] = now
    topic   = TOPIC_MAP.get(label, "backyard/detection")
    payload = json.dumps({
        "class":      label,
        "confidence": round(confidence, 3),
        "timestamp":  datetime.now().isoformat(),
        "source":     "backyard_ptz",
    })
    try:
        mqtt_client.publish(topic, payload)
        log.info("MQTT → %s | %s", topic, payload)
    except Exception as e:
        log.warning("MQTT publish failed: %s", e)

# ─────────────────────────────────────────────
# CAMERA CAPTURE THREAD
# ─────────────────────────────────────────────

latest_frame = None
cap_lock     = threading.Lock()


def capture_thread():
    global latest_frame
    while True:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning("RTSP open failed — retrying in 5s...")
            time.sleep(5)
            continue
        log.info("RTSP stream opened")
        while True:
            ret, frame = cap.read()
            if ret:
                with cap_lock:
                    latest_frame = frame
            else:
                log.warning("Frame read failed — reconnecting...")
                break
        cap.release()
        time.sleep(3)


threading.Thread(target=capture_thread, daemon=True).start()
log.info("Waiting for first frame...")
for _ in range(40):
    with cap_lock:
        if latest_frame is not None:
            break
    time.sleep(0.5)
else:
    log.error("Camera never produced a frame — check RTSP URL.")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────

log.info("Loading YOLO: %s", YOLO_MODEL)
model = YOLO(YOLO_MODEL)
model.to("cuda")
log.info("YOLO ready on GPU")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def calc_speed(offset: float, max_offset: float, Kp: float) -> float:
    if abs(offset) < DEAD_ZONE:
        return 0.0
    speed = Kp * (offset / max_offset)
    if abs(speed) < MIN_SPEED:
        speed = MIN_SPEED * (1 if speed > 0 else -1)
    return max(-MAX_SPEED, min(MAX_SPEED, speed))

# ── Survey state ────────────────────────────────────────
survey_leg_idx   = 0
survey_leg_start = time.time()


def run_survey(now: float):
    global survey_leg_idx, survey_leg_start
    pan_spd, tilt_spd, duration = SURVEY_LEGS[survey_leg_idx]
    if now - survey_leg_start >= duration:
        survey_leg_idx   = (survey_leg_idx + 1) % len(SURVEY_LEGS)
        survey_leg_start = now
        pan_spd, tilt_spd, _ = SURVEY_LEGS[survey_leg_idx]
    move_camera(pan_spd, tilt_spd, 0.0)

# ── Predictive tracking state ────────────────────────────
PREDICT_LEAD     = 0.0    # disabled
VELOCITY_SMOOTH  = 0.6

prev_target_cx   = None
prev_target_cy   = None
prev_target_time = None
vel_x            = 0.0    # smoothed pixel velocity x
vel_y            = 0.0    # smoothed pixel velocity y


def predicted_offset(cx: int, cy: int, now: float):
    """Return (offset_x, offset_y) with velocity lead applied."""
    global prev_target_cx, prev_target_cy, prev_target_time, vel_x, vel_y

    if prev_target_cx is not None and prev_target_time is not None:
        dt = now - prev_target_time
        if 0 < dt < 0.5:   # ignore stale or zero dt to avoid spikes
            raw_vx = (cx - prev_target_cx) / dt
            raw_vy = (cy - prev_target_cy) / dt
            # Clamp raw velocity to avoid huge spikes
            raw_vx = max(-500, min(500, raw_vx))
            raw_vy = max(-500, min(500, raw_vy))
            vel_x = VELOCITY_SMOOTH * vel_x + (1 - VELOCITY_SMOOTH) * raw_vx
            vel_y = VELOCITY_SMOOTH * vel_y + (1 - VELOCITY_SMOOTH) * raw_vy

    prev_target_cx   = cx
    prev_target_cy   = cy
    prev_target_time = now

    # Predicted future position
    pred_cx = cx + vel_x * PREDICT_LEAD
    pred_cy = cy + vel_y * PREDICT_LEAD

    return int(pred_cx - CENTER_X), int(pred_cy - CENTER_Y)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

last_detection  = 0.0
last_ptz_cmd    = 0.0
last_zoom_cmd   = 0.0
last_pan_dir    = 1
last_boxes      = []
frame_count     = 0
confirm_count: dict = {}

# ── Pan budget (time-based soft limits) ─────────────────
# 14 steps × 1s move = 14s total pan budget each direction.
# We track accumulated pan time and stop when budget is used.
PAN_BUDGET_SEC  = 14.0   # max seconds of pan in one direction
pan_time_used   = 0.0    # accumulated pan seconds in current direction
pan_move_start  = None   # when current pan move started

def _box_priority(b):
    return (0 if b[0] == "person" else 1, -((b[4]-b[2])*(b[5]-b[3])))


def pan_within_budget(pan: float, tilt: float, zoom: float = 0.0) -> bool:
    """Move camera only if pan budget allows. Returns True if move was sent."""
    global pan_time_used, pan_move_start, last_pan_dir
    now = time.time()

    if pan == 0.0:
        pan_move_start = None
        move_camera(0.0, tilt, zoom)
        return True

    direction = 1 if pan > 0 else -1

    # Reset budget when direction reverses
    if direction != last_pan_dir:
        pan_time_used = 0.0
        pan_move_start = None

    if pan_time_used >= PAN_BUDGET_SEC:
        stop_camera()
        pan_move_start = None
        log.debug("Pan budget exhausted — holding")
        return False

    if pan_move_start is not None:
        pan_time_used += now - pan_move_start
    pan_move_start = now

    move_camera(pan, tilt, zoom)
    return True
log.info("Survey mode active. Ctrl+C to stop.")

try:
    while True:
        with cap_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            frame = latest_frame.copy()

        frame_count += 1
        now = time.time()

        # ── If manual, skip all auto PTZ — just annotate ────
        if is_manual():
            set_mode(MODE_MANUAL)
            resume_in = max(0, MANUAL_TIMEOUT - (now - last_manual))
            cv2.putText(frame, f"MANUAL  auto in {resume_in:.0f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (77, 184, 255), 2)
            cv2.line(frame, (CENTER_X-30, CENTER_Y), (CENTER_X+30, CENTER_Y), (255,255,255), 1)
            cv2.line(frame, (CENTER_X, CENTER_Y-30), (CENTER_X, CENTER_Y+30), (255,255,255), 1)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            with frame_lock:
                output_frame = frame.copy()
            time.sleep(0.03)
            continue

        # ── Auto mode — reset survey on handback ────────────
        if get_mode() == MODE_MANUAL:
            survey_leg_start = now
            log.info("Auto mode resumed")

        # ── YOLO every N frames ──────────────────────────────
        watch  = set(cfg("watch_classes") or ["person", "dog", "cat", "bird", "horse", "bear"])
        conf_t = cfg("confidence") or CONFIDENCE
        snap   = cfg("snapshots") if cfg("snapshots") is not None else True
        mqtt_e = cfg("mqtt_enabled") if cfg("mqtt_enabled") is not None else True
        mon    = cfg("monitor_only") or False
        track  = cfg("tracking") if cfg("tracking") is not None else True

        if frame_count % YOLO_EVERY == 0:
            results     = model(frame, conf=conf_t, verbose=False, device=DEVICE)
            raw_boxes   = []
            seen_labels = set()

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label  = model.names[cls_id]
                    if label not in watch:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_boxes.append((label, float(box.conf[0]), x1, y1, x2, y2))
                    seen_labels.add(label)

            # Decay counters for unseen labels (don't hard reset on 1 missed frame)
            for label in list(confirm_count.keys()):
                if label not in seen_labels:
                    confirm_count[label] = max(0, confirm_count[label] - 1)
            for label in seen_labels:
                confirm_count[label] = confirm_count.get(label, 0) + 1

            # Only pass through boxes whose label has hit CONFIRM_FRAMES
            last_boxes = [
                b for b in raw_boxes
                if confirm_count.get(b[0], 0) >= CONFIRM_FRAMES
            ]

        boxes = last_boxes

        if boxes:
            label, conf, x1, y1, x2, y2 = sorted(boxes, key=_box_priority)[0]
            last_detection = now
            set_mode(MODE_TRACK)

            target_cx = (x1 + x2) // 2
            target_cy = y1 + (y2 - y1) // 3
            offset_x  = target_cx - CENTER_X
            offset_y  = target_cy - CENTER_Y

            if abs(offset_x) > DEAD_ZONE:
                last_pan_dir = 1 if offset_x > 0 else -1

            if (not mon) and track and now - last_ptz_cmd > PTZ_COOLDOWN:
                pan  = calc_speed(offset_x, CENTER_X, Kp_pan)
                tilt = -calc_speed(offset_y, CENTER_Y, Kp_tilt)
                # Only send zoom if pan/tilt are settled (in dead zone)
                if pan != 0 or tilt != 0:
                    pan_within_budget(pan, tilt, 0.0)
                else:
                    stop_camera()
                    if now - last_zoom_cmd > ZOOM_COOLDOWN:
                        box_ratio = (y2 - y1) / FRAME_H
                        zoom_err  = ZOOM_TARGET_RATIO - box_ratio
                        if abs(zoom_err) > ZOOM_TOLERANCE:
                            pan_within_budget(0.0, 0.0, ZOOM_SPEED if zoom_err > 0 else -ZOOM_SPEED)
                            last_zoom_cmd = now
                last_ptz_cmd = now
            elif mon or not track:
                stop_camera()

            # ── Draw annotations first so snapshot includes green box ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.circle(frame, (target_cx, target_cy), 8, (0, 200, 80), -1)
            cv2.line(frame, (CENTER_X, CENTER_Y), (target_cx, target_cy), (0, 255, 255), 1)
            cv2.putText(frame, f"{label} {conf:.0%}",
                        (x1, max(y1 - 10, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 80), 2)
            cv2.putText(frame, f"TRACKING  offset({offset_x},{offset_y})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

            # ── Snapshot + MQTT on cooldown ───────────────────
            snap_due = (now - last_trigger.get(label, 0)) >= COOLDOWN_SEC
            if snap_due:
                if mqtt_e:
                    publish_event(label, conf)
                img_path = None
                if snap:
                    snap_dir = os.path.expanduser(
                        "~/robotics/jetson-vision/" + (cfg("snapshot_dir") or "detections/backyard")
                    )
                    os.makedirs(snap_dir, exist_ok=True)
                    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"{label}_{ts}.jpg"
                    img_path = os.path.join(snap_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    log.info("SNAPSHOT  class=%-10s  conf=%.0f%%  file=%s",
                             label, conf * 100, img_name)
                log_event(label, conf, img_path)

        else:
            since_last = now - last_detection
            # Only reset predictive state after a solid gap
            if since_last > 1.0:
                prev_target_cx = prev_target_cy = prev_target_time = None
                vel_x = vel_y = 0.0

            if last_detection > 0 and since_last < 2.0:
                set_mode(MODE_SEARCH)
                if (not mon) and track and now - last_ptz_cmd > PTZ_COOLDOWN:
                    if since_last < 1.0:
                        pan_within_budget(last_pan_dir * MIN_SPEED, 0.0, 0.0)
                    else:
                        stop_camera()
                    last_ptz_cmd = now
                cv2.putText(frame, f"SEARCHING  {2.0 - since_last:.0f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                if get_mode() != MODE_SURVEY:
                    log.info("Target lost — resuming survey")
                    survey_leg_start = now
                    pan_time_used    = 0.0
                    pan_move_start   = None
                set_mode(MODE_SURVEY)
                if (not mon) and track:
                    run_survey(now)
                cv2.putText(frame, "SURVEYING" if (not mon and track) else "MONITOR",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)

            if (not mon) and track and last_detection > 0 and since_last > ZOOM_OUT_DELAY:
                if now - last_zoom_cmd > ZOOM_COOLDOWN:
                    move_camera(0.0, 0.0, -ZOOM_SPEED)
                    last_zoom_cmd = now

        # ── Crosshair + timestamp ────────────────────────────
        cv2.line(frame, (CENTER_X-30, CENTER_Y), (CENTER_X+30, CENTER_Y), (255,255,255), 1)
        cv2.line(frame, (CENTER_X, CENTER_Y-30), (CENTER_X, CENTER_Y+30), (255,255,255), 1)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                    (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        with frame_lock:
            output_frame = frame.copy()

        time.sleep(0.03)

except KeyboardInterrupt:
    log.info("Stopped by user.")
finally:
    stop_camera()
    mqtt_client.loop_stop()
    log.info("Shutdown complete.")
