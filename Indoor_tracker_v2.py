import cv2
import time
import json
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

CAM_ID      = "indoor"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras_config.json")
EVENTS_FILE = os.path.expanduser("~/robotics/jetson-vision/detections/events.jsonl")
MQTT_BROKER = "192.168.1.18"
MQTT_PORT   = 1883
COOLDOWN_SEC = 3

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

# ── MQTT ──────────────────────────────────────
_last_trigger: dict = {}
_trigger_lock = threading.Lock()


def _build_mqtt():
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        c = mqtt.Client()
    c.on_connect = lambda client, ud, flags, rc, *a: \
        print("MQTT connected" if rc == 0 else f"MQTT failed rc={rc}")
    return c


_mqtt_client = _build_mqtt()


def _mqtt_connect():
    try:
        _mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        _mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT connect failed: {e}")


threading.Thread(target=_mqtt_connect, daemon=True).start()


def publish_event(label: str, confidence: float):
    now = time.time()
    with _trigger_lock:
        if now - _last_trigger.get(label, 0) < COOLDOWN_SEC:
            return
        _last_trigger[label] = now
    payload = json.dumps({
        "class":      label,
        "confidence": round(confidence, 3),
        "timestamp":  datetime.now().isoformat(),
        "source":     CAM_ID,
    })
    try:
        _mqtt_client.publish(f"indoor/{label}", payload)
    except Exception as e:
        print(f"MQTT publish failed: {e}")

# ── Flask ──────────────────────────────────────────────
app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()

@app.route('/config', methods=['GET', 'POST'])
def config_route():
    if request.method == 'GET':
        with _cfg_lock:
            return jsonify(dict(_cfg))
    update_cfg(request.get_json(force=True))
    return jsonify({'ok': True})


@app.route('/status')
def status():
    return jsonify({'camera': CAM_ID, 'mode': 'TRACK' if len(last_boxes) > 0 else 'IDLE', 'online': True})


@app.route('/stream')
def stream():
    def generate():
        while True:
            time.sleep(0.05)
            with cap_lock:
                if latest_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5002, threaded=True), daemon=True).start()

# ── Settings ───────────────────────────────────────────
RTSP_URL  = 'rtsp://admin:123456@192.168.1.116:554/stream1'
PTZ_IP    = '192.168.1.116'
PTZ_PORT  = 80
PTZ_USER  = 'admin'
PTZ_PASS  = '123456'

FRAME_W   = 704
FRAME_H   = 576
CENTER_X  = FRAME_W // 2
CENTER_Y  = FRAME_H // 2

# Proportional control gains — lower = smoother, less oscillation
Kp_pan    = 0.3
Kp_tilt   = 0.3

# Speed limits
MIN_SPEED  = 0.04
MAX_SPEED  = 0.35
DEAD_ZONE  = 60    # pixels — bigger = settles easier

# Timing
PTZ_COOLDOWN = 0.4    # seconds between PTZ commands
YOLO_EVERY   = 3      # run YOLO every N frames

# Zoom based on subject size in frame
ZOOM_SPEED        = 0.15
ZOOM_TARGET_RATIO = 0.45   # ideal box height = 45% of frame
ZOOM_TOLERANCE    = 0.08   # don't zoom if within 8% of target
ZOOM_COOLDOWN     = 1.5    # seconds between zoom commands
ZOOM_OUT_DELAY    = 5.0    # seconds before zooming out after losing target

# ── ONVIF ──────────────────────────────────────────────
print('Connecting PTZ...')
cam     = ONVIFCamera(PTZ_IP, PTZ_PORT, PTZ_USER, PTZ_PASS)
media   = cam.create_media_service()
ptz     = cam.create_ptz_service()
token   = media.GetProfiles()[0].token
print('PTZ ready')

def move_camera(pan, tilt, zoom=0):
    req = ptz.create_type('ContinuousMove')
    req.ProfileToken = token
    req.Velocity = {'PanTilt': {'x': pan, 'y': tilt}, 'Zoom': {'x': zoom}}
    ptz.ContinuousMove(req)

def stop_camera():
    ptz.Stop({'ProfileToken': token})

# ── YOLO ───────────────────────────────────────────────
print('Loading YOLO...')
model = YOLO('yolov8n.pt')
model.to('cpu')
print('YOLO ready')

# ── Capture thread ─────────────────────────────────────
latest_frame = None
cap_lock = threading.Lock()

def capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if ret:
            with cap_lock:
                latest_frame = frame

threading.Thread(target=capture_thread, daemon=True).start()
time.sleep(2)

# ── Proportional speed calculator ──────────────────────
def calc_speed(offset, max_offset, Kp):
    if abs(offset) < DEAD_ZONE:
        return 0.0
    normalized = offset / max_offset
    speed = Kp * normalized
    if abs(speed) < MIN_SPEED:
        speed = MIN_SPEED * (1 if speed > 0 else -1)
    speed = max(-MAX_SPEED, min(MAX_SPEED, speed))
    return speed

# ── State ──────────────────────────────────────────────
last_pan_dir   = 0
last_tilt_dir  = 0
last_ptz_cmd   = 0
last_zoom_cmd  = 0
last_detection = 0
zoomed_in      = False
last_boxes     = []
frame_count    = 0

# ── Main loop ──────────────────────────────────────────
print('Streaming at http://192.168.1.17:5002/stream')
print('Ctrl+C to quit')

while True:
    with cap_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame_count += 1
    now = time.time()

    # ── Read live config ────────────────────────────────
    watch  = set(cfg("watch_classes") or ["person"])
    conf_t = cfg("confidence") or 0.50
    snap   = cfg("snapshots") if cfg("snapshots") is not None else False
    mqtt_e = cfg("mqtt_enabled") if cfg("mqtt_enabled") is not None else False
    mon    = cfg("monitor_only") or False
    track  = cfg("tracking") if cfg("tracking") is not None else True

    # Map class names to YOLO class IDs for filtering
    class_ids = [i for i, n in model.names.items() if n in watch]

    # ── YOLO every N frames ────────────────────────────
    if frame_count % YOLO_EVERY == 0:
        results = model(frame, classes=class_ids if class_ids else None,
                        conf=conf_t, verbose=False)[0]
        last_boxes = results.boxes

    boxes = last_boxes

    if len(boxes) > 0:
        last_detection = now

        # Largest detected box
        largest = max(boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest.xyxy[0])
        det_label = model.names[int(largest.cls[0])]

        # Aim at upper third of box (chest/head area)
        person_cx = (x1 + x2) // 2
        person_cy = y1 + (y2 - y1) // 3

        offset_x = person_cx - CENTER_X
        offset_y = person_cy - CENTER_Y
        conf      = int(largest.conf[0] * 100)

        # Save direction for predictive search
        if abs(offset_x) > DEAD_ZONE:
            last_pan_dir  = 1 if offset_x > 0 else -1
        if abs(offset_y) > DEAD_ZONE:
            last_tilt_dir = 1 if offset_y > 0 else -1

        # ── Pan/Tilt command ───────────────────────────
        if (not mon) and track and now - last_ptz_cmd > PTZ_COOLDOWN:
            pan  = calc_speed(offset_x, CENTER_X, Kp_pan)
            tilt = calc_speed(offset_y, CENTER_Y, Kp_tilt)
            tilt = -tilt  # invert — pixel Y increases downward

            if pan != 0 or tilt != 0:
                move_camera(pan, tilt, 0)
            else:
                stop_camera()

            last_ptz_cmd = now
            print(f'P:{pan:.2f} T:{tilt:.2f} Offset:({offset_x},{offset_y})')
        elif mon or not track:
            stop_camera()

        # ── Zoom based on subject size ─────────────────
        if (not mon) and track and now - last_zoom_cmd > ZOOM_COOLDOWN:
            box_height_ratio = (y2 - y1) / FRAME_H
            zoom_error = ZOOM_TARGET_RATIO - box_height_ratio

            if abs(zoom_error) > ZOOM_TOLERANCE:
                zoom_cmd = ZOOM_SPEED if zoom_error > 0 else -ZOOM_SPEED
                move_camera(0, 0, zoom_cmd)
                last_zoom_cmd = now
                print(f'Zoom: box_ratio={box_height_ratio:.2f} target={ZOOM_TARGET_RATIO} cmd={zoom_cmd:.2f}')

        # ── Snapshot + MQTT + event log ────────────────
        with _trigger_lock:
            since_trigger = now - _last_trigger.get(det_label, 0)
        if since_trigger >= COOLDOWN_SEC:
            img_path = None
            if snap:
                snap_dir = os.path.expanduser(
                    "~/robotics/jetson-vision/" + (cfg("snapshot_dir") or "detections/indoor")
                )
                os.makedirs(snap_dir, exist_ok=True)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = f"{det_label}_{ts}.jpg"
                img_path = os.path.join(snap_dir, img_name)
                cv2.imwrite(img_path, frame)
            if mqtt_e:
                publish_event(det_label, conf / 100.0)
            log_event(det_label, conf / 100.0, img_path)

        # ── Draw annotations ──────────────────────────
        box_h_ratio = (y2 - y1) / FRAME_H
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (person_cx, person_cy), 8, (0, 255, 0), -1)
        cv2.line(frame, (CENTER_X, CENTER_Y), (person_cx, person_cy), (0, 255, 255), 1)
        mode_label = "MONITOR" if (mon or not track) else "TRACKING"
        cv2.putText(frame, f'{det_label} {conf}% [{mode_label}]', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Offset X:{offset_x} Y:{offset_y}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'BoxRatio:{box_h_ratio:.2f} Target:{ZOOM_TARGET_RATIO}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    else:
        # ── No detection ──────────────────────────────

        # Predictive search for 2 seconds in last known direction
        if (not mon) and track and now - last_detection < 2.0 and last_detection > 0:
            if now - last_ptz_cmd > PTZ_COOLDOWN:
                move_camera(last_pan_dir * MIN_SPEED * 2, 0, 0)
                last_ptz_cmd = now
            cv2.putText(frame, 'Searching...', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            stop_camera()
            cv2.putText(frame, 'No Detection', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Zoom out after delay
        if (not mon) and track and now - last_detection > ZOOM_OUT_DELAY and last_detection > 0:
            if now - last_zoom_cmd > ZOOM_COOLDOWN:
                move_camera(0, 0, -ZOOM_SPEED)
                last_zoom_cmd = now

    # ── Center crosshair ──────────────────────────────
    cv2.line(frame, (CENTER_X-30, CENTER_Y), (CENTER_X+30, CENTER_Y), (255, 255, 255), 2)
    cv2.line(frame, (CENTER_X, CENTER_Y-30), (CENTER_X, CENTER_Y+30), (255, 255, 255), 2)
    cv2.circle(frame, (CENTER_X, CENTER_Y), 5, (255, 255, 255), 1)

    # ── Push to Flask ─────────────────────────────────
    with frame_lock:
        output_frame = frame.copy()

    time.sleep(0.03)