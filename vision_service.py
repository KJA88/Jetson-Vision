"""
vision_service.py
Jetson Orin Nano | ~/robotics/jetson-vision/

Single vision service: one YOLOv8n on GPU, all three cameras.
- One CameraWorker per camera (RTSP capture thread)
- Central inference loop processes frames from all cameras sequentially
- PTZ auto-tracking for backyard + indoor (reads tracking/monitor_only from config)
- MJPEG streams served directly on port 8081 at /stream/<cam_id>
- Config hot-reloads from cameras_config.json every 30 frames

Usage:
    source ~/jetson_yolo_gpu/bin/activate
    python3 ~/robotics/jetson-vision/vision_service.py

Streams:
    http://192.168.1.17:8081/stream/frontyard
    http://192.168.1.17:8081/stream/backyard
    http://192.168.1.17:8081/stream/indoor
"""

import cv2
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

from flask import Flask, Response, jsonify
from flask_cors import CORS
from onvif import ONVIFCamera
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "cameras_config.json")
EVENTS_FILE = os.path.expanduser("~/robotics/jetson-vision/detections/events.jsonl")
YOLO_MODEL  = os.path.join(BASE_DIR, "yolov8n.pt")

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────

_config_lock = threading.Lock()
_config: dict = {}


def _load_full_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def reload_config():
    global _config
    try:
        data = _load_full_config()
        with _config_lock:
            _config = data
    except Exception as e:
        logging.warning("[config] reload failed: %s", e)


def cam_cfg(cam_id: str, key: str, default=None):
    with _config_lock:
        return _config.get("cameras", {}).get(cam_id, {}).get(key, default)


reload_config()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

os.makedirs(os.path.expanduser("~/robotics/jetson-vision/logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.expanduser("~/robotics/jetson-vision/logs/vision_service.log")
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# EVENTS LOG
# ─────────────────────────────────────────────

_events_lock = threading.Lock()


def log_event(cam_id: str, label: str, confidence: float, image_path: str = None):
    event = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "camera":     cam_id,
        "class":      label,
        "confidence": round(confidence, 3),
        "image":      os.path.basename(image_path) if image_path else None,
    }
    with _events_lock:
        os.makedirs(os.path.dirname(EVENTS_FILE), exist_ok=True)
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")


# ─────────────────────────────────────────────
# MQTT
# ─────────────────────────────────────────────

MQTT_BROKER  = "192.168.1.18"
MQTT_PORT    = 1883
COOLDOWN_SEC = 3

TOPIC_MAP = {
    "frontyard": {
        "person":     "outdoor/person",
        "car":        "outdoor/vehicle",
        "truck":      "outdoor/vehicle",
        "bus":        "outdoor/vehicle",
        "motorcycle": "outdoor/vehicle",
        "bicycle":    "outdoor/vehicle",
        "dog":        "outdoor/animal",
        "cat":        "outdoor/animal",
        "bird":       "outdoor/animal",
    },
    "backyard": {
        "person": "backyard/person",
        "dog":    "backyard/animal",
        "cat":    "backyard/animal",
        "bird":   "backyard/animal",
        "horse":  "backyard/animal",
        "bear":   "backyard/animal",
    },
    "indoor": {
        "person": "indoor/person",
    },
}


def _build_mqtt():
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        c = mqtt.Client()
    c.on_connect = lambda cl, ud, flags, rc, *a: \
        log.info("MQTT connected" if rc == 0 else "MQTT failed rc=%s" % rc)
    return c


_mqtt = _build_mqtt()


def _mqtt_connect():
    try:
        _mqtt.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        _mqtt.loop_start()
    except Exception as e:
        log.warning("MQTT connect failed: %s", e)


threading.Thread(target=_mqtt_connect, daemon=True).start()

# ─────────────────────────────────────────────
# CAMERA WORKER  (RTSP capture thread)
# ─────────────────────────────────────────────

class CameraWorker:
    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id   = cam_id
        self.rtsp_url = rtsp_url
        self._frame   = None
        self._lock    = threading.Lock()
        self._stop    = False
        threading.Thread(target=self._reader, daemon=True,
                         name="cap-%s" % cam_id).start()

    def _reader(self):
        while not self._stop:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("[%s] RTSP open failed — retrying in 5s", self.cam_id)
                time.sleep(5)
                continue
            log.info("[%s] RTSP opened", self.cam_id)
            while not self._stop:
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                else:
                    log.warning("[%s] frame read failed — reconnecting", self.cam_id)
                    break
            cap.release()
            if not self._stop:
                time.sleep(3)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop = True


# ─────────────────────────────────────────────
# VEHICLE MOTION TRACKER  (frontyard only)
# ─────────────────────────────────────────────

VEHICLE_CLASSES        = {"car", "truck", "bus", "motorcycle", "bicycle"}
MOTION_THRESHOLD       = 25
MOTION_FRAMES_REQUIRED = 3
TRACK_STALE_FRAMES     = 60


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


class VehicleTracker:
    def __init__(self):
        self._tracks  = {}
        self._counter = 0

    def is_moving(self, label: str, x1: int, y1: int, x2: int, y2: int) -> bool:
        box = (x1, y1, x2, y2)
        cx  = (x1 + x2) // 2
        cy  = (y1 + y2) // 2

        best_id  = None
        best_iou = 0.0
        for tid, t in self._tracks.items():
            if t["label"] != label:
                continue
            score = _iou(box, t["box"])
            if score > best_iou:
                best_iou = score
                best_id  = tid

        if best_id is None or best_iou < 0.25:
            for tid, t in self._tracks.items():
                if t["label"] != label:
                    continue
                pcx, pcy = t["center"]
                if abs(cx - pcx) < 40 and abs(cy - pcy) < 40:
                    best_id = tid
                    break

        if best_id is None:
            self._counter += 1
            self._tracks[self._counter] = {
                "label":        label,
                "box":          box,
                "center":       (cx, cy),
                "motion_count": 0,
                "missed":       0,
            }
            return False

        t = self._tracks[best_id]
        prev_cx, prev_cy = t["center"]
        dx = abs(cx - prev_cx)
        dy = abs(cy - prev_cy)

        if dx >= MOTION_THRESHOLD or dy >= MOTION_THRESHOLD:
            t["motion_count"] += 1
            t["center"] = (cx, cy)
            t["box"]    = box
        else:
            t["motion_count"] = 0

        t["missed"] = 0

        stale = [tid for tid, tr in self._tracks.items()
                 if tid != best_id and tr["missed"] > TRACK_STALE_FRAMES]
        for tid in stale:
            del self._tracks[tid]
        for tid, tr in self._tracks.items():
            if tid != best_id:
                tr["missed"] += 1

        return t["motion_count"] >= MOTION_FRAMES_REQUIRED


# ─────────────────────────────────────────────
# PTZ CONTROLLER  (per PTZ camera)
# ─────────────────────────────────────────────

# Tuning params preserved from the original per-camera scripts
PTZ_PARAMS = {
    "backyard": {
        "Kp_pan":            1.0,
        "Kp_tilt":           1.0,
        "MIN_SPEED":         0.15,
        "MAX_SPEED":         1.0,
        "DEAD_ZONE":         25,
        "PTZ_COOLDOWN":      0.2,
        "ZOOM_SPEED":        0.15,
        "ZOOM_TARGET_RATIO": 0.45,
        "ZOOM_TOLERANCE":    0.08,
        "ZOOM_COOLDOWN":     1.5,
        "ZOOM_OUT_DELAY":    5.0,
        "CONFIRM_FRAMES":    2,
    },
    "indoor": {
        "Kp_pan":            0.3,
        "Kp_tilt":           0.3,
        "MIN_SPEED":         0.04,
        "MAX_SPEED":         0.35,
        "DEAD_ZONE":         60,
        "PTZ_COOLDOWN":      0.4,
        "ZOOM_SPEED":        0.15,
        "ZOOM_TARGET_RATIO": 0.45,
        "ZOOM_TOLERANCE":    0.08,
        "ZOOM_COOLDOWN":     1.5,
        "ZOOM_OUT_DELAY":    5.0,
        "CONFIRM_FRAMES":    2,
    },
}


class PTZController:
    def __init__(self, cam_id: str, host: str, port: int, user: str, password: str):
        self.cam_id = cam_id
        self.p      = PTZ_PARAMS[cam_id]
        log.info("[%s] Connecting PTZ at %s:%d ...", cam_id, host, port)
        onvif_cam   = ONVIFCamera(host, port, user, password)
        media_svc   = onvif_cam.create_media_service()
        ptz_svc     = onvif_cam.create_ptz_service()
        token       = media_svc.GetProfiles()[0].token
        self._ptz   = ptz_svc
        self._token = token
        self._lock  = threading.Lock()
        log.info("[%s] PTZ ready", cam_id)

        # Per-camera tracking state
        self.last_detection = 0.0
        self.last_ptz_cmd   = 0.0
        self.last_zoom_cmd  = 0.0
        self.confirm_count: dict = {}

    def move(self, pan: float, tilt: float, zoom: float = 0.0):
        with self._lock:
            req = self._ptz.create_type("ContinuousMove")
            req.ProfileToken = self._token
            req.Velocity = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": {"x": zoom}}
            self._ptz.ContinuousMove(req)

    def stop(self):
        with self._lock:
            self._ptz.Stop({"ProfileToken": self._token})

    def calc_speed(self, offset: float, max_offset: float, Kp: float) -> float:
        p = self.p
        if abs(offset) < p["DEAD_ZONE"]:
            return 0.0
        speed = Kp * (offset / max_offset)
        if abs(speed) < p["MIN_SPEED"]:
            speed = p["MIN_SPEED"] * (1 if speed > 0 else -1)
        return max(-p["MAX_SPEED"], min(p["MAX_SPEED"], speed))


# ─────────────────────────────────────────────
# CAMERA PROCESSOR  (per-camera state + output frame)
# ─────────────────────────────────────────────

class CameraProcessor:
    def __init__(self, cam_id: str):
        self.cam_id       = cam_id
        self._out_frame   = None
        self._out_lock    = threading.Lock()
        self.last_trigger: dict = {}
        self.vtracker     = VehicleTracker() if cam_id == "frontyard" else None
        self.ptz: Optional[PTZController] = None

    def set_ptz(self, ptz: PTZController):
        self.ptz = ptz

    def put_frame(self, frame):
        with self._out_lock:
            self._out_frame = frame

    def get_frame(self):
        with self._out_lock:
            return self._out_frame

    def cooldown_ok(self, label: str) -> bool:
        now = time.time()
        if now - self.last_trigger.get(label, 0) >= COOLDOWN_SEC:
            self.last_trigger[label] = now
            return True
        return False


# ─────────────────────────────────────────────
# SNAPSHOT + MQTT + EVENT HELPER
# ─────────────────────────────────────────────

def _trigger_action(cam_id: str, label: str, conf: float, frame, snap: bool, mqtt_e: bool):
    img_path = None
    if snap:
        snap_dir = os.path.expanduser(
            "~/robotics/jetson-vision/" +
            (cam_cfg(cam_id, "snapshot_dir") or "detections/%s" % cam_id)
        )
        os.makedirs(snap_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(snap_dir, "%s_%s.jpg" % (label, ts))
        cv2.imwrite(img_path, frame)
        log.info("[%s] SNAPSHOT  %s %.0f%%  %s",
                 cam_id, label, conf * 100, os.path.basename(img_path))
    if mqtt_e:
        topic   = TOPIC_MAP.get(cam_id, {}).get(label, "%s/detection" % cam_id)
        payload = json.dumps({
            "class":      label,
            "confidence": round(conf, 3),
            "timestamp":  datetime.now().isoformat(),
            "source":     cam_id,
        })
        try:
            _mqtt.publish(topic, payload)
        except Exception as e:
            log.warning("[%s] MQTT publish failed: %s", cam_id, e)
    log_event(cam_id, label, conf, img_path)


# ─────────────────────────────────────────────
# PER-FRAME PROCESSING
# ─────────────────────────────────────────────

def process_frame(proc: CameraProcessor, frame, model, now: float):
    cam_id = proc.cam_id
    h, w   = frame.shape[:2]
    cx0    = w // 2
    cy0    = h // 2

    # Snapshot config values once per frame
    raw_watch = cam_cfg(cam_id, "watch_classes") or ["person"]

    # Backward compatible watch_classes handling:
    # Old format: ["person", "car"] uses shared camera confidence.
    # New format: {"person": 0.55, "car": 0.70} uses per-class confidence.
    legacy_conf = cam_cfg(cam_id, "confidence")
    if legacy_conf is None:
        legacy_conf = 0.50

    if isinstance(raw_watch, dict):
        watch_conf = {}
        for label, value in raw_watch.items():
            try:
                watch_conf[label] = float(value)
            except (TypeError, ValueError):
                watch_conf[label] = float(legacy_conf)
    else:
        watch_conf = {label: float(legacy_conf) for label in raw_watch}

    # Use a low YOLO floor, then apply our own per-class threshold below.
    # This allows deer/person/car/etc. to each have different thresholds.
    yolo_floor = 0.05

    snap   = cam_cfg(cam_id, "snapshots")    if cam_cfg(cam_id, "snapshots")    is not None else False
    mqtt_e = cam_cfg(cam_id, "mqtt_enabled") if cam_cfg(cam_id, "mqtt_enabled") is not None else False
    mon    = cam_cfg(cam_id, "monitor_only") or False
    track  = cam_cfg(cam_id, "tracking")     if cam_cfg(cam_id, "tracking")     is not None else True

    # YOLO inference (shared GPU model)
    results   = model(frame, conf=yolo_floor, verbose=False)
    annotated = frame.copy()
    raw_boxes = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            if label not in watch_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < watch_conf[label]:
                continue

            # Frontyard: skip parked vehicles
            if proc.vtracker is not None and label in VEHICLE_CLASSES:
                if not proc.vtracker.is_moving(label, x1, y1, x2, y2):
                    log.debug("[frontyard] PARKED  %s %.0f%%", label, conf * 100)
                    continue

            raw_boxes.append((label, conf, x1, y1, x2, y2))

    # ── PTZ camera ────────────────────────────────────────────
    if proc.ptz is not None:
        ptz = proc.ptz
        p   = ptz.p

        # Confirmation filter (decay on miss, increment on hit)
        seen = {b[0] for b in raw_boxes}
        for lbl in list(ptz.confirm_count.keys()):
            if lbl not in seen:
                ptz.confirm_count[lbl] = max(0, ptz.confirm_count[lbl] - 1)
        for lbl in seen:
            ptz.confirm_count[lbl] = ptz.confirm_count.get(lbl, 0) + 1

        confirmed = [b for b in raw_boxes
                     if ptz.confirm_count.get(b[0], 0) >= p["CONFIRM_FRAMES"]]

        if confirmed:
            # Priority: person first, then largest box
            label, conf, x1, y1, x2, y2 = sorted(
                confirmed,
                key=lambda b: (0 if b[0] == "person" else 1,
                               -((b[4] - b[2]) * (b[5] - b[3])))
            )[0]
            ptz.last_detection = now

            target_cx = (x1 + x2) // 2
            target_cy = y1 + (y2 - y1) // 3
            offset_x  = target_cx - cx0
            offset_y  = target_cy - cy0

            # Pan / tilt
            if (not mon) and track and now - ptz.last_ptz_cmd > p["PTZ_COOLDOWN"]:
                pan  = ptz.calc_speed(offset_x, cx0, p["Kp_pan"])
                tilt = -ptz.calc_speed(offset_y, cy0, p["Kp_tilt"])
                if pan != 0 or tilt != 0:
                    ptz.move(pan, tilt, 0.0)
                else:
                    ptz.stop()
                    # Zoom only when centered
                    if now - ptz.last_zoom_cmd > p["ZOOM_COOLDOWN"]:
                        box_ratio = (y2 - y1) / h
                        zoom_err  = p["ZOOM_TARGET_RATIO"] - box_ratio
                        if abs(zoom_err) > p["ZOOM_TOLERANCE"]:
                            ptz.move(0.0, 0.0,
                                     p["ZOOM_SPEED"] if zoom_err > 0 else -p["ZOOM_SPEED"])
                            ptz.last_zoom_cmd = now
                ptz.last_ptz_cmd = now
            elif mon or not track:
                ptz.stop()

            # Annotate
            mode_label = "MONITOR" if (mon or not track) else "TRACKING"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.circle(annotated, (target_cx, target_cy), 8, (0, 200, 80), -1)
            cv2.line(annotated, (cx0, cy0), (target_cx, target_cy), (0, 255, 255), 1)
            cv2.putText(annotated, "%s %d%%" % (label, int(conf * 100)),
                        (x1, max(y1 - 10, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 80), 2)
            cv2.putText(annotated, "%s  offset(%d,%d)" % (mode_label, offset_x, offset_y),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

            # Snapshot / MQTT / event log
            if proc.cooldown_ok(label):
                _trigger_action(cam_id, label, conf, annotated, snap, mqtt_e)

        else:
            # No confirmed detection
            since_last = now - ptz.last_detection
            if since_last >= 0.5:
                ptz.stop()

            # Zoom out after losing target
            if (not mon) and track and ptz.last_detection > 0 and since_last > p["ZOOM_OUT_DELAY"]:
                if now - ptz.last_zoom_cmd > p["ZOOM_COOLDOWN"]:
                    ptz.move(0.0, 0.0, -p["ZOOM_SPEED"])
                    ptz.last_zoom_cmd = now

            cv2.putText(annotated, "IDLE" if (not mon and track) else "MONITOR",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)

        # Crosshair
        cv2.line(annotated, (cx0 - 30, cy0), (cx0 + 30, cy0), (255, 255, 255), 1)
        cv2.line(annotated, (cx0, cy0 - 30), (cx0, cy0 + 30), (255, 255, 255), 1)

    # ── Fixed camera (frontyard) ──────────────────────────────
    else:
        for label, conf, x1, y1, x2, y2 in raw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(annotated, "%s %d%%" % (label, int(conf * 100)),
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 80), 2)

            if not mon and proc.cooldown_ok(label):
                _trigger_action(cam_id, label, conf, annotated, snap, mqtt_e)

    # Timestamp
    cv2.putText(annotated, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    proc.put_frame(annotated)


# ─────────────────────────────────────────────
# FLASK — MJPEG STREAMS  (port 8081)
# ─────────────────────────────────────────────

flask_app  = Flask(__name__)
CORS(flask_app)
_processors: dict = {}  # populated in main() before inference starts


def _generate_stream(cam_id: str):
    proc = _processors.get(cam_id)
    if proc is None:
        return
    while True:
        time.sleep(0.04)
        frame = proc.get_frame()
        if frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


@flask_app.route("/stream/<cam_id>")
def stream_route(cam_id):
    if cam_id not in _processors:
        return Response("Unknown camera: %s" % cam_id, status=404)
    return Response(_generate_stream(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/status/<cam_id>")
def status_route(cam_id):
    proc = _processors.get(cam_id)
    if proc is None:
        return Response("Not found", status=404)
    return jsonify({"camera": cam_id, "online": proc.get_frame() is not None,
                    "mode": "ACTIVE"})


@flask_app.route("/health")
def health():
    return jsonify({cam: _processors[cam].get_frame() is not None
                    for cam in _processors})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load YOLO ─────────────────────────────
    log.info("Loading YOLO: %s on GPU ...", YOLO_MODEL)
    model = YOLO(YOLO_MODEL)
    model.to("cuda")
    log.info("YOLO ready on GPU")

    # ── Init cameras ──────────────────────────
    cameras_conf = _load_full_config()["cameras"]
    workers:    dict = {}
    processors: dict = {}

    for cam_id, ccfg in cameras_conf.items():
        workers[cam_id]    = CameraWorker(cam_id, ccfg["rtsp_url"])
        processors[cam_id] = CameraProcessor(cam_id)

    # ── Init PTZ controllers ───────────────────
    for cam_id, ccfg in cameras_conf.items():
        if ccfg.get("type") == "ptz" and cam_id in PTZ_PARAMS:
            try:
                ptz = PTZController(
                    cam_id,
                    ccfg["host"],
                    ccfg.get("onvif_port", 80),
                    ccfg.get("ptz_user", "admin"),
                    ccfg.get("ptz_pass", ""),
                )
                processors[cam_id].set_ptz(ptz)
            except Exception as e:
                log.error("[%s] PTZ init failed: %s — running without auto-PTZ", cam_id, e)

    # ── Expose to Flask ────────────────────────
    global _processors
    _processors = processors

    # ── Start Flask ────────────────────────────
    threading.Thread(
        target=lambda: flask_app.run(host="0.0.0.0", port=8081,
                                     debug=False, use_reloader=False, threaded=True),
        daemon=True,
        name="flask-streams",
    ).start()
    log.info("MJPEG streams → http://192.168.1.17:8081/stream/{frontyard,backyard,indoor}")

    # ── Wait for first frames ──────────────────
    log.info("Waiting for camera frames...")
    for cam_id, worker in workers.items():
        for _ in range(40):
            if worker.read() is not None:
                log.info("[%s] first frame ready", cam_id)
                break
            time.sleep(0.5)
        else:
            log.warning("[%s] no frame after 20s — will keep trying", cam_id)

    log.info("Inference loop starting. Ctrl+C to stop.")

    # ── Central inference loop ─────────────────
    frame_count = 0
    try:
        while True:
            now = time.time()
            frame_count += 1

            if frame_count % 30 == 0:
                reload_config()

            for cam_id, worker in workers.items():
                frame = worker.read()
                if frame is None:
                    continue
                try:
                    process_frame(processors[cam_id], frame, model, now)
                except Exception as e:
                    log.warning("[%s] process_frame error: %s", cam_id, e)

            time.sleep(0.03)

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        for proc in processors.values():
            if proc.ptz:
                try:
                    proc.ptz.stop()
                except Exception:
                    pass
        for worker in workers.values():
            worker.stop()
        _mqtt.loop_stop()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
