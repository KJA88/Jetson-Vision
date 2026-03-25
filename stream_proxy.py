"""
stream_proxy.py — MJPEG streaming server for DHRAS (Session 1B)

Reads annotated frames from CameraWorker instances that are already being
populated by vision_service.py. Falls back to the raw frame if no annotated
frame exists yet, and shows an OFFLINE slate if neither is available.

Started in a daemon thread by VisionService.start().

Browser URLs (Jetson at 192.168.1.17):
  http://192.168.1.17:8080/                       — all cameras on one page
  http://192.168.1.17:8080/stream/frontyard        — MJPEG stream
  http://192.168.1.17:8080/stream/backyard
  http://192.168.1.17:8080/stream/indoor
  http://192.168.1.17:8080/snapshot/frontyard      — single JPEG (phone-friendly)
  http://192.168.1.17:8080/status                  — camera status as JSON
"""

import logging
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request as flask_request
from onvif import ONVIFCamera

logger = logging.getLogger("stream_proxy")

STREAM_FPS = 10          # target FPS for MJPEG delivery to browser
JPEG_QUALITY = 75        # 0-100; lower = faster / smaller, higher = sharper

# ---------------------------------------------------------------------------
# Index page — plain HTML, no JS framework
# ---------------------------------------------------------------------------
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DHRAS Cameras</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #ddd; font-family: monospace; padding: 12px; }
    header { font-size: 0.8rem; color: #666; margin-bottom: 12px; }
    .grid { display: flex; flex-wrap: wrap; gap: 10px; }
    .cam { flex: 1 1 300px; max-width: 700px; }
    .cam-title { font-size: 0.8rem; color: #aaa; padding: 4px 0; display: flex; justify-content: space-between; }
    .cam-title a { color: #555; text-decoration: none; }
    .cam-title a:hover { color: #888; }
    .cam img { width: 100%; display: block; background: #1a1a1a; min-height: 180px; }
  </style>
</head>
<body>
  <header>DHRAS &middot; Live Feeds &middot; {{ count }} camera{{ 's' if count != 1 else '' }}</header>
  <div class="grid">
    {% for cam_id, name in cameras %}
    <div class="cam">
      <div class="cam-title">
        <span>{{ name }}</span>
        <span>
          <a href="/snapshot/{{ cam_id }}">snapshot</a> &nbsp;
          <a href="/stream/{{ cam_id }}">raw stream</a>
        </span>
      </div>
      <img src="/stream/{{ cam_id }}" alt="{{ cam_id }}" loading="lazy">
    </div>
    {% endfor %}
  </div>
</body>
</html>
"""


_VALID_DIRS = {"up", "down", "left", "right", "zoomin", "zoomout", "stop"}


class _PtzController:
    """
    Lazy-initialising ONVIF PTZ wrapper for one camera.
    ONVIFCamera connection is created on first use and cached.
    Re-connects automatically if a command fails.
    A threading.Lock prevents overlapping ContinuousMove calls.
    """

    SPEED = 0.5  # normalised velocity magnitude, 0.0–1.0

    def __init__(self, cam_id: str, cfg: dict):
        self._cam_id = cam_id
        self._host = cfg["host"]
        self._port = cfg.get("onvif_port", 80)
        self._user = cfg.get("ptz_user", "admin")
        self._pass = cfg.get("ptz_pass", "")
        self._lock = threading.Lock()
        self._ptz = None
        self._token = None

    # ------------------------------------------------------------------
    def send(self, direction: str) -> tuple:
        """Send a PTZ command. Returns (ok: bool, message: str)."""
        with self._lock:
            try:
                if self._ptz is None:
                    self._connect()
                if direction == "stop":
                    self._do_stop()
                else:
                    self._do_move(direction)
                return True, "ok"
            except Exception as exc:
                logger.warning(f"[{self._cam_id}] PTZ '{direction}' failed: {exc}")
                # Invalidate so next call triggers a fresh connect
                self._ptz = None
                self._token = None
                return False, str(exc)

    # ------------------------------------------------------------------
    def _connect(self):
        cam = ONVIFCamera(self._host, self._port, self._user, self._pass)
        ptz = cam.create_ptz_service()
        token = cam.create_media_service().GetProfiles()[0].token
        self._ptz = ptz
        self._token = token
        logger.info(f"[{self._cam_id}] ONVIF PTZ connected (token={token})")

    def _do_move(self, direction: str):
        s = self.SPEED
        vx = vy = vz = 0.0
        if   direction == "left":    vx = -s
        elif direction == "right":   vx = +s
        elif direction == "up":      vy = +s
        elif direction == "down":    vy = -s
        elif direction == "zoomin":  vz = +s
        elif direction == "zoomout": vz = -s

        req = self._ptz.create_type("ContinuousMove")
        req.ProfileToken = self._token
        req.Velocity = {"PanTilt": {"x": vx, "y": vy}, "Zoom": {"x": vz}}
        self._ptz.ContinuousMove(req)

    def _do_stop(self):
        req = self._ptz.create_type("Stop")
        req.ProfileToken = self._token
        req.PanTilt = True
        req.Zoom = True
        self._ptz.Stop(req)


class StreamProxy:
    def __init__(self, workers: dict, cam_configs: dict, port: int = 8080):
        """
        workers:     {cam_id: CameraWorker} — shared reference from VisionService
        cam_configs: {cam_id: config dict}  — for display names
        port:        HTTP port to bind on all interfaces
        """
        self._workers = workers
        self._cam_configs = cam_configs
        self._port = port

        # PTZ controllers — only for cameras with type=="ptz"
        self._ptz_controllers: dict[str, _PtzController] = {
            cam_id: _PtzController(cam_id, cfg)
            for cam_id, cfg in cam_configs.items()
            if cfg.get("type") == "ptz"
        }

        # Silence Flask/werkzeug request logs — they'd flood vision_service.log
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        logging.getLogger("flask.app").setLevel(logging.WARNING)

        self._app = Flask(__name__)
        self._register_routes()

    def start_in_thread(self):
        t = threading.Thread(
            target=lambda: self._app.run(
                host="0.0.0.0",
                port=self._port,
                threaded=True,
                use_reloader=False,
            ),
            name="stream_proxy",
            daemon=True,
        )
        t.start()
        logger.info(f"Stream proxy listening on http://0.0.0.0:{self._port}")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def _register_routes(self):
        app = self._app
        workers = self._workers
        cam_configs = self._cam_configs

        @app.route("/")
        def index():
            cameras = [
                (cam_id, cfg.get("name", cam_id))
                for cam_id, cfg in cam_configs.items()
            ]
            return render_template_string(
                _INDEX_HTML, cameras=cameras, count=len(cameras)
            )

        @app.route("/stream/<cam_id>")
        def stream(cam_id):
            worker = workers.get(cam_id)
            if worker is None:
                return f"Unknown camera: {cam_id}", 404
            return Response(
                _mjpeg_generator(worker, cam_id),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/snapshot/<cam_id>")
        def snapshot(cam_id):
            worker = workers.get(cam_id)
            if worker is None:
                return f"Unknown camera: {cam_id}", 404
            annotated = worker.get_annotated_frame()
            frame = annotated if annotated is not None else worker.get_frame()
            if frame is None:
                frame = _offline_frame(cam_id)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return Response(buf.tobytes(), mimetype="image/jpeg")

        @app.route("/status")
        def status():
            now = time.time()
            result = {}
            for cam_id, worker in workers.items():
                result[cam_id] = {
                    "online": worker.online,
                    "paused": worker.paused,
                    "stale": worker.is_stale,
                    "last_frame_age_sec": round(now - worker.last_frame_time, 1),
                }
            return jsonify(result)

        ptz_controllers = self._ptz_controllers

        @app.route("/ptz/<cam_id>", methods=["POST"])
        def ptz(cam_id):
            controller = ptz_controllers.get(cam_id)
            if controller is None:
                return jsonify({"error": f"'{cam_id}' is not a PTZ camera"}), 400

            body = flask_request.get_json(silent=True) or {}
            direction = str(body.get("dir", "")).lower()

            if direction not in _VALID_DIRS:
                return jsonify({
                    "error": f"Invalid dir '{direction}'",
                    "valid": sorted(_VALID_DIRS),
                }), 400

            ok, msg = controller.send(direction)
            status_code = 200 if ok else 502
            return jsonify({"cam_id": cam_id, "dir": direction, "ok": ok, "msg": msg}), status_code


# ---------------------------------------------------------------------------
# MJPEG generator — runs in Flask's per-request thread
# ---------------------------------------------------------------------------
def _mjpeg_generator(worker, cam_id: str):
    interval = 1.0 / STREAM_FPS
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        t0 = time.monotonic()

        annotated = worker.get_annotated_frame()
        frame = annotated if annotated is not None else worker.get_frame()
        if frame is None:
            frame = _offline_frame(cam_id)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            time.sleep(interval)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )

        elapsed = time.monotonic() - t0
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Offline slate — shown when camera has no frames yet
# ---------------------------------------------------------------------------
def _offline_frame(cam_id: str):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"{cam_id}  OFFLINE",
        (160, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (60, 60, 60),
        2,
    )
    return frame
