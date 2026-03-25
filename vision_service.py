"""
vision_service.py — DHRAS Shared Vision Service (Session 1A)

Runs on Jetson Orin Nano (192.168.1.17).

Responsibilities:
  - Load YOLOv8n once on GPU at startup
  - Create one CameraWorker per camera (RTSP reader threads)
  - Central inference loop: round-robin cameras by priority, 120ms cycle budget
  - Detection persistence: confirm class in 2/3 frames, enforce cooldown
  - Publish detections to MQTT (dhras/cameras/{cam_id}/detection)
  - Save snapshots and append to events.jsonl
  - Publish heartbeat to dhras/health/jetson every 5 seconds

Session 1B adds: stream_proxy.py (reads annotated frames from workers),
                 ptz_controller.py (subscribes to detections, drives ONVIF).
"""

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

from camera_worker import CameraWorker
from stream_proxy import StreamProxy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "cameras_config.json"
DETECTIONS_DIR = BASE_DIR / "detections"
EVENTS_LOG = DETECTIONS_DIR / "events.jsonl"
LOGS_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
DETECTIONS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "vision_service.log"),
    ],
)
logger = logging.getLogger("vision_service")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MQTT client wrapper
# Handles reconnection automatically (paho loop_start).
# Supports exact topic and wildcard subscriptions.
# ---------------------------------------------------------------------------
class MQTTClient:
    def __init__(self, broker_host: str, broker_port: int = 1883):
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._client = mqtt.Client(client_id="dhras-vision")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._subscriptions: dict[str, callable] = {}
        self._lock = threading.Lock()
        self.connected = False

    def connect(self):
        self._client.connect_async(self._broker_host, self._broker_port, keepalive=60)
        self._client.loop_start()

    def subscribe(self, topic: str, callback: callable):
        with self._lock:
            self._subscriptions[topic] = callback
        self._client.subscribe(topic)

    def publish(self, topic: str, payload, retain: bool = False):
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        self._client.publish(topic, payload, retain=retain)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"MQTT connected to {self._broker_host}:{self._broker_port}")
            self.connected = True
            # Re-subscribe after reconnect
            with self._lock:
                for topic in self._subscriptions:
                    client.subscribe(topic)
        else:
            logger.warning(f"MQTT connect failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        logger.warning(f"MQTT disconnected (rc={rc}) — paho will auto-reconnect")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        callback = None
        with self._lock:
            # Exact match first, then wildcard scan
            if topic in self._subscriptions:
                callback = self._subscriptions[topic]
            else:
                for sub_topic, sub_cb in self._subscriptions.items():
                    if mqtt.topic_matches_sub(sub_topic, topic):
                        callback = sub_cb
                        break
        if callback:
            try:
                payload = json.loads(msg.payload.decode())
                callback(topic, payload)
            except Exception as exc:
                logger.error(f"Error handling MQTT message on {topic}: {exc}")


# ---------------------------------------------------------------------------
# Detection persistence tracker
#
# Tracks the last 3 inference results per (camera, class).
# A class is "confirmed" when it appears in >= 2 of the last 3 frames.
# Cooldown prevents re-publishing the same camera+class too frequently.
# ---------------------------------------------------------------------------
class DetectionTracker:
    WINDOW = 3
    MIN_HITS = 2

    def __init__(self, cooldown_sec: float = 60.0):
        self.cooldown_sec = cooldown_sec
        # cam_id → class_name → deque[bool]
        self._history: dict = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.WINDOW)))
        # cam_id → class_name → timestamp of last publish
        self._last_published: dict = defaultdict(lambda: defaultdict(float))

    def update(self, cam_id: str, detected_classes: set) -> list:
        """
        Record inference result. Returns list of class names that are confirmed
        and past cooldown — these should be published.

        detected_classes: set of class names that appeared above confidence threshold.
        """
        # Record presence/absence for every class this camera has ever seen
        all_tracked = set(self._history[cam_id].keys()) | detected_classes
        for cls in all_tracked:
            self._history[cam_id][cls].append(cls in detected_classes)

        now = time.time()
        to_publish = []
        for cls in detected_classes:
            hits = sum(self._history[cam_id][cls])
            if hits >= self.MIN_HITS:
                if now - self._last_published[cam_id][cls] >= self.cooldown_sec:
                    to_publish.append(cls)
                    self._last_published[cam_id][cls] = now

        return to_publish


# ---------------------------------------------------------------------------
# Priority scheduler
#
# Decides which cameras are due for inference this cycle.
# Cameras are ranked by urgency = time_since_last_inference * target_fps.
# Starvation protection: any camera idle >= 3 seconds is forced to the front.
#
# FPS targets per active trigger count (from architecture spec):
#   0 triggered  → base_fps
#   1 triggered  → triggered cam: triggered_fps,     others: 1.5 FPS
#   2 triggered  → triggered cams: triggered_fps×0.65, others: 1.0 FPS
#   3+ triggered → triggered cams: triggered_fps×0.40, others: 1.0 FPS
# ---------------------------------------------------------------------------
class PriorityScheduler:
    STARVATION_LIMIT_SEC = 3.0
    TRIGGER_TIMEOUT_SEC = 30.0   # seconds of no detection before trigger expires

    def __init__(self, camera_configs: dict):
        self._configs = camera_configs
        self._last_inferred: dict[str, float] = {cam_id: 0.0 for cam_id in camera_configs}
        self._triggered: dict[str, bool] = {cam_id: False for cam_id in camera_configs}
        self._trigger_expires: dict[str, float] = {cam_id: 0.0 for cam_id in camera_configs}

    def set_triggered(self, cam_id: str, triggered: bool = True):
        self._triggered[cam_id] = triggered
        if triggered:
            self._trigger_expires[cam_id] = time.time() + self.TRIGGER_TIMEOUT_SEC
            logger.debug(f"[{cam_id}] Triggered — boosting inference priority")

    def mark_inferred(self, cam_id: str):
        self._last_inferred[cam_id] = time.time()

    def get_target_fps(self, cam_id: str) -> float:
        self._expire_triggers()
        triggered_count = sum(self._triggered.values())
        cfg = self._configs[cam_id]
        base = cfg.get("base_fps", 3)
        trig = cfg.get("triggered_fps", 10)

        if triggered_count == 0:
            return base
        elif triggered_count == 1:
            return trig if self._triggered[cam_id] else 1.5
        elif triggered_count == 2:
            return trig * 0.65 if self._triggered[cam_id] else 1.0
        else:  # 3+
            return trig * 0.40 if self._triggered[cam_id] else 1.0

    def reduce_resolution_needed(self) -> bool:
        """True when 3+ cameras are triggered simultaneously."""
        return sum(self._triggered.values()) >= 3

    def get_ordered_cameras(self, active_cam_ids: list) -> list:
        """
        Return cameras that are due for inference, sorted by urgency.
        Starvation-forced cameras come first regardless of urgency score.
        Cameras not yet due for their next frame are excluded this cycle.
        """
        self._expire_triggers()
        now = time.time()
        forced = []
        due = []

        for cam_id in active_cam_ids:
            elapsed = now - self._last_inferred[cam_id]

            # Starvation protection: force-include if idle too long
            if elapsed >= self.STARVATION_LIMIT_SEC:
                forced.append(cam_id)
                continue

            target_fps = self.get_target_fps(cam_id)
            if target_fps <= 0:
                continue
            target_interval = 1.0 / target_fps

            if elapsed >= target_interval:
                urgency = elapsed * target_fps  # larger = more overdue
                due.append((urgency, cam_id))

        due.sort(reverse=True)  # most overdue first
        return forced + [cam_id for _, cam_id in due]

    def _expire_triggers(self):
        now = time.time()
        for cam_id, active in list(self._triggered.items()):
            if active and now >= self._trigger_expires[cam_id]:
                self._triggered[cam_id] = False
                logger.debug(f"[{cam_id}] Trigger expired — returning to base priority")


# ---------------------------------------------------------------------------
# Snapshot + event log helpers
# ---------------------------------------------------------------------------
def save_snapshot(frame, cam_id: str, class_name: str) -> str:
    """Save annotated frame as JPEG. Returns absolute path."""
    cam_dir = DETECTIONS_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = cam_dir / f"{class_name}_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


_events_lock = threading.Lock()

def append_event(event: dict):
    """Append a detection event to the fast JSONL log. Thread-safe."""
    with _events_lock:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# VisionService — main class
# ---------------------------------------------------------------------------
class VisionService:
    HEARTBEAT_INTERVAL_SEC = 5.0
    CYCLE_BUDGET_MS = 120.0          # max allowed inference+annotate+publish time
    SCALE_STEP_DOWN = 0.10           # reduce resolution by this much when over budget
    SCALE_STEP_UP = 0.05             # recover resolution gradually when under budget
    SCALE_MIN = 0.50                 # never go below 50% of original resolution
    SCALE_RECOVERY_THRESHOLD = 0.75  # recover when cycle < 75% of budget

    def __init__(self, config: dict):
        self._config = config
        mqtt_cfg = config.get("mqtt", {})
        self._broker_host = mqtt_cfg.get("broker", "192.168.1.18")
        self._broker_port = mqtt_cfg.get("port", 1883)
        self._cooldown_sec = config.get("cooldown_sec", 60)

        self._cam_configs: dict = config["cameras"]
        self._workers: dict[str, CameraWorker] = {}
        self._model = None
        self._mqtt = MQTTClient(self._broker_host, self._broker_port)
        self._tracker = DetectionTracker(cooldown_sec=self._cooldown_sec)
        self._scheduler: PriorityScheduler = None
        self._input_scale = 1.0  # dynamic resolution scale
        self._running = False
        self._write_queue: queue.Queue = queue.Queue()
        self._config_mtime: float = 0.0  # mtime of last loaded cameras_config.json

        # FPS tracking per camera for status logging
        self._fps_counter: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))

    def start(self):
        logger.info("=" * 60)
        logger.info("DHRAS Vision Service starting")

        # Load model once on GPU — fail immediately with a clear message if missing
        model_path = BASE_DIR / "yolov8n.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}\n"
                f"Download it with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {model_path}"
            )
        logger.info(f"Loading YOLO model: {model_path}")
        self._model = YOLO(str(model_path))
        self._model.to("cuda")
        # Warm up model to avoid first-frame latency spike
        import numpy as np
        self._model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("YOLO model loaded and warmed up on GPU")

        # Create camera workers
        for cam_id, cam_cfg in self._cam_configs.items():
            worker = CameraWorker(cam_id, cam_cfg)
            self._workers[cam_id] = worker
            (DETECTIONS_DIR / cam_id).mkdir(parents=True, exist_ok=True)
            logger.info(f"[{cam_id}] Worker created — {cam_cfg.get('name', cam_id)}")

        self._scheduler = PriorityScheduler(self._cam_configs)

        # Connect MQTT
        self._mqtt.connect()
        time.sleep(1.0)  # allow connection to establish before subscribing

        # Subscribe to per-camera command topics
        for cam_id in self._workers:
            self._mqtt.subscribe(
                f"dhras/cameras/{cam_id}/command",
                self._handle_camera_command,
            )
        # Subscribe to radar priority-boost events (published by sensor_hub, Session 3)
        self._mqtt.subscribe("dhras/sensors/radar/#", self._handle_radar_event)

        # Background writer thread — snapshot saves and JSONL writes happen here,
        # never on the inference loop. Daemon=False so it flushes on clean shutdown.
        self._running = True
        self._config_mtime = CONFIG_PATH.stat().st_mtime  # baseline — don't reload on first poll
        threading.Thread(
            target=self._writer_loop, name="writer", daemon=False
        ).start()
        threading.Thread(
            target=self._heartbeat_loop, name="heartbeat", daemon=True
        ).start()
        threading.Thread(
            target=self._config_reload_loop, name="config-reload", daemon=True
        ).start()

        # MJPEG stream server — reads annotated frames from workers
        stream_port = 8082
        StreamProxy(self._workers, self._cam_configs, port=stream_port).start_in_thread()

        # Run inference loop — blocks until stop()
        logger.info("Entering inference loop")
        try:
            self._inference_loop()
        except Exception as exc:
            logger.exception(f"Inference loop crashed: {exc}")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        for w in self._workers.values():
            w.stop()
        # Sentinel tells writer thread to drain and exit
        self._write_queue.put(None)
        self._write_queue.join()
        logger.info("VisionService stopped")

    # ------------------------------------------------------------------
    # MQTT command handlers
    # ------------------------------------------------------------------
    def _handle_camera_command(self, topic: str, payload: dict):
        # topic: dhras/cameras/{cam_id}/command
        parts = topic.split("/")
        if len(parts) < 4:
            return
        cam_id = parts[2]
        worker = self._workers.get(cam_id)
        if not worker:
            logger.warning(f"Command received for unknown camera: {cam_id}")
            return

        action = payload.get("action", "")
        if action == "pause":
            worker.pause()
        elif action == "resume":
            worker.resume()
        elif action == "set_triggered":
            self._scheduler.set_triggered(cam_id, payload.get("triggered", True))
        elif action == "reload_config":
            # Dashboard POSTed new config to /api/config/{cam_id} and saved the file;
            # apply it immediately rather than waiting for the 10s poll cycle.
            self._reload_config()
        else:
            logger.debug(f"[{cam_id}] Unhandled command action: {action}")

    def _handle_radar_event(self, topic: str, payload: dict):
        # Radar presence event from sensor_hub can carry a camera_id to boost.
        # Full zone-to-camera mapping is implemented in Session 3 on Pi5.
        cam_id = payload.get("camera_id")
        if cam_id and cam_id in self._workers:
            self._scheduler.set_triggered(cam_id, True)
            logger.info(f"[{cam_id}] Priority boosted by radar event on {topic}")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def _heartbeat_loop(self):
        while self._running:
            active_count = sum(
                1 for w in self._workers.values() if not w.paused and not w.is_stale
            )
            self._mqtt.publish(
                "dhras/health/jetson",
                {
                    "timestamp": time.time(),
                    "service": "vision_service",
                    "active_cameras": active_count,
                    "input_scale": round(self._input_scale, 2),
                },
            )
            time.sleep(self.HEARTBEAT_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Config hot-reload
    # ------------------------------------------------------------------

    # Fields the dashboard is allowed to change without restarting.
    # Structural fields (rtsp_url, host, type, base_fps, …) require restart.
    _RELOADABLE_FIELDS = {"watch_classes", "confidence", "snapshots", "monitor_only", "tracking"}

    def _config_reload_loop(self):
        """Poll cameras_config.json every 10 s. Reload when mtime changes."""
        while self._running:
            time.sleep(10)
            try:
                mtime = CONFIG_PATH.stat().st_mtime
                if mtime != self._config_mtime:
                    self._config_mtime = mtime
                    self._reload_config()
            except Exception as exc:
                logger.error(f"Config reload poll error: {exc}")

    def _reload_config(self):
        """
        Read cameras_config.json and apply hot-reloadable fields to every
        running camera. Each camera's config dict is replaced atomically so
        the inference loop always sees a consistent snapshot — it snapshots
        cam_cfg = self._cam_configs[cam_id] at the top of each camera's
        processing block, so the new config takes effect on the next cycle.
        """
        try:
            with open(CONFIG_PATH) as f:
                new_config = json.load(f)
        except Exception as exc:
            logger.error(f"Config reload: failed to parse {CONFIG_PATH}: {exc}")
            return

        new_cameras = new_config.get("cameras", {})
        any_change = False

        for cam_id, new_cfg in new_cameras.items():
            if cam_id not in self._cam_configs:
                logger.debug(f"[{cam_id}] New camera in config — restart required to add worker")
                continue

            old_cfg = self._cam_configs[cam_id]
            changes = {
                field: (old_cfg.get(field), new_cfg[field])
                for field in self._RELOADABLE_FIELDS
                if field in new_cfg and new_cfg[field] != old_cfg.get(field)
            }

            if not changes:
                continue

            # Build a fully merged dict and replace the reference atomically.
            # The inference loop holds a local reference to the old dict for the
            # current camera's cycle; the new dict is picked up next cycle.
            self._cam_configs[cam_id] = {**old_cfg, **{f: new_cfg[f] for f in self._RELOADABLE_FIELDS if f in new_cfg}}
            any_change = True

            for field, (old_val, new_val) in changes.items():
                logger.info(f"[{cam_id}] config reload: {field} {old_val!r} → {new_val!r}")

        if any_change:
            logger.info("Config hot-reload complete")

    # ------------------------------------------------------------------
    # Background writer — snapshot saves and JSONL appends
    # ------------------------------------------------------------------
    def _writer_loop(self):
        """
        Consumes write tasks from the queue. Runs in its own thread so that
        cv2.imwrite() and file I/O never stall the inference loop.
        Exits when it receives the None sentinel from stop().
        """
        while True:
            task = self._write_queue.get()
            try:
                if task is None:  # shutdown sentinel
                    break
                snapshot_path = task["snapshot_path"]
                frame = task["frame"]
                event = task["event"]

                if snapshot_path and frame is not None:
                    try:
                        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(snapshot_path, frame)
                    except Exception as exc:
                        logger.error(f"Snapshot write failed ({snapshot_path}): {exc}")

                try:
                    with open(EVENTS_LOG, "a") as f:
                        f.write(json.dumps(event) + "\n")
                except Exception as exc:
                    logger.error(f"Event log write failed: {exc}")

            finally:
                self._write_queue.task_done()

        logger.info("Writer thread exited")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    def _inference_loop(self):
        status_publish_interval = 5.0
        last_status_publish = 0.0

        while self._running:
            # A camera is active when it's not paused and has sent a frame recently.
            # is_stale covers both "never connected" and "stream dropped" — it is
            # the correct signal here rather than worker.online, which can lag on
            # a silent stream that hasn't formally errored yet.
            active = [
                cam_id
                for cam_id, w in self._workers.items()
                if not w.paused and not w.is_stale
            ]

            if not active:
                logger.debug("No active cameras — waiting")
                time.sleep(1.0)
                continue

            ordered = self._scheduler.get_ordered_cameras(active)
            if not ordered:
                # All cameras are up to date — sleep until next camera is due
                time.sleep(0.01)
                continue

            reduce_res = self._scheduler.reduce_resolution_needed()
            cycle_start = time.perf_counter()
            over_budget = False

            for cam_id in ordered:
                worker = self._workers[cam_id]
                frame = worker.get_frame()
                if frame is None:
                    # Worker online but no frame yet — skip this camera this cycle
                    continue

                # Apply dynamic resolution scaling
                scale = self._input_scale
                if reduce_res and scale > 0.75:
                    # 3-way trigger: force additional reduction
                    scale = min(scale, 0.75)

                if scale < 1.0:
                    h, w = frame.shape[:2]
                    frame_in = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    frame_in = frame

                # --- YOLO inference ---
                results = self._model(frame_in, verbose=False)[0]

                # --- Process detections ---
                cam_cfg = self._cam_configs[cam_id]
                conf_threshold = cam_cfg.get("confidence", 0.60)
                watch_classes = set(cam_cfg.get("watch_classes", []))
                snapshots_enabled = cam_cfg.get("snapshots", False)
                mqtt_enabled = cam_cfg.get("mqtt_enabled", True)

                # best confidence per class for confirmed detections
                detected_classes_conf: dict[str, float] = {}
                # all boxes for annotation and publish payload
                boxes_data: list[dict] = []

                if results.boxes is not None and len(results.boxes):
                    for box in results.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self._model.names[cls_id]

                        if conf < conf_threshold:
                            continue
                        if watch_classes and cls_name not in watch_classes:
                            continue

                        xyxy = [round(float(v), 1) for v in box.xyxy[0]]
                        boxes_data.append({
                            "class": cls_name,
                            "confidence": round(conf, 3),
                            "bbox": xyxy,
                        })

                        if cls_name not in detected_classes_conf or conf > detected_classes_conf[cls_name]:
                            detected_classes_conf[cls_name] = conf

                # Scale bounding boxes back to original resolution if we downscaled
                if scale < 1.0 and boxes_data:
                    inv = 1.0 / scale
                    for b in boxes_data:
                        b["bbox"] = [round(v * inv, 1) for v in b["bbox"]]

                # --- Annotate frame (bounding boxes drawn here, not in stream_proxy) ---
                annotated = frame.copy()
                for b in boxes_data:
                    x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
                    label = f"{b['class']} {b['confidence']:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    )
                worker.set_annotated_frame(annotated)

                # --- MQTT publish annotated frame info ---
                # Publish raw detection list (all boxes, before persistence filter)
                # so dashboard can show live inference even without confirmed events
                if mqtt_enabled and boxes_data:
                    self._mqtt.publish(
                        f"dhras/cameras/{cam_id}/status",
                        {
                            "online": True,
                            "detections_raw": len(boxes_data),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                # --- Detection persistence check ---
                confirmed = self._tracker.update(cam_id, set(detected_classes_conf.keys()))

                if confirmed:
                    self._scheduler.set_triggered(cam_id, True)

                for cls_name in confirmed:
                    conf = detected_classes_conf[cls_name]
                    best_box = next(
                        (b for b in boxes_data if b["class"] == cls_name), {}
                    )

                    # Compute the snapshot path now (deterministic timestamp) so the
                    # MQTT payload can include it immediately. The actual JPEG write
                    # and JSONL append are handed off to the background writer thread.
                    snapshot_path = None
                    if snapshots_enabled:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        snapshot_path = str(DETECTIONS_DIR / cam_id / f"{cls_name}_{ts}.jpg")

                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "camera_id": cam_id,
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": best_box.get("bbox"),
                        "snapshot": snapshot_path,
                    }

                    # Queue disk I/O — never block the inference loop on file writes
                    self._write_queue.put({
                        "frame": annotated.copy() if snapshot_path else None,
                        "snapshot_path": snapshot_path,
                        "event": event,
                    })

                    if mqtt_enabled:
                        self._mqtt.publish(
                            f"dhras/cameras/{cam_id}/detection",
                            {
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": best_box.get("bbox"),
                                "zone": None,           # zone mapping added in Session 3
                                "timestamp": event["timestamp"],
                                "snapshot": snapshot_path,
                                "camera_id": cam_id,
                            },
                        )
                        logger.info(
                            f"[{cam_id}] DETECTION: {cls_name} "
                            f"conf={conf:.2f}  snapshot={snapshot_path}"
                        )

                # Record inference time for FPS tracking
                self._fps_counter[cam_id].append(time.time())
                self._scheduler.mark_inferred(cam_id)

                # --- Cycle budget check ---
                elapsed_ms = (time.perf_counter() - cycle_start) * 1000
                if elapsed_ms > self.CYCLE_BUDGET_MS:
                    # Over budget: reduce input resolution for next cycle and
                    # skip remaining cameras (starvation protection handles them next cycle)
                    if self._input_scale > self.SCALE_MIN:
                        self._input_scale = max(
                            self.SCALE_MIN, self._input_scale - self.SCALE_STEP_DOWN
                        )
                        logger.warning(
                            f"Cycle budget exceeded ({elapsed_ms:.0f}ms > {self.CYCLE_BUDGET_MS}ms)"
                            f" — input scale reduced to {self._input_scale:.2f}"
                        )
                    over_budget = True
                    break  # skip remaining cameras this cycle

            # Recover resolution gradually when cycle is well within budget
            cycle_ms = (time.perf_counter() - cycle_start) * 1000
            if (
                not over_budget
                and cycle_ms < self.CYCLE_BUDGET_MS * self.SCALE_RECOVERY_THRESHOLD
                and self._input_scale < 1.0
            ):
                self._input_scale = min(1.0, self._input_scale + self.SCALE_STEP_UP)

            # Periodic status publish (throttled to avoid MQTT flooding)
            now = time.time()
            if now - last_status_publish >= status_publish_interval:
                last_status_publish = now
                self._publish_camera_status()
                logger.debug(
                    f"Cycle: {cycle_ms:.1f}ms  scale={self._input_scale:.2f}"
                    f"  active={len(active)}"
                )

    def _publish_camera_status(self):
        """Publish online/offline and achieved FPS for each camera every 5s."""
        now = time.time()
        for cam_id, worker in self._workers.items():
            # Compute achieved FPS from recent inference timestamps
            times = list(self._fps_counter[cam_id])
            if len(times) >= 2:
                window = now - times[0]
                achieved_fps = round((len(times) - 1) / window, 1) if window > 0 else 0.0
            else:
                achieved_fps = 0.0

            self._mqtt.publish(
                f"dhras/cameras/{cam_id}/status",
                {
                    "online": worker.online,
                    "paused": worker.paused,
                    "fps": achieved_fps,
                    "last_frame_age_sec": round(now - worker.last_frame_time, 1),
                    "input_scale": round(self._input_scale, 2),
                    "timestamp": datetime.now().isoformat(),
                },
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    config = load_config()
    service = VisionService(config)
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")


if __name__ == "__main__":
    main()
