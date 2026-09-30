"""
Integration check of the patched vision_service.process_frame against the
unpatched main baseline, with cv2/flask/onvif/ultralytics/paho replaced by
in-memory stubs (nothing touches a camera, network, GPU, or the Jetson).
Simulated clock.

The baseline is the unpatched vision service from 247460d (main before
parked-car dedup). Current main already contains that dedup, so origin/main
is no longer the unpatched file. HOME is a temp dir so import side effects
(logs, events) stay off real paths.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── stubs ──────────────────────────────────────────────────────────────
def _install_stubs():
    cv2 = types.ModuleType("cv2")
    for n in ("rectangle", "putText", "circle", "line", "imwrite"):
        setattr(cv2, n, lambda *a, **k: None)
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.CAP_FFMPEG = 0
    cv2.IMWRITE_JPEG_QUALITY = 1
    cv2.imencode = lambda *a, **k: (True, b"")
    cv2.VideoCapture = object
    sys.modules["cv2"] = cv2

    flask = types.ModuleType("flask")

    class Flask:
        def __init__(self, *a, **k): pass
        def route(self, *a, **k): return lambda f: f
        def run(self, *a, **k): pass
    flask.Flask = Flask
    flask.Response = lambda *a, **k: None
    flask.jsonify = lambda *a, **k: None
    sys.modules["flask"] = flask
    fc = types.ModuleType("flask_cors"); fc.CORS = lambda *a, **k: None
    sys.modules["flask_cors"] = fc
    onvif = types.ModuleType("onvif"); onvif.ONVIFCamera = object
    sys.modules["onvif"] = onvif
    ul = types.ModuleType("ultralytics"); ul.YOLO = object
    sys.modules["ultralytics"] = ul

    paho = types.ModuleType("paho"); pm = types.ModuleType("paho.mqtt")
    pmc = types.ModuleType("paho.mqtt.client")

    class Client:
        def __init__(self, *a, **k): pass
        def connect(self, *a, **k): raise OSError("stub: no network")
        def loop_start(self): pass
        def loop_stop(self): pass
        def publish(self, *a, **k): pass
    pmc.Client = Client
    pmc.CallbackAPIVersion = types.SimpleNamespace(VERSION2=2)
    sys.modules.update({"paho": paho, "paho.mqtt": pm, "paho.mqtt.client": pmc})



def _load(name, path, extra_sys_path=None):
    _install_stubs()
    # vision_service writes logs/events under ~/robotics/... at import.
    os.environ["HOME"] = tempfile.mkdtemp(prefix="dedup_home_")
    inserted = False
    if extra_sys_path:
        sys.path.insert(0, extra_sys_path)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if inserted:
            sys.path.pop(0)
    return mod


def _baseline_vision_service():
    """Unpatched vision_service.py from origin/main, isolated in a temp dir."""
    dest_dir = tempfile.mkdtemp(prefix="dedup_baseline_")
    dest = os.path.join(dest_dir, "vision_service.py")
    data = subprocess.check_output(
        ["git", "show", "247460d732df01da9e62bfc002383986d69a9ba1:vision_service.py"],
        cwd=ROOT,
    )
    with open(dest, "wb") as f:
        f.write(data)
    return dest


class Clock:
    def __init__(self): self.t = 1_000_000.0
    def time(self): return self.t
    def sleep(self, s): self.t += s


class FakeFrame:
    shape = (360, 640, 3)
    def copy(self): return self


class Box:
    def __init__(self, cls_id, xyxy, conf):
        self.cls = [cls_id]; self.xyxy = [xyxy]; self.conf = [conf]


class FakeModel:
    names = {2: "car", 0: "person"}
    def __init__(self): self.next_boxes = []
    def __call__(self, frame, conf=0.05, verbose=False):
        return [types.SimpleNamespace(boxes=[Box(c, b, s) for c, b, s in self.next_boxes])]


def run_scenario(mod, cam_cfg_extra, boxes_at, seconds, fps=2.0):
    clock = Clock()
    mod.time = clock                      # cooldown_ok uses time.time()
    saves = []
    mod._trigger_action = lambda cam, label, conf, frame, snap, mq: saves.append((round(clock.t - 1_000_000.0, 1), label))
    cam = {"type": "fixed", "snapshots": True, "mqtt_enabled": False,
           "monitor_only": False, "tracking": False,
           "watch_classes": {"car": 0.3, "person": 0.5}}
    cam.update(cam_cfg_extra)
    mod._config = {"cooldown_sec": 60, "cameras": {"frontyard": cam}}
    proc = mod.CameraProcessor("frontyard")
    model = FakeModel()
    frame = FakeFrame()
    steps = int(seconds * fps)
    for i in range(steps):
        rel = i / fps
        model.next_boxes = boxes_at(i, rel)
        mod.process_frame(proc, frame, model, clock.t)
        clock.t += 1.0 / fps
    return saves


PARKED = (100, 200, 300, 320)


def parked_with_jitter_bursts(i, rel):
    """Parked car; every 30 s a 3-frame +-30 px wobble trips is_moving()."""
    k = int(rel * 2) % 60
    dx = 30 if k in (0, 2) else 0
    b = (PARKED[0] + dx, PARKED[1], PARKED[2] + dx, PARKED[3])
    return [(2, b, 0.8)]


def with_passing_car(i, rel):
    boxes = parked_with_jitter_bursts(i, rel)
    # a car crosses the street at t = 600..608 s and 1210..1218 s
    for t0 in (600, 1210):
        if t0 <= rel < t0 + 8:
            x = (rel - t0) * 60
            boxes.append((2, (int(x), 40, int(x) + 150, 120), 0.8))
    return boxes



ORIG = _load("vs_original", _baseline_vision_service())
PATCH = _load(
    "vs_patched",
    os.path.join(ROOT, "vision_service.py"),
    extra_sys_path=ROOT,
)
_orig_src = open(ORIG.__file__, encoding="utf-8").read()
_patch_src = open(PATCH.__file__, encoding="utf-8").read()
assert "from snapshot_dedup import" not in _orig_src
assert "from snapshot_dedup import" in _patch_src
assert os.path.abspath(ORIG.__file__) != os.path.abspath(PATCH.__file__)
REAL_TRIGGER = PATCH._trigger_action


def test_disabled_by_default_uses_the_live_60s_cooldown():
    """Dedup stays off unless enabled. The normal path uses cooldown_sec 60, not main's old 3s constant."""
    for scen in (parked_with_jitter_bursts, with_passing_car):
        main_saves = run_scenario(ORIG, {}, scen, 3600)
        live = run_scenario(PATCH, {}, scen, 3600)
        disabled = run_scenario(PATCH, {"dedup": {"classes": ["car"]}}, scen, 3600)
        assert live == disabled
        assert len(main_saves) > len(live) > 0
        gaps = [b[0] - a[0] for a, b in zip(live, live[1:])]
        assert gaps and min(gaps) >= 60


def test_explicit_backstop_overrides_the_60s_cooldown():
    clock = Clock()
    PATCH.time = clock
    PATCH._config = {"cooldown_sec": 60, "cameras": {}}
    proc = PATCH.CameraProcessor("frontyard")
    assert proc.cooldown_ok("person") is True
    clock.t += 10
    assert proc.cooldown_ok("person") is False
    assert proc.cooldown_ok("car", 5) is True


def test_enabled_parked_car_saved_once_then_every_30_min():
    saves = run_scenario(PATCH, {"dedup": {"enabled": True}}, parked_with_jitter_bursts, 2 * 3600)
    orig = run_scenario(ORIG, {}, parked_with_jitter_bursts, 2 * 3600)
    assert len(orig) >= 100
    assert 3 <= len(saves) <= 5          # first + ~every 30 min (only when is_moving fires)
    gaps = [b[0] - a[0] for a, b in zip(saves, saves[1:])]
    assert all(g >= 1800 for g in gaps)


def test_enabled_passing_cars_still_saved():
    saves = run_scenario(PATCH, {"dedup": {"enabled": True}}, with_passing_car, 1800)
    times = [t for t, _ in saves]
    assert any(600 <= t < 610 for t in times)
    assert any(1210 <= t < 1220 for t in times)


def test_ptz_camera_never_deduped():
    PATCH.time = Clock()
    proc = PATCH.CameraProcessor("backyard")
    proc.ptz = object()
    PATCH._config = {"cameras": {"backyard": {"dedup": {"enabled": True}}}}
    proc.refresh_dedup(PATCH.cam_cfg("backyard", "dedup"))
    assert proc.dedup_cfg["enabled"] is True and proc.dedup_active() is False


def test_trigger_action_stores_a_safe_relative_snapshot_key():
    import json
    from media_store import safe_media_path

    old = '{"camera":"frontyard","class":"car","image":"car_20260101_000000.jpg"}\n'
    os.makedirs(os.path.dirname(PATCH.EVENTS_FILE), exist_ok=True)
    with open(PATCH.EVENTS_FILE, "w", encoding="utf-8") as handle:
        handle.write(old)
    PATCH._config = {
        "cameras": {"frontyard": {"snapshot_dir": "detections/frontyard"}},
    }
    REAL_TRIGGER("frontyard", "person", 0.81, FakeFrame(), True, False)
    REAL_TRIGGER("frontyard", "person", 0.42, FakeFrame(), False, False)
    with open(PATCH.EVENTS_FILE, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    assert lines[0] == old.strip()
    attached = json.loads(lines[1])
    missing = json.loads(lines[2])
    image = attached["image"]
    assert image.startswith("detections/frontyard/person_")
    assert image.endswith(".jpg")
    assert safe_media_path(image) == image
    assert image != os.path.basename(image)
    assert "\\" not in image and not image.startswith("/")
    assert missing["image"] is None


def test_hot_reload_toggle_resets_state():
    PATCH.time = Clock()
    proc = PATCH.CameraProcessor("frontyard")
    proc.refresh_dedup({"enabled": True})
    proc.deduper.observe("car", PARKED, 0.0)
    assert len(proc.deduper) == 1 and proc.dedup_active()
    proc.refresh_dedup({"enabled": False})
    assert len(proc.deduper) == 0 and not proc.dedup_active()
    proc.refresh_dedup(None)
    assert not proc.dedup_active()
