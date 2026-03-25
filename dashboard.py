"""
dashboard.py — Central security camera dashboard
Jetson Orin Nano | http://192.168.1.17:8080

Usage:
    source ~/jetson_yolo_gpu/bin/activate
    cd ~/robotics/jetson-vision
    python3 dashboard.py
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_file

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "cameras_config.json"
DETECT_DIR  = BASE_DIR / "detections"
EVENTS_FILE = DETECT_DIR / "events.jsonl"

FLASK_PORT  = 8080

app = Flask(__name__)

_cfg_lock = threading.Lock()


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────
# API — CONFIG
# ─────────────────────────────────────────────

@app.route("/api/config")
def api_config():
    return jsonify(load_config())


@app.route("/api/config/<cam_id>", methods=["POST"])
def api_update_config(cam_id):
    with _cfg_lock:
        cfg = load_config()
        if cam_id not in cfg["cameras"]:
            return jsonify({"error": "unknown camera"}), 404
        data = request.get_json(force=True)
        cfg["cameras"][cam_id].update(data)
        save_config(cfg)
        cam = cfg["cameras"][cam_id]

    # Push live to camera script
    api_url = f"http://{cam['host']}:{cam['stream_port']}/config"
    try:
        r = requests.post(api_url, json=data, timeout=2)
        return jsonify({"ok": True, "camera_ack": r.json()})
    except Exception as e:
        return jsonify({"ok": True, "warn": f"Config saved but camera unreachable: {e}"})


# ─────────────────────────────────────────────
# API — STATUS
# ─────────────────────────────────────────────

@app.route("/api/cameras/status")
def api_cameras_status():
    cfg = load_config()
    statuses = {}
    for cam_id, cam in cfg["cameras"].items():
        url = f"http://{cam['host']}:{cam['stream_port']}/status"
        try:
            r = requests.get(url, timeout=1.5)
            s = r.json()
            s["online"] = True
            statuses[cam_id] = s
        except Exception:
            statuses[cam_id] = {"online": False}
    return jsonify(statuses)


# ─────────────────────────────────────────────
# API — EVENTS LOG
# ─────────────────────────────────────────────

@app.route("/api/events")
def api_events():
    limit  = int(request.args.get("limit", 100))
    cam_id = request.args.get("camera")
    events = []
    if EVENTS_FILE.exists():
        with open(EVENTS_FILE) as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if cam_id and ev.get("camera") != cam_id:
                    continue
                events.append(ev)
                if len(events) >= limit:
                    break
            except Exception:
                pass
    return jsonify(events)


@app.route("/api/events/clear", methods=["POST"])
def api_events_clear():
    if EVENTS_FILE.exists():
        EVENTS_FILE.unlink()
    return jsonify({"ok": True})


# ─────────────────────────────────────────────
# API — GALLERY
# ─────────────────────────────────────────────

@app.route("/api/gallery")
def api_gallery():
    limit  = int(request.args.get("limit", 80))
    cam_id = request.args.get("camera")

    cfg = load_config()
    images = []

    search_dirs = []
    if cam_id and cam_id in cfg["cameras"]:
        snap_dir = BASE_DIR / cfg["cameras"][cam_id]["snapshot_dir"]
        if snap_dir.exists():
            search_dirs = [snap_dir]
    else:
        search_dirs = [DETECT_DIR]

    all_files = []
    for d in search_dirs:
        all_files.extend(d.rglob("*.jpg"))

    for path in sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        rel = path.relative_to(BASE_DIR)
        images.append({
            "path": str(rel).replace("\\", "/"),
            "name": path.name,
            "camera": path.parent.name,
            "mtime": path.stat().st_mtime,
            "ts": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify(images)


@app.route("/api/gallery/clear", methods=["POST"])
def api_gallery_clear():
    cam_id = (request.get_json(force=True) or {}).get("camera")
    cfg = load_config()
    removed = 0
    if cam_id and cam_id in cfg["cameras"]:
        snap_dir = BASE_DIR / cfg["cameras"][cam_id]["snapshot_dir"]
        for f in snap_dir.glob("*.jpg"):
            f.unlink()
            removed += 1
    else:
        for f in DETECT_DIR.rglob("*.jpg"):
            f.unlink()
            removed += 1
    return jsonify({"ok": True, "removed": removed})


@app.route("/snapshots/<path:filepath>")
def serve_snapshot(filepath):
    safe = (BASE_DIR / filepath).resolve()
    if not str(safe).startswith(str(BASE_DIR.resolve())):
        return "Forbidden", 403
    if not safe.is_file():
        return "Not found", 404
    return send_file(safe, mimetype="image/jpeg")


# ─────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>Vision Hub</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0a0a0a;
  --surface:  #111;
  --border:   #222;
  --border2:  #333;
  --text:     #e0e0e0;
  --muted:    #555;
  --green:    #4cff90;
  --orange:   #ff9800;
  --blue:     #4db8ff;
  --red:      #ff4444;
  --yellow:   #ffe066;
}

html, body { height: 100%; background: var(--bg); color: var(--text);
             font-family: 'Courier New', monospace; font-size: 14px; }

/* ── Header ── */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px; border-bottom: 1px solid var(--border);
  background: var(--surface); position: sticky; top: 0; z-index: 50;
}
header h1 { font-size: 0.85rem; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
#live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
            box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Tabs ── */
nav.tabs {
  display: flex; gap: 2px; padding: 8px 16px;
  border-bottom: 1px solid var(--border); background: var(--surface);
  position: sticky; top: 41px; z-index: 49;
}
nav.tabs button {
  background: none; border: none; color: var(--muted); cursor: pointer;
  font-family: inherit; font-size: 0.75rem; letter-spacing: 0.12em;
  text-transform: uppercase; padding: 6px 14px; border-radius: 4px;
  transition: color 0.15s, background 0.15s;
}
nav.tabs button.active { color: var(--text); background: var(--border); }
nav.tabs button:hover:not(.active) { color: var(--text); }

/* ── Sections ── */
section { display: none; padding: 16px; }
section.active { display: block; }

/* ── Camera Grid ── */
#camera-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
@media (min-width: 900px) {
  #camera-grid { grid-template-columns: repeat(3, 1fr); }
}

.cam-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; overflow: hidden; position: relative;
}
.cam-card img.stream {
  width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover;
  background: #050505;
}
.cam-card .cam-overlay {
  position: absolute; top: 0; left: 0; right: 0;
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 8px;
  background: linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%);
}
.cam-card .cam-name { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: #ccc; }
.cam-card .status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex-shrink: 0;
}
.cam-card .status-dot.online { background: var(--green); box-shadow: 0 0 5px var(--green); }
.cam-card .status-dot.offline { background: var(--red); }

.cam-card .cam-footer {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 8px; background: var(--surface);
}
.cam-card .mode-badge {
  font-size: 0.65rem; letter-spacing: 0.1em; color: var(--muted);
  text-transform: uppercase;
}
.cam-card .cam-btns { display: flex; gap: 6px; }
.cam-card .cam-btns button {
  background: none; border: 1px solid var(--border2); border-radius: 4px;
  color: var(--muted); cursor: pointer; font-size: 0.9rem; padding: 3px 7px;
  transition: color 0.1s, border-color 0.1s;
}
.cam-card .cam-btns button:hover { color: var(--text); border-color: var(--text); }

/* ── Modals ── */
.modal-backdrop {
  display: none; position: fixed; inset: 0; z-index: 100;
  background: rgba(0,0,0,0.92); align-items: center; justify-content: center;
}
.modal-backdrop.open { display: flex; }

/* Full view modal */
#modal-view .modal-inner {
  width: min(96vw, 900px); background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
#modal-view .modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; border-bottom: 1px solid var(--border);
}
#modal-view .modal-header span { font-size: 0.8rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; }
#modal-view img { width: 100%; display: block; }
.modal-close {
  background: none; border: none; color: var(--muted); cursor: pointer;
  font-size: 1.2rem; line-height: 1; padding: 2px 6px;
}
.modal-close:hover { color: var(--text); }

/* PTZ controls inside full view */
#ptz-controls {
  padding: 10px; display: flex; flex-direction: column; align-items: center; gap: 6px;
}
.ptz-row { display: flex; gap: 6px; }
.ptz-btn {
  width: 52px; height: 52px; background: #161616; border: 1px solid var(--border2);
  border-radius: 6px; color: #ccc; font-size: 1.2rem; cursor: pointer;
  -webkit-tap-highlight-color: transparent; transition: background 0.1s, border-color 0.1s;
}
.ptz-btn:active, .ptz-btn.held { background: #2a2a2a; border-color: var(--green); color: var(--green); }
.ptz-btn.zoom { width: 70px; height: 40px; font-size: 0.85rem; }

/* Settings modal */
#modal-settings .modal-inner {
  width: min(96vw, 480px); max-height: 90vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
}
#modal-settings .modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  position: sticky; top: 0; background: var(--surface); z-index: 1;
}
#modal-settings .modal-header span { font-size: 0.85rem; letter-spacing: 0.15em; text-transform: uppercase; }
.settings-body { padding: 16px; display: flex; flex-direction: column; gap: 20px; }
.settings-section h3 {
  font-size: 0.7rem; letter-spacing: 0.15em; color: var(--muted);
  text-transform: uppercase; margin-bottom: 10px;
  border-bottom: 1px solid var(--border); padding-bottom: 6px;
}

/* Toggle row */
.toggle-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 0;
}
.toggle-row label { font-size: 0.8rem; color: #bbb; }
.toggle {
  position: relative; width: 40px; height: 22px; flex-shrink: 0;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
  position: absolute; inset: 0; background: var(--border2); border-radius: 11px;
  cursor: pointer; transition: background 0.2s;
}
.toggle-slider::before {
  content: ''; position: absolute; width: 16px; height: 16px; border-radius: 50%;
  background: #888; left: 3px; top: 3px; transition: transform 0.2s, background 0.2s;
}
.toggle input:checked + .toggle-slider { background: #1a4a2a; }
.toggle input:checked + .toggle-slider::before { transform: translateX(18px); background: var(--green); }

/* Slider row */
.slider-row { display: flex; flex-direction: column; gap: 6px; }
.slider-row label { font-size: 0.8rem; color: #bbb; display: flex; justify-content: space-between; }
.slider-row input[type=range] {
  width: 100%; accent-color: var(--green); cursor: pointer;
}

/* Classes grid */
.classes-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
}
.class-chip {
  display: flex; align-items: center; gap: 5px;
  padding: 5px 8px; background: var(--bg); border: 1px solid var(--border2);
  border-radius: 4px; cursor: pointer; font-size: 0.72rem; color: var(--muted);
  transition: color 0.1s, border-color 0.1s, background 0.1s;
}
.class-chip input { display: none; }
.class-chip.selected { color: var(--green); border-color: var(--green); background: #0a1a10; }

/* Save button */
#settings-save {
  width: 100%; padding: 10px; background: #0d2a16;
  border: 1px solid var(--green); border-radius: 6px;
  color: var(--green); font-family: inherit; font-size: 0.8rem;
  letter-spacing: 0.15em; text-transform: uppercase; cursor: pointer;
  transition: background 0.15s;
}
#settings-save:hover { background: #1a4a2a; }
#settings-save:disabled { opacity: 0.4; cursor: default; }

/* ── Detections ── */
#events-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px; gap: 8px; flex-wrap: wrap;
}
#events-toolbar select {
  background: var(--surface); border: 1px solid var(--border2);
  color: var(--text); font-family: inherit; font-size: 0.75rem;
  padding: 5px 8px; border-radius: 4px; cursor: pointer;
}
#events-toolbar button {
  background: none; border: 1px solid var(--border2); color: var(--muted);
  font-family: inherit; font-size: 0.72rem; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 5px 10px; border-radius: 4px; cursor: pointer;
}
#events-toolbar button:hover { color: var(--text); border-color: var(--text); }

#events-list { display: flex; flex-direction: column; gap: 4px; }
.event-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; font-size: 0.75rem;
}
.event-row .ev-cam {
  font-size: 0.65rem; padding: 2px 6px; border-radius: 3px;
  background: var(--border); color: #aaa; letter-spacing: 0.08em;
  flex-shrink: 0; text-transform: uppercase;
}
.event-row .ev-class { color: var(--green); letter-spacing: 0.05em; flex-shrink: 0; min-width: 70px; }
.event-row .ev-conf { color: var(--muted); flex-shrink: 0; min-width: 40px; }
.event-row .ev-ts { color: var(--muted); font-size: 0.68rem; margin-left: auto; flex-shrink: 0; }
.event-row .ev-thumb {
  width: 48px; height: 36px; object-fit: cover; border: 1px solid var(--border);
  border-radius: 2px; flex-shrink: 0; cursor: pointer;
}

/* ── Gallery ── */
#gallery-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px; gap: 8px; flex-wrap: wrap;
}
#gallery-toolbar select {
  background: var(--surface); border: 1px solid var(--border2);
  color: var(--text); font-family: inherit; font-size: 0.75rem;
  padding: 5px 8px; border-radius: 4px; cursor: pointer;
}
#gallery-toolbar button {
  background: none; border: 1px solid #550000; color: var(--red);
  font-family: inherit; font-size: 0.72rem; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 5px 10px; border-radius: 4px; cursor: pointer;
}
#gallery-toolbar button:hover { background: #1a0000; }

#gallery-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}
@media (min-width: 600px) {
  #gallery-grid { grid-template-columns: repeat(4, 1fr); }
}
@media (min-width: 900px) {
  #gallery-grid { grid-template-columns: repeat(6, 1fr); }
}
.gallery-item { position: relative; cursor: pointer; }
.gallery-item img {
  width: 100%; aspect-ratio: 4/3; object-fit: cover;
  border: 1px solid var(--border); border-radius: 3px; display: block;
  transition: border-color 0.1s;
}
.gallery-item:hover img { border-color: var(--border2); }
.gallery-item .gi-label {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: rgba(0,0,0,0.7); padding: 2px 4px;
  font-size: 0.6rem; color: #aaa; letter-spacing: 0.05em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Image lightbox */
#modal-img .modal-inner {
  width: min(96vw, 900px);
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
}
#modal-img img { width: 100%; display: block; }
#modal-img .modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 12px; border-bottom: 1px solid var(--border);
}
#modal-img .modal-header span { font-size: 0.72rem; color: var(--muted); }

/* ── Misc ── */
.empty-msg { color: var(--muted); font-size: 0.8rem; padding: 24px 0; text-align: center; letter-spacing: 0.1em; }
</style>
</head>
<body>

<header>
  <h1>Vision Hub</h1>
  <div id="live-dot" title="Live"></div>
</header>

<nav class="tabs">
  <button class="active" onclick="showTab('cameras')">Cameras</button>
  <button onclick="showTab('detections')">Detections</button>
  <button onclick="showTab('gallery')">Gallery</button>
</nav>

<!-- ── CAMERAS TAB ── -->
<section id="tab-cameras" class="active">
  <div id="camera-grid"></div>
</section>

<!-- ── DETECTIONS TAB ── -->
<section id="tab-detections">
  <div id="events-toolbar">
    <select id="events-cam-filter" onchange="loadEvents()">
      <option value="">All Cameras</option>
    </select>
    <button onclick="clearEvents()">Clear Log</button>
  </div>
  <div id="events-list"><p class="empty-msg">Loading...</p></div>
</section>

<!-- ── GALLERY TAB ── -->
<section id="tab-gallery">
  <div id="gallery-toolbar">
    <select id="gallery-cam-filter" onchange="loadGallery()">
      <option value="">All Cameras</option>
    </select>
    <button onclick="clearGallery()">Clear Snapshots</button>
  </div>
  <div id="gallery-grid"></div>
</section>

<!-- ── FULL VIEW MODAL ── -->
<div class="modal-backdrop" id="modal-view" onclick="closeOnBackdrop(event,'modal-view')">
  <div class="modal-inner">
    <div class="modal-header">
      <span id="modal-view-title">Camera</span>
      <button class="modal-close" onclick="closeModal('modal-view')">✕</button>
    </div>
    <img id="modal-view-stream" src="" alt="stream">
    <div id="ptz-controls" style="display:none">
      <div class="ptz-row">
        <button class="ptz-btn" data-dir="up">▲</button>
      </div>
      <div class="ptz-row">
        <button class="ptz-btn" data-dir="left">◀</button>
        <button class="ptz-btn" data-dir="stop">■</button>
        <button class="ptz-btn" data-dir="right">▶</button>
      </div>
      <div class="ptz-row">
        <button class="ptz-btn" data-dir="down">▼</button>
      </div>
      <div class="ptz-row" style="margin-top:4px">
        <button class="ptz-btn zoom" data-dir="zoomin">＋ Zoom</button>
        <button class="ptz-btn zoom" data-dir="zoomout">－ Zoom</button>
      </div>
    </div>
  </div>
</div>

<!-- ── SETTINGS MODAL ── -->
<div class="modal-backdrop" id="modal-settings" onclick="closeOnBackdrop(event,'modal-settings')">
  <div class="modal-inner">
    <div class="modal-header">
      <span id="settings-title">Settings</span>
      <button class="modal-close" onclick="closeModal('modal-settings')">✕</button>
    </div>
    <div class="settings-body">

      <div class="settings-section">
        <h3>Detection Classes</h3>
        <div class="classes-grid" id="settings-classes"></div>
      </div>

      <div class="settings-section">
        <h3>Confidence Threshold</h3>
        <div class="slider-row">
          <label>
            <span>Minimum confidence</span>
            <span id="conf-val">0.50</span>
          </label>
          <input type="range" id="settings-conf" min="5" max="95" step="5"
                 oninput="document.getElementById('conf-val').textContent=(this.value/100).toFixed(2)">
        </div>
      </div>

      <div class="settings-section">
        <h3>Behaviour</h3>
        <div class="toggle-row">
          <label>Snapshots</label>
          <label class="toggle">
            <input type="checkbox" id="settings-snapshots">
            <span class="toggle-slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>MQTT Triggers</label>
          <label class="toggle">
            <input type="checkbox" id="settings-mqtt">
            <span class="toggle-slider"></span>
          </label>
        </div>
        <div class="toggle-row">
          <label>Monitor Only (no PTZ)</label>
          <label class="toggle">
            <input type="checkbox" id="settings-monitor">
            <span class="toggle-slider"></span>
          </label>
        </div>
        <div class="toggle-row" id="row-tracking">
          <label>Auto Tracking</label>
          <label class="toggle">
            <input type="checkbox" id="settings-tracking">
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <button id="settings-save" onclick="saveSettings()">Save &amp; Apply</button>
      <div id="settings-msg" style="font-size:0.75rem;color:var(--muted);text-align:center;min-height:1.2em"></div>
    </div>
  </div>
</div>

<!-- ── IMAGE LIGHTBOX ── -->
<div class="modal-backdrop" id="modal-img" onclick="closeOnBackdrop(event,'modal-img')">
  <div class="modal-inner">
    <div class="modal-header">
      <span id="modal-img-title"></span>
      <button class="modal-close" onclick="closeModal('modal-img')">✕</button>
    </div>
    <img id="modal-img-src" src="" alt="">
  </div>
</div>

<script>
// ─── State ──────────────────────────────────
let config = {};
let statuses = {};
let currentSettingsCam = null;
let ptzApiBase = null;
let ptzHoldInterval = null;

const ALL_CLASSES = [
  'person','car','truck','bus','motorcycle','bicycle',
  'dog','cat','bird','horse','bear','cow','sheep','backpack','cell phone'
];

// ─── Tabs ────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'detections') loadEvents();
  if (name === 'gallery')    loadGallery();
}

// ─── Modals ──────────────────────────────────
function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
  if (id === 'modal-view') {
    document.getElementById('modal-view-stream').src = '';
    stopPtzHold();
  }
}
function closeOnBackdrop(e, id) {
  if (e.target === document.getElementById(id)) closeModal(id);
}

// ─── Load Config & Build Grid ─────────────────
async function init() {
  try {
    const r = await fetch('/api/config');
    config = await r.json();
    buildGrid();
    populateCameraFilters();
  } catch(e) { console.error('Config load failed', e); }
  pollStatus();
}

function buildGrid() {
  const grid = document.getElementById('camera-grid');
  grid.innerHTML = '';
  for (const [id, cam] of Object.entries(config.cameras)) {
    const streamUrl = `http://192.168.1.17:8082/stream/${id}`;
    const card = document.createElement('div');
    card.className = 'cam-card';
    card.dataset.camId = id;
    card.innerHTML = `
      <img class="stream" src="${streamUrl}" alt="${cam.name}">
      <div class="cam-overlay">
        <span class="cam-name">${cam.name}</span>
        <span class="status-dot" id="dot-${id}"></span>
      </div>
      <div class="cam-footer">
        <span class="mode-badge" id="mode-${id}">–</span>
        <div class="cam-btns">
          <button title="Full view" onclick="openFullView('${id}')">⤢</button>
          <button title="Settings"  onclick="openSettings('${id}')">⚙</button>
        </div>
      </div>`;
    grid.appendChild(card);
  }
}

function populateCameraFilters() {
  const selectors = ['#events-cam-filter', '#gallery-cam-filter'];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    for (const [id, cam] of Object.entries(config.cameras)) {
      const opt = document.createElement('option');
      opt.value = id; opt.textContent = cam.name;
      el.appendChild(opt);
    }
  }
}

// ─── Status Polling ───────────────────────────
async function pollStatus() {
  try {
    const r = await fetch('/api/cameras/status');
    statuses = await r.json();
    for (const [id, s] of Object.entries(statuses)) {
      const dot  = document.getElementById('dot-' + id);
      const mode = document.getElementById('mode-' + id);
      if (!dot) continue;
      dot.className = 'status-dot ' + (s.online ? 'online' : 'offline');
      if (mode) mode.textContent = s.online ? (s.mode || 'online') : 'offline';
    }
  } catch(e) {}
  setTimeout(pollStatus, 5000);
}

// ─── Full View ────────────────────────────────
function openFullView(camId) {
  const cam = config.cameras[camId];
  document.getElementById('modal-view-title').textContent = cam.name;
  document.getElementById('modal-view-stream').src = `http://192.168.1.17:8082/stream/${camId}`;

  const ptzDiv = document.getElementById('ptz-controls');
  if (cam.type === 'ptz') {
    ptzApiBase = `http://192.168.1.17:8082/ptz/${camId}`;
    ptzDiv.style.display = 'flex';
    bindPtzButtons();
  } else {
    ptzApiBase = null;
    ptzDiv.style.display = 'none';
  }
  openModal('modal-view');
}

function bindPtzButtons() {
  document.querySelectorAll('#ptz-controls .ptz-btn').forEach(btn => {
    btn.onmousedown  = btn.ontouchstart = (e) => { e.preventDefault(); startPtz(btn.dataset.dir); btn.classList.add('held'); };
    btn.onmouseup    = btn.ontouchend   = (e) => { e.preventDefault(); stopPtz(btn.dataset.dir); btn.classList.remove('held'); };
    btn.onmouseleave = () => { stopPtz(btn.dataset.dir); btn.classList.remove('held'); };
  });
}

function startPtz(dir) {
  if (!ptzApiBase) return;
  fetch(ptzApiBase + '/ptz', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({dir, action: 'start'})
  }).catch(()=>{});
}
function stopPtz(dir) {
  if (!ptzApiBase || dir === 'stop') return;
  fetch(ptzApiBase + '/ptz', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({dir, action: 'stop'})
  }).catch(()=>{});
}
function stopPtzHold() {
  if (ptzHoldInterval) { clearInterval(ptzHoldInterval); ptzHoldInterval = null; }
}

// ─── Settings ────────────────────────────────
function openSettings(camId) {
  currentSettingsCam = camId;
  const cam = config.cameras[camId];
  document.getElementById('settings-title').textContent = cam.name + ' — Settings';

  // Classes
  const grid = document.getElementById('settings-classes');
  grid.innerHTML = '';
  for (const cls of ALL_CLASSES) {
    const selected = (cam.watch_classes || []).includes(cls);
    const chip = document.createElement('div');
    chip.className = 'class-chip' + (selected ? ' selected' : '');
    chip.dataset.cls = cls;
    chip.textContent = cls;
    chip.onclick = () => { chip.classList.toggle('selected'); };
    grid.appendChild(chip);
  }

  // Confidence
  const confEl = document.getElementById('settings-conf');
  const confVal = Math.round((cam.confidence || 0.5) * 100);
  confEl.value = confVal;
  document.getElementById('conf-val').textContent = (confVal / 100).toFixed(2);

  // Toggles
  document.getElementById('settings-snapshots').checked = !!cam.snapshots;
  document.getElementById('settings-mqtt').checked      = !!cam.mqtt_enabled;
  document.getElementById('settings-monitor').checked   = !!cam.monitor_only;
  document.getElementById('settings-tracking').checked  = !!cam.tracking;
  document.getElementById('row-tracking').style.display = cam.type === 'ptz' ? 'flex' : 'none';

  document.getElementById('settings-msg').textContent = '';
  document.getElementById('settings-save').disabled = false;
  openModal('modal-settings');
}

async function saveSettings() {
  const btn = document.getElementById('settings-save');
  const msg = document.getElementById('settings-msg');
  btn.disabled = true;
  msg.textContent = 'Saving...';

  const classes = [...document.querySelectorAll('#settings-classes .class-chip.selected')]
                    .map(c => c.dataset.cls);
  const data = {
    watch_classes: classes,
    confidence:    parseFloat(document.getElementById('settings-conf').value) / 100,
    snapshots:     document.getElementById('settings-snapshots').checked,
    mqtt_enabled:  document.getElementById('settings-mqtt').checked,
    monitor_only:  document.getElementById('settings-monitor').checked,
    tracking:      document.getElementById('settings-tracking').checked,
  };

  try {
    const r = await fetch('/api/config/' + currentSettingsCam, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    const result = await r.json();
    config.cameras[currentSettingsCam] = Object.assign(config.cameras[currentSettingsCam], data);
    msg.style.color = 'var(--green)';
    msg.textContent = result.warn ? ('Saved (warn: ' + result.warn + ')') : 'Applied.';
  } catch(e) {
    msg.style.color = 'var(--red)';
    msg.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}

// ─── Detections ──────────────────────────────
async function loadEvents() {
  const camId = document.getElementById('events-cam-filter').value;
  const url = '/api/events?limit=100' + (camId ? '&camera=' + camId : '');
  const list = document.getElementById('events-list');
  try {
    const r = await fetch(url);
    const events = await r.json();
    if (!events.length) { list.innerHTML = '<p class="empty-msg">No detections yet.</p>'; return; }
    list.innerHTML = events.map(ev => {
      const thumb = ev.image
        ? `<img class="ev-thumb" src="/snapshots/${camDir(ev.camera)}/${ev.image}"
               onerror="this.style.display='none'"
               onclick="openImgModal(this.src,'${ev.image}')">`
        : '';
      return `<div class="event-row">
        ${thumb}
        <span class="ev-cam">${ev.camera || '?'}</span>
        <span class="ev-class">${ev.class || ''}</span>
        <span class="ev-conf">${ev.confidence ? (ev.confidence*100).toFixed(0)+'%' : ''}</span>
        <span class="ev-ts">${ev.timestamp ? ev.timestamp.replace('T',' ').slice(0,19) : ''}</span>
      </div>`;
    }).join('');
  } catch(e) { list.innerHTML = '<p class="empty-msg">Error loading events.</p>'; }
}

function camDir(camId) {
  if (!config.cameras || !config.cameras[camId]) return camId || '';
  const sd = config.cameras[camId].snapshot_dir || ('detections/' + camId);
  return sd.replace(/^detections\//, '');
}

async function clearEvents() {
  if (!confirm('Clear all detection events?')) return;
  await fetch('/api/events/clear', {method:'POST'});
  loadEvents();
}

// ─── Gallery ─────────────────────────────────
async function loadGallery() {
  const camId = document.getElementById('gallery-cam-filter').value;
  const url = '/api/gallery?limit=80' + (camId ? '&camera=' + camId : '');
  const grid = document.getElementById('gallery-grid');
  try {
    const r = await fetch(url);
    const images = await r.json();
    if (!images.length) { grid.innerHTML = '<p class="empty-msg">No snapshots yet.</p>'; return; }
    grid.innerHTML = images.map(img => `
      <div class="gallery-item" onclick="openImgModal('/snapshots/${img.path}','${img.name}')">
        <img src="/snapshots/${img.path}" alt="${img.name}" loading="lazy">
        <div class="gi-label">${img.name}</div>
      </div>`).join('');
  } catch(e) { grid.innerHTML = '<p class="empty-msg">Error loading gallery.</p>'; }
}

async function clearGallery() {
  const camId = document.getElementById('gallery-cam-filter').value;
  const label = camId ? (config.cameras[camId]?.name || camId) : 'ALL cameras';
  if (!confirm('Delete all snapshots for ' + label + '?')) return;
  await fetch('/api/gallery/clear', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({camera: camId || null})
  });
  loadGallery();
}

// ─── Image lightbox ───────────────────────────
function openImgModal(src, name) {
  document.getElementById('modal-img-src').src = src;
  document.getElementById('modal-img-title').textContent = name;
  openModal('modal-img');
}

// ─── Boot ────────────────────────────────────
init();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DETECT_DIR.mkdir(exist_ok=True)
    print(f"Dashboard → http://192.168.1.17:{FLASK_PORT}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
