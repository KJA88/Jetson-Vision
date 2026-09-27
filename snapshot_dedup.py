"""
snapshot_dedup.py
Jetson Orin Nano | ~/robotics/jetson-vision/

Persistence / dedup rule for fixed-camera vehicle snapshots ("parked-car fix").

Problem it solves: a parked vehicle whose YOLO box jitters can trip
VehicleTracker.is_moving(), and the per-label cooldown then lets it be
re-saved about once per cooldown period, all day.

Rule (per camera, per class):
  * observe() every vehicle box of a dedup class, every frame, BEFORE the
    is_moving() filter, so that parked vehicles keep their object "alive".
  * should_save() is asked only for boxes that would otherwise be saved.
    It says yes when:
      - "new"          : first sighting of this object (never saved yet),
                         once it has been seen on new_confirm_frames
                         observations without a gap > CONFIRM_GAP_SEC
                         (filters one-off phantom boxes)
      - "moved"        : IoU(current box, saved box) < move_iou, or the
                         center moved > move_center_frac * saved-box
                         diagonal, for move_confirm_frames consecutive
                         observations (filters single-frame jitter),
                         at most once per move_min_interval_sec. After a
                         "moved" save the object is "in motion": its
                         reference box follows it and no further "moved"
                         saves happen until it has settled again (box
                         stable for move_confirm_frames observations), so
                         a re-park gives one save and a car driving off
                         gives one save (at the start of the move).
      - "max_interval" : object has been saved before and at least
                         max_interval_sec have passed (0 disables)
    Otherwise it returns None (suppress).
  * An object not observed for more than absent_sec is forgotten, so a
    vehicle that leaves and comes back is a new sighting again.
  * mark_saved() commits a save (call it only if the snapshot really
    happened, i.e. after the cooldown backstop also allowed it).

Pure stdlib, no cv2/YOLO/Flask imports, so it can be unit-tested anywhere.
The inference loop is single-threaded, but a lock is kept so the class is
safe if it is ever called from more than one thread.
"""

import math
import threading
from typing import Dict, Optional, Tuple

Box = Tuple[float, float, float, float]

DEFAULT_DEDUP_CLASSES = ("car", "truck", "bus", "motorcycle")

DEDUP_DEFAULTS = {
    "enabled":               False,   # off unless config turns it on
    "classes":               frozenset(DEFAULT_DEDUP_CLASSES),
    "match_iou":             0.3,     # associate box -> existing object
    "move_iou":              0.5,     # below this vs saved box = moved
    "move_center_frac":      0.25,    # center shift > frac * diag = moved
    "move_confirm_frames":   3,       # consecutive "moved" observations
    "move_min_interval_sec": 30.0,    # min gap between saves for "moved"
    "new_confirm_frames":    3,       # observations before a new object may save
    "absent_sec":            45.0,    # forget object after this long unseen
    "max_interval_sec":      1800.0,  # re-save a still object after this (0=off)
    "backstop_cooldown_sec": 5.0,     # per-label cooldown used for dedup classes
    "max_objects":           64,      # hard cap on remembered objects
}


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def box_center(b: Box) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def box_diag(b: Box) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def center_dist(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    return math.hypot(ax - bx, ay - by)


# ─────────────────────────────────────────────
# Config parsing (hot-reload friendly, never raises)
# ─────────────────────────────────────────────

def _num(raw, key, default, lo=None, hi=None, cast=float):
    try:
        val = cast(raw.get(key, default))
    except (TypeError, ValueError):
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    if lo is not None and val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


def parse_dedup_cfg(raw) -> dict:
    """Turn a cameras.<cam>.dedup block into a validated dict.

    Missing block / non-dict / missing "enabled" -> enabled False.
    Invalid individual values fall back to their defaults.
    """
    cfg = dict(DEDUP_DEFAULTS)
    if not isinstance(raw, dict):
        return cfg
    cfg["enabled"] = raw.get("enabled") is True
    classes = raw.get("classes")
    if isinstance(classes, (list, tuple)) and all(isinstance(c, str) for c in classes):
        cfg["classes"] = frozenset(classes)
    cfg["match_iou"]             = _num(raw, "match_iou", DEDUP_DEFAULTS["match_iou"], 0.01, 1.0)
    cfg["move_iou"]              = _num(raw, "move_iou", DEDUP_DEFAULTS["move_iou"], 0.01, 1.0)
    cfg["move_center_frac"]      = _num(raw, "move_center_frac", DEDUP_DEFAULTS["move_center_frac"], 0.0)
    cfg["move_confirm_frames"]   = _num(raw, "move_confirm_frames", DEDUP_DEFAULTS["move_confirm_frames"], 1, 100, int)
    cfg["move_min_interval_sec"] = _num(raw, "move_min_interval_sec", DEDUP_DEFAULTS["move_min_interval_sec"], 0.0)
    cfg["new_confirm_frames"]    = _num(raw, "new_confirm_frames", DEDUP_DEFAULTS["new_confirm_frames"], 1, 100, int)
    cfg["absent_sec"]            = _num(raw, "absent_sec", DEDUP_DEFAULTS["absent_sec"], 0.1)
    cfg["max_interval_sec"]      = _num(raw, "max_interval_sec", DEDUP_DEFAULTS["max_interval_sec"], 0.0)
    cfg["backstop_cooldown_sec"] = _num(raw, "backstop_cooldown_sec", DEDUP_DEFAULTS["backstop_cooldown_sec"], 0.0)
    cfg["max_objects"]           = _num(raw, "max_objects", DEDUP_DEFAULTS["max_objects"], 1, 10000, int)
    return cfg


# ─────────────────────────────────────────────
# Deduper
# ─────────────────────────────────────────────

class _Obj:
    __slots__ = ("oid", "label", "last_box", "last_seen", "claimed_at",
                 "saved_box", "last_saved", "moved_streak", "saves", "hits",
                 "in_motion", "settle_anchor", "settle_streak")

    def __init__(self, oid: int, label: str, box: Box, now: float):
        self.oid          = oid
        self.label        = label
        self.last_box     = box
        self.last_seen    = now
        self.claimed_at   = None     # `now` of the frame that last matched it
        self.saved_box    = None     # box at the last committed save
        self.last_saved   = None     # time of the last committed save
        self.moved_streak = 0
        self.saves        = 0
        self.hits         = 0        # recent observation streak (for "new")
        self.in_motion    = False    # set by a "moved" save until it settles
        self.settle_anchor = None
        self.settle_streak = 0


class SnapshotDeduper:
    DUP_BOX_IOU     = 0.7   # same-frame duplicate box -> same object
    CONFIRM_GAP_SEC = 2.0   # a gap longer than this restarts the "new" streak

    def __init__(self, cfg: Optional[dict] = None, **overrides):
        self._lock    = threading.RLock()
        self._objs: Dict[int, _Obj] = {}
        self._next_id = 0
        base = dict(DEDUP_DEFAULTS)
        if cfg:
            base.update(cfg)
        base.update(overrides)
        if not isinstance(base["classes"], frozenset):
            base["classes"] = frozenset(base["classes"])
        self.cfg = base

    # ── configuration / state ────────────────
    def configure(self, cfg: dict):
        """Apply a parsed config (parse_dedup_cfg). Keeps tracked objects."""
        with self._lock:
            new = dict(self.cfg)
            new.update(cfg)
            if not isinstance(new["classes"], frozenset):
                new["classes"] = frozenset(new["classes"])
            self.cfg = new
            # Drop objects of classes no longer handled.
            for oid in [o.oid for o in self._objs.values()
                        if o.label not in new["classes"]]:
                del self._objs[oid]

    def reset(self):
        with self._lock:
            self._objs.clear()

    def handles(self, label: str) -> bool:
        return label in self.cfg["classes"]

    def __len__(self):
        return len(self._objs)

    def prune(self, now: float):
        with self._lock:
            self._prune(now)

    def _prune(self, now: float):
        absent = self.cfg["absent_sec"]
        for oid in [o.oid for o in self._objs.values() if now - o.last_seen > absent]:
            del self._objs[oid]
        cap = self.cfg["max_objects"]
        if len(self._objs) > cap:
            for o in sorted(self._objs.values(), key=lambda o: o.last_seen)[:len(self._objs) - cap]:
                del self._objs[o.oid]

    # ── main API ─────────────────────────────
    def observe(self, label: str, box, now: float) -> Optional[int]:
        """Record one detection. Returns an object id, or None if the class
        is not handled. Call for every box of a handled class, every frame."""
        if label not in self.cfg["classes"]:
            return None
        box = tuple(float(v) for v in box)
        with self._lock:
            self._prune(now)
            obj = self._match(label, box, now)
            if obj is None:
                self._next_id += 1
                obj = _Obj(self._next_id, label, box, now)
                self._objs[obj.oid] = obj
                if len(self._objs) > self.cfg["max_objects"]:
                    self._prune(now)
            if now - obj.last_seen > self.CONFIRM_GAP_SEC:
                obj.hits = 0
            if obj.claimed_at != now:
                obj.hits += 1
            obj.last_box   = box
            obj.last_seen  = now
            obj.claimed_at = now
            if obj.in_motion:
                self._track_settling(obj, box)
            elif obj.saved_box is not None and self._is_moved(box, obj.saved_box):
                obj.moved_streak += 1
            else:
                obj.moved_streak = 0
            return obj.oid

    def _track_settling(self, obj: _Obj, box: Box):
        obj.moved_streak = 0
        obj.saved_box = box                    # reference follows the vehicle
        if obj.settle_anchor is None or self._is_moved(box, obj.settle_anchor):
            obj.settle_anchor = box
            obj.settle_streak = 0
        else:
            obj.settle_streak += 1
        if obj.settle_streak >= self.cfg["move_confirm_frames"]:
            obj.in_motion = False
            obj.settle_anchor = None
            obj.settle_streak = 0

    @staticmethod
    def _anchors(o: _Obj):
        # Match against the last box AND the saved box, so one outlier frame
        # can't drag a parked object's identity away from its parking spot.
        return (o.last_box,) if o.saved_box is None else (o.last_box, o.saved_box)

    def _match(self, label: str, box: Box, now: float) -> Optional[_Obj]:
        best, best_iou = None, 0.0
        for o in self._objs.values():
            if o.label != label:
                continue
            s = max(box_iou(box, a) for a in self._anchors(o))
            if o.claimed_at == now:
                # Already matched in this frame: only accept an obvious
                # duplicate box for the same vehicle.
                if s >= self.DUP_BOX_IOU and s > best_iou:
                    best, best_iou = o, s
                continue
            if s > best_iou:
                best, best_iou = o, s
        if best is not None and best_iou >= self.cfg["match_iou"]:
            return best
        # Center-distance fallback for small / fast boxes.
        best, best_d = None, float("inf")
        for o in self._objs.values():
            if o.label != label or o.claimed_at == now:
                continue
            for a in self._anchors(o):
                d = center_dist(box, a)
                if d <= 0.5 * box_diag(a) and d < best_d:
                    best, best_d = o, d
        return best

    def _is_moved(self, box: Box, saved: Box) -> bool:
        if box_iou(box, saved) < self.cfg["move_iou"]:
            return True
        return center_dist(box, saved) > self.cfg["move_center_frac"] * box_diag(saved)

    def should_save(self, oid: Optional[int], now: float) -> Optional[str]:
        """Decision only (no state change). Returns "new", "moved",
        "max_interval", or None to suppress."""
        with self._lock:
            o = self._objs.get(oid)
            if o is None:
                return None
            if o.last_saved is None:
                return "new" if o.hits >= self.cfg["new_confirm_frames"] else None
            if o.in_motion:
                return None
            if (o.moved_streak >= self.cfg["move_confirm_frames"]
                    and now - o.last_saved >= self.cfg["move_min_interval_sec"]):
                return "moved"
            mi = self.cfg["max_interval_sec"]
            if mi > 0 and now - o.last_saved >= mi:
                return "max_interval"
            return None

    def mark_saved(self, oid: Optional[int], now: float, reason: Optional[str] = None):
        """Commit a save that actually happened (pass the should_save reason)."""
        with self._lock:
            o = self._objs.get(oid)
            if o is None:
                return
            o.last_saved   = now
            o.saved_box    = o.last_box
            o.moved_streak = 0
            o.saves       += 1
            if reason == "moved":
                o.in_motion     = True
                o.settle_anchor = None
                o.settle_streak = 0
