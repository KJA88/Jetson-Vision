"""
Standalone tests for snapshot_dedup.SnapshotDeduper (pure class, no cv2/YOLO).

Simulation model (worst case for the parked-car bug): every frame, every
vehicle box is treated as a save candidate, as if VehicleTracker.is_moving()
had fired on it. The per-label backstop gate mirrors CameraProcessor.cooldown_ok.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snapshot_dedup import SnapshotDeduper, parse_dedup_cfg, DEDUP_DEFAULTS  # noqa: E402

FPS = 2.0             # effective per-camera inference rate assumed
DT = 1.0 / FPS
PARKED = (100, 200, 300, 320)    # 200x120 box in a 640x360 frame


class Pipeline:
    """Mimics the patched fixed-camera branch: observe -> should_save ->
    backstop cooldown -> mark_saved."""

    def __init__(self, backstop=DEDUP_DEFAULTS["backstop_cooldown_sec"], **cfg):
        self.d = SnapshotDeduper(parse_dedup_cfg({"enabled": True, **cfg}))
        self.backstop = backstop
        self.last_trigger = {}
        self.saves = []   # (t, label, reason)

    def cooldown_ok(self, label, now):
        if now - self.last_trigger.get(label, -1e9) >= self.backstop:
            self.last_trigger[label] = now
            return True
        return False

    def frame(self, now, boxes):
        ids = [(lbl, self.d.observe(lbl, b, now)) for lbl, b in boxes]
        for lbl, oid in ids:
            reason = self.d.should_save(oid, now)
            if reason and self.cooldown_ok(lbl, now):
                self.d.mark_saved(oid, now, reason)
                self.saves.append((now, lbl, reason))


def jitter(box, rng, px=8):
    return tuple(v + rng.randint(-px, px) for v in box)


def run(p, t0, t1, boxes_fn):
    t = t0
    while t < t1:
        p.frame(t, boxes_fn(t))
        t += DT
    return t


def test_parked_car_with_jitter_saves_once_then_again_after_30_min():
    rng = random.Random(1)
    p = Pipeline()
    run(p, 0, 29 * 60, lambda t: [("car", jitter(PARKED, rng))])
    assert [r for _, _, r in p.saves] == ["new"]
    run(p, 29 * 60, 31 * 60, lambda t: [("car", jitter(PARKED, rng))])
    assert [r for _, _, r in p.saves] == ["new", "max_interval"]
    assert 1800 <= p.saves[1][0] - p.saves[0][0] < 1800 + 2 * DT
    run(p, 31 * 60, 59 * 60, lambda t: [("car", jitter(PARKED, rng))])
    assert len(p.saves) == 2


def test_heavy_jitter_single_frame_spike_does_not_count_as_move():
    rng = random.Random(2)
    p = Pipeline()

    def boxes(t):
        # every 20 s one wild frame (box shifted 120 px), otherwise small jitter
        if int(t * FPS) % 40 == 39:
            return [("car", (220, 200, 420, 320))]
        return [("car", jitter(PARKED, rng))]

    run(p, 0, 20 * 60, boxes)
    assert len(p.saves) == 1


def test_car_leaves_over_45s_and_returns_saves_again():
    rng = random.Random(3)
    p = Pipeline()
    t = run(p, 0, 300, lambda t: [("car", jitter(PARKED, rng))])
    t = run(p, t, t + 50, lambda t: [])              # gone 50 s (> absent_sec)
    run(p, t, t + 300, lambda t: [("car", jitter(PARKED, rng))])
    assert [r for _, _, r in p.saves] == ["new", "new"]


def test_short_dropout_under_45s_does_not_resave():
    rng = random.Random(4)
    p = Pipeline()
    t = run(p, 0, 300, lambda t: [("car", jitter(PARKED, rng))])
    t = run(p, t, t + 30, lambda t: [])              # occluded / missed 30 s
    run(p, t, t + 300, lambda t: [("car", jitter(PARKED, rng))])
    assert len(p.saves) == 1


def test_material_move_saves_gradual_repark():
    """Car pulls forward 160 px over 4 s and stays: tracked as the same
    object, saved once more with reason "moved"."""
    rng = random.Random(5)
    p = Pipeline()
    t = run(p, 0, 120, lambda t: [("car", jitter(PARKED, rng))])
    start = t

    def boxes(t):
        dx = min(160, (t - start) * 40)
        return [("car", jitter((PARKED[0] + dx, PARKED[1], PARKED[2] + dx, PARKED[3]), rng, 3))]

    run(p, t, t + 30 * 60, boxes)       # stays at the new spot 30 min
    assert [r for _, _, r in p.saves] == ["new", "moved"]


def test_parked_car_driving_away_saves_once_at_move_start():
    rng = random.Random(11)
    p = Pipeline()
    t = run(p, 0, 300, lambda t: [("car", jitter(PARKED, rng))])
    start = t

    def boxes(t):
        dx = (t - start) * 50                 # pulls out, leaves frame after ~7 s
        return [] if dx > 400 else [("car", (PARKED[0] + dx, PARKED[1], PARKED[2] + dx, PARKED[3]))]

    run(p, t, t + 120, boxes)
    assert [r for _, _, r in p.saves] == ["new", "moved"]


def test_material_move_saves_abrupt_jump():
    """Box jumps 160 px between frames (e.g. missed frames while moving):
    it can't be associated, so it is saved as a fresh sighting."""
    rng = random.Random(10)
    p = Pipeline()
    moved = (260, 190, 460, 310)
    t = run(p, 0, 120, lambda t: [("car", jitter(PARKED, rng))])
    run(p, t, t + 120, lambda t: [("car", jitter(moved, rng))])
    assert len(p.saves) == 2 and p.saves[1][2] in ("moved", "new")


def test_two_cars_in_different_spots_each_save_once():
    rng = random.Random(6)
    p = Pipeline()
    other = (400, 180, 600, 300)
    run(p, 0, 600, lambda t: [("car", jitter(PARKED, rng)), ("car", jitter(other, rng))])
    assert [r for _, _, r in p.saves] == ["new", "new"]
    assert len(p.d) == 2


def test_car_driving_past_right_after_parked_save_still_saves():
    rng = random.Random(7)
    p = Pipeline()
    t = run(p, 0, 10, lambda t: [("car", jitter(PARKED, rng))])
    assert len(p.saves) == 1
    start = t

    def boxes(t):
        # passing car on the street, 60 px/s left->right, visible ~8 s
        x = 0 + (t - start) * 60
        return [("car", jitter(PARKED, rng)), ("car", (x, 40, x + 150, 120))]

    run(p, t, t + 8, boxes)
    labels = [r for _, _, r in p.saves]
    assert labels[0] == "new" and labels[1] == "new"
    assert 2 <= len(p.saves) <= 3        # passing car saved once (maybe a "moved" too)
    # the parked car itself is never re-saved
    assert p.d.should_save(1, t + 8) is None


def test_old_60s_backstop_would_have_dropped_the_passing_car():
    """Documents why dedup classes use a short backstop cooldown."""
    rng = random.Random(8)
    p = Pipeline(backstop=60.0)
    t = run(p, 0, 10, lambda t: [("car", jitter(PARKED, rng))])
    start = t
    run(p, t, t + 8, lambda t: [("car", jitter(PARKED, rng)),
                                ("car", (0 + (t - start) * 60, 40, 150 + (t - start) * 60, 120))])
    assert len(p.saves) == 1


def test_non_dedup_classes_are_ignored():
    d = SnapshotDeduper(parse_dedup_cfg({"enabled": True}))
    assert d.observe("person", (0, 0, 10, 10), 0.0) is None
    assert d.observe("bicycle", (0, 0, 10, 10), 0.0) is None
    assert d.observe("truck", (0, 0, 10, 10), 0.0) is not None


def test_state_is_pruned_and_capped():
    d = SnapshotDeduper(parse_dedup_cfg({"enabled": True, "max_objects": 16}))
    for i in range(500):                 # 500 distinct flickering boxes, 1 per frame
        x = (i * 37) % 600
        y = (i * 53) % 300
        d.observe("car", (x, y, x + 10, y + 10), i * DT)
        assert len(d) <= 16
    d.prune(500 * DT + 46)
    assert len(d) == 0


def test_config_parsing_defaults_and_validation():
    assert parse_dedup_cfg(None)["enabled"] is False           # block missing
    assert parse_dedup_cfg({})["enabled"] is False             # enabled missing
    assert parse_dedup_cfg({"enabled": "yes"})["enabled"] is False
    c = parse_dedup_cfg({"enabled": True, "absent_sec": -5, "move_iou": "x",
                         "classes": ["car"], "max_interval_sec": 0})
    assert c["enabled"] is True
    assert c["absent_sec"] == 45.0 and c["move_iou"] == 0.5
    assert c["classes"] == frozenset({"car"})
    assert c["max_interval_sec"] == 0
    d = parse_dedup_cfg({"enabled": True})
    assert d["classes"] == frozenset({"car", "truck", "bus", "motorcycle"})
    assert (d["move_iou"], d["move_center_frac"], d["absent_sec"], d["max_interval_sec"]) == \
        (0.5, 0.25, 45.0, 1800.0)


def test_max_interval_zero_disables_periodic_resave():
    rng = random.Random(9)
    p = Pipeline(max_interval_sec=0)
    run(p, 0, 70 * 60, lambda t: [("car", jitter(PARKED, rng))])
    assert len(p.saves) == 1


def test_should_save_has_no_side_effects_until_mark_saved():
    d = SnapshotDeduper(parse_dedup_cfg({"enabled": True}))
    oid = d.observe("car", PARKED, 0.0)
    assert d.should_save(oid, 0.0) is None      # needs new_confirm_frames hits
    d.observe("car", PARKED, 0.5)
    d.observe("car", PARKED, 1.0)
    assert d.should_save(oid, 1.0) == "new"
    assert d.should_save(oid, 1.0) == "new"     # backstop blocked -> still pending
    d.mark_saved(oid, 1.0)
    assert d.should_save(oid, 1.0) is None
