# DHRAS parked-car snapshot dedup

Fixed-camera vehicle snapshots can re-save a parked car all day when box jitter trips `VehicleTracker.is_moving()` and the per-label cooldown then lets another save through. `snapshot_dedup.py` remembers each vehicle and suppresses those repeats. It stays **off** unless a camera's `dedup` block sets `"enabled": true`.

A missing `dedup` block, or `enabled` set to anything other than the boolean `true`, leaves behaviour identical to main. Dedup applies only to fixed cameras (`ptz is None`) and only to the configured classes (default `car`, `truck`, `bus`, `motorcycle`). `person` and `bicycle` keep the original path.

`process_frame` calls `refresh_dedup()` on every frame. `reload_config()` re-reads `cameras_config.json` every 30 frames, so a config edit is picked up within 30 frames and does not need a restart.

## Cooldown

Normal detections call `cooldown_ok(label)`. That uses `get_cooldown_sec()`, which reads the top-level `cooldown_sec` key and defaults to 60 seconds. This is the live Jetson cooldown from `diag/motion-tracker-instrumentation` at `4959985`.

Dedup-managed saves call `cooldown_ok(label, backstop_cooldown_sec)`. The default backstop is 5 seconds, so a car passing just after a parked-car save is still captured. A 60 second gate on that path would drop it.

`VISION_MOTION_DIAG` logging from that same live commit is included and stays off unless the environment variable is set. It does not change motion decisions.

## Enable

`cameras_config.example.json` shows the block on `frontyard` with `"enabled": false`. The real file on the device is `cameras_config.json` (not committed). To turn dedup on, set the block under the fixed camera and wait for the 30-frame reload:

```json
"dedup": {
  "enabled": true,
  "classes": ["car", "truck", "bus", "motorcycle"],
  "match_iou": 0.3,
  "move_iou": 0.5,
  "move_center_frac": 0.25,
  "move_confirm_frames": 3,
  "move_min_interval_sec": 30,
  "new_confirm_frames": 3,
  "absent_sec": 45,
  "max_interval_sec": 1800,
  "backstop_cooldown_sec": 5,
  "max_objects": 64
}
```

A save happens for a **new** sighting (3 hits, no gap over 2 s), a **moved** box (IoU under 0.5 or center shift over 0.25 of the diagonal, 3 frames in a row, at most once per 30 s), or **max_interval** (default 1800 s; `0` disables the periodic re-save). An object unseen for `absent_sec` (default 45 s) is forgotten.

## Deploy

Deploy `vision_service.py` and `snapshot_dedup.py` together. The service imports `snapshot_dedup` at startup, so one file without the other will fail to start.

1. This branch already contains the Phase 4 snapshot and status routes, parked-car dedup, and the live 60-second cooldown plus `VISION_MOTION_DIAG` instrumentation.
2. Keep a copy of the current `vision_service.py` and `cameras_config.json`.
3. Install both `vision_service.py` and `snapshot_dedup.py`.
4. Check them: `python -m py_compile vision_service.py snapshot_dedup.py`
5. Restart vision-hub. With no `dedup` block, or with `"enabled": false`, behaviour matches the previous service.
6. Enable with the `dedup` block in `cameras_config.json`. It hot-reloads within 30 frames. No second restart.

## Rollback

Instant, no restart: set `"enabled": false`, or delete the `dedup` block, in `cameras_config.json`. The running service drops dedup state on the next config reload (within 30 frames) and returns to the main save path.

Full rollback: restore the previous `vision_service.py`, remove `snapshot_dedup.py`, then restart vision-hub.
