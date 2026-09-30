"""Snapshot archive paths and event-log retention.

No camera credentials, stream URLs, or model policy live here.
"""

import os
from datetime import datetime
from pathlib import Path

MEDIA_ROOTS = frozenset({"detections", "archive"})
GALLERY_BATCH_MAX = 40
EVENT_RETAIN_MAX = 200


def safe_token(value):
    return isinstance(value, str) and bool(value) and all(
        char.isalnum() or char in "_-" for char in value
    )


def safe_media_path(value, roots=None):
    """Relative JPEG key under detections/ or archive/ only."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.startswith("/") or "\\" in text or ".." in text.split("/"):
        return None
    parts = [part for part in text.split("/") if part]
    allowed = MEDIA_ROOTS if roots is None else roots
    if len(parts) < 3 or parts[0] not in allowed:
        return None
    if not all(safe_token(part) for part in parts[1:-1]):
        return None
    name = parts[-1]
    if not name.endswith(".jpg"):
        return None
    if not safe_token(name[:-4]):
        return None
    return "/".join(parts)


def snapshot_paths(paths, roots):
    if not isinstance(paths, list) or not paths or len(paths) > GALLERY_BATCH_MAX:
        return None
    cleaned = []
    for item in paths:
        safe = safe_media_path(item, roots)
        if safe is None or safe in cleaned:
            return None
        cleaned.append(safe)
    return cleaned


def _contained(root, path):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def classify_media(base_dir, filepath):
    rel = safe_media_path(filepath)
    if rel is None:
        return "forbidden", None
    base = Path(base_dir).resolve()
    path = (base / rel).resolve()
    if not _contained(base, path):
        return "forbidden", None
    if not path.parts or path.relative_to(base).parts[0] not in MEDIA_ROOTS:
        return "forbidden", None
    if not path.is_file():
        return "missing", None
    return "ok", path


def _jpeg_rows(base, paths):
    rows = []
    root = Path(base).resolve()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not _contained(root, resolved):
            continue
        rel = resolved.relative_to(root).as_posix()
        if safe_media_path(rel) is None:
            continue
        modified = resolved.stat().st_mtime
        rows.append((modified, {
            "path": rel,
            "name": resolved.name,
            "camera": resolved.parent.name,
            "mtime": modified,
            "ts": datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S"),
            "archived": rel.startswith("archive/"),
        }))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in rows]


def gallery_entries(base_dir, camera_id, snapshot_dir, limit):
    base = Path(base_dir).resolve()
    files = []
    if camera_id:
        snap = Path(snapshot_dir) if snapshot_dir else None
        if snap is not None and snap.is_dir() and _contained(base, snap.resolve()):
            files.extend(path for path in snap.glob("*.jpg") if path.is_file())
        archive_cam = base / "archive" / camera_id
        if archive_cam.is_dir():
            files.extend(path for path in archive_cam.glob("*.jpg") if path.is_file())
    else:
        detect = base / "detections"
        archive = base / "archive"
        if detect.is_dir():
            files.extend(path for path in detect.rglob("*.jpg") if path.is_file())
        if archive.is_dir():
            files.extend(path for path in archive.rglob("*.jpg") if path.is_file())
    rows = _jpeg_rows(base, files)
    if isinstance(limit, int) and limit >= 0:
        return rows[:limit]
    return rows


def _detection_files(base_dir, relatives):
    cleaned = snapshot_paths(relatives, frozenset({"detections"}))
    if cleaned is None:
        return None
    base = Path(base_dir).resolve()
    detect = (base / "detections").resolve()
    found = []
    for rel in cleaned:
        path = (base / rel).resolve()
        if not _contained(detect, path) or not path.is_file():
            return None
        found.append((rel, path))
    return found


def _unique_dest(directory, name):
    dest = directory / name
    if not dest.exists():
        return dest
    stem = name[:-4]
    number = 1
    while True:
        candidate = directory / ("%s-%d.jpg" % (stem, number))
        if not candidate.exists():
            return candidate
        number += 1


def archive_snapshots(base_dir, paths):
    found = _detection_files(base_dir, paths)
    if found is None:
        return {"ok": False}
    archive = (Path(base_dir).resolve() / "archive").resolve()
    moved = []
    for rel, path in found:
        camera = rel.split("/")[1]
        dest_dir = archive / camera
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = _unique_dest(dest_dir, path.name)
        os.replace(str(path), str(dest))
        moved.append("archive/%s/%s" % (camera, dest.name))
    return {"ok": True, "paths": moved}


def delete_snapshots(base_dir, paths):
    found = _detection_files(base_dir, paths)
    if found is None:
        return {"ok": False}
    removed = []
    for rel, path in found:
        path.unlink()
        removed.append(rel)
    return {"ok": True, "paths": removed, "removed": len(removed)}


def clear_unarchived(base_dir):
    detect = (Path(base_dir).resolve() / "detections").resolve()
    removed = 0
    if detect.is_dir():
        for path in list(detect.rglob("*.jpg")):
            resolved = path.resolve()
            if not _contained(detect, resolved) or not resolved.is_file():
                continue
            resolved.unlink()
            removed += 1
    return {"ok": True, "removed": removed}


def is_archived_file(archive_dir, path):
    return _contained(archive_dir, Path(path).resolve())


def trim_event_log(path, max_records=EVENT_RETAIN_MAX):
    """Keep the newest records while an event is being stored. Not a read path."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    lines = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line if line.endswith("\n") else line + "\n")
    if len(lines) <= max_records:
        return len(lines)
    kept = lines[-max_records:]
    temporary = file_path.with_name(file_path.name + ".tmp")
    temporary.write_text("".join(kept), encoding="utf-8")
    os.replace(temporary, file_path)
    return len(kept)
