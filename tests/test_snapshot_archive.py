"""Protected snapshot archive stays separate from the event log."""
import tempfile
import unittest
from pathlib import Path

from media_store import (
    archive_snapshots,
    classify_media,
    clear_unarchived,
    delete_snapshots,
    gallery_entries,
    trim_event_log,
)

ROOT = Path(__file__).resolve().parents[1]


def write_jpeg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff" + path.name.encode("ascii"))


class SnapshotArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.disposable = self.base / "detections" / "backyard" / "person_20260929_120000.jpg"
        self.other = self.base / "detections" / "frontyard" / "car_20260929_120001.jpg"
        write_jpeg(self.disposable)
        write_jpeg(self.other)
        self.config = self.base / "cameras_config.json"
        self.config.write_text('{"ptz_pass":"secret"}', encoding="utf-8")
        self.events = self.base / "detections" / "events.jsonl"
        self.events.write_text(
            '{"camera":"backyard","class":"person","image":"person_20260929_120000.jpg"}\n',
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_delete_and_clear_keep_protected_files_and_events(self):
        archived = archive_snapshots(self.base, [
            "detections/backyard/person_20260929_120000.jpg",
        ])
        self.assertTrue(archived["ok"])
        self.assertEqual(archived["paths"], ["archive/backyard/person_20260929_120000.jpg"])
        self.assertFalse(self.disposable.exists())
        self.assertTrue((self.base / archived["paths"][0]).is_file())
        self.assertTrue(self.events.is_file())

        listed = gallery_entries(self.base, None, None, 80)
        self.assertEqual(
            {item["path"]: item["archived"] for item in listed},
            {
                "detections/frontyard/car_20260929_120001.jpg": False,
                "archive/backyard/person_20260929_120000.jpg": True,
            },
        )
        self.assertEqual(listed, sorted(listed, key=lambda item: item["mtime"], reverse=True))
        filtered = gallery_entries(
            self.base,
            "backyard",
            self.base / "detections" / "backyard",
            80,
        )
        self.assertEqual([item["path"] for item in filtered], archived["paths"])

        rejected = delete_snapshots(self.base, archived["paths"])
        self.assertFalse(rejected["ok"])
        self.assertTrue((self.base / archived["paths"][0]).is_file())

        deleted = delete_snapshots(self.base, ["detections/frontyard/car_20260929_120001.jpg"])
        self.assertEqual(deleted["removed"], 1)
        self.assertFalse(self.other.exists())
        self.assertTrue((self.base / archived["paths"][0]).is_file())

        write_jpeg(self.other)
        cleared = clear_unarchived(self.base)
        self.assertEqual(cleared["removed"], 1)
        self.assertFalse(self.other.exists())
        self.assertTrue((self.base / archived["paths"][0]).is_file())
        self.assertTrue(self.events.is_file())

    def test_snapshot_route_rejects_config_and_traversal(self):
        kind, path = classify_media(self.base, "detections/frontyard/car_20260929_120001.jpg")
        self.assertEqual(kind, "ok")
        self.assertTrue(path.is_file())
        self.assertEqual(classify_media(self.base, "cameras_config.json")[0], "forbidden")
        self.assertEqual(classify_media(self.base, "detections/../cameras_config.json")[0], "forbidden")
        self.assertEqual(classify_media(self.base, "/etc/passwd")[0], "forbidden")
        self.assertEqual(classify_media(self.base, "detections/backyard/notes.txt")[0], "forbidden")
        self.assertEqual(classify_media(self.base, "detections/missing/no_such.jpg")[0], "missing")
        self.assertFalse(archive_snapshots(self.base, ["archive/backyard/a.jpg"])["ok"])
        self.assertFalse(archive_snapshots(self.base, ["../cameras_config.json"])["ok"])
        too_many = ["detections/backyard/shot_%02d.jpg" % n for n in range(41)]
        self.assertFalse(delete_snapshots(self.base, too_many)["ok"])

    def test_event_retention_runs_on_write_and_event_reads_stay_reads(self):
        lines = ['{"n":%d}\n' % n for n in range(205)]
        self.events.write_text("".join(lines), encoding="utf-8")
        self.assertEqual(trim_event_log(self.events, 200), 200)
        kept = self.events.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(kept), 200)
        self.assertEqual(kept[-1], '{"n":204}')
        self.assertEqual(kept[0], '{"n":5}')

        dashboard = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        events = dashboard[dashboard.index("def api_events"):dashboard.index("@app.route", dashboard.index("def api_events"))]
        self.assertNotIn("trim_event_log", events)
        self.assertNotIn("/api/events/retain", dashboard)
        clear = dashboard[dashboard.index("def api_events_clear"):dashboard.index("@app.route", dashboard.index("def api_events_clear"))]
        self.assertIn("EVENTS_FILE.unlink()", clear)
        self.assertNotIn("ARCHIVE_DIR", clear)
        self.assertIn('"/api/gallery/archive"', dashboard)
        self.assertIn('"/api/gallery/delete"', dashboard)
        self.assertIn('"/api/gallery/clear-unarchived"', dashboard)
        self.assertIn("classify_media", dashboard)
        vision = (ROOT / "vision_service.py").read_text(encoding="utf-8")
        writer = vision[vision.index("def log_event"):vision.index("# MQTT", vision.index("def log_event"))]
        self.assertIn("trim_event_log", writer)


if __name__ == "__main__":
    unittest.main()
