"""Source contract for the current-frame snapshot route."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SnapshotRouteTests(unittest.TestCase):
    def test_snapshot_returns_the_current_frame_without_inference(self):
        text = (ROOT / "vision_service.py").read_text(encoding="utf-8")
        start = text.index("def snapshot_route")
        end = text.index("\n@flask_app.route", start)
        body = text[start:end]
        self.assertIn('"/snapshot/<cam_id>"', text)
        self.assertIn("get_frame()", body)
        self.assertIn('mimetype="image/jpeg"', body)
        self.assertNotIn("model(", body)
        self.assertNotIn("YOLO", body)
        self.assertNotIn("_trigger_action", body)


if __name__ == "__main__":
    unittest.main()
