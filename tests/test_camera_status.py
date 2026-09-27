"""Camera status follows the configured camera set."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeRequests:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout=1.5):
        self.urls.append(url)
        if url.endswith("/dock"):
            return FakeResponse({"online": True, "mode": "ACTIVE"})
        if url.endswith("/frontyard"):
            raise OSError("camera down")
        return FakeResponse(["not", "an", "object"])


class CameraStatusTests(unittest.TestCase):
    def route(self, cameras):
        text = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        start = text.index("def api_cameras_status")
        end = text.index("@app.route", start)
        requests = FakeRequests()
        namespace = {
            "jsonify": lambda payload: payload,
            "load_config": lambda: {"cameras": cameras},
            "requests": requests,
        }
        exec(text[start:end], namespace)
        return namespace["api_cameras_status"](), requests

    def test_new_camera_is_included_and_one_failure_stays_local(self):
        statuses, requests = self.route({
            "frontyard": {"name": "Front Yard"},
            "dock": {"name": "Dock"},
            "gate": {"name": "Gate"},
        })
        self.assertEqual(statuses["frontyard"], {"online": False, "mode": "offline"})
        self.assertEqual(statuses["dock"], {"online": True, "mode": "ACTIVE"})
        self.assertEqual(statuses["gate"], {"online": False, "mode": "offline"})
        self.assertEqual(
            requests.urls,
            [
                "http://127.0.0.1:8081/status/frontyard",
                "http://127.0.0.1:8081/status/dock",
                "http://127.0.0.1:8081/status/gate",
            ],
        )

    def test_status_route_is_not_a_fixed_camera_list(self):
        text = (ROOT / "dashboard.py").read_text(encoding="utf-8")
        start = text.index("def api_cameras_status")
        end = text.index("@app.route", start)
        body = text[start:end]
        self.assertIn('load_config()["cameras"]', body)
        self.assertNotIn('["frontyard", "backyard", "indoor"]', body)
        self.assertIn("for cam_id in cameras", body)


if __name__ == "__main__":
    unittest.main()
