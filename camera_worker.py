"""
camera_worker.py — CameraWorker class for DHRAS Vision Service

One instance per camera. Runs a dedicated RTSP reader thread that always
overwrites the latest-frame slot. The inference loop calls get_frame()
non-blocking — it always gets the freshest frame or None.

Auto-reconnects on stream drop. Never crashes the main inference loop.
"""

import threading
import time
import logging
import cv2

logger = logging.getLogger(__name__)

RECONNECT_MAX_BACKOFF = 30   # seconds, cap on exponential backoff
STALE_THRESHOLD_SEC = 10     # seconds without a frame before considered stale


class CameraWorker:
    def __init__(self, cam_id: str, config: dict):
        self.cam_id = cam_id
        self.config = config
        self.rtsp_url = config["rtsp_url"]

        # Latest raw frame — single slot, always overwritten, never queued
        self._frame = None
        self._frame_lock = threading.Lock()

        # Latest annotated frame — populated by inference loop, consumed by stream_proxy (Session 1B)
        self._annotated_frame = None
        self._annotated_lock = threading.Lock()

        # Health state — read by inference loop and health monitor
        self.online = False
        self.last_frame_time = 0.0

        # Control flags
        self.paused = False
        self._stop_event = threading.Event()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"rtsp-{cam_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{cam_id}] CameraWorker started")

    # ------------------------------------------------------------------
    # Public interface used by inference loop
    # ------------------------------------------------------------------

    def get_frame(self):
        """Non-blocking. Returns a copy of the latest raw frame, or None."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def set_annotated_frame(self, frame):
        """Store the bounding-box-annotated frame. Called from inference loop."""
        with self._annotated_lock:
            self._annotated_frame = frame.copy()

    def get_annotated_frame(self):
        """Return latest annotated frame. Used by stream_proxy (Session 1B)."""
        with self._annotated_lock:
            return self._annotated_frame.copy() if self._annotated_frame is not None else None

    def pause(self):
        """Remove from inference queue and stop reading frames."""
        logger.info(f"[{self.cam_id}] Paused")
        self.paused = True
        self.online = False

    def resume(self):
        """Re-add to inference queue."""
        logger.info(f"[{self.cam_id}] Resumed")
        self.paused = False

    def stop(self):
        """Signal the reader thread to exit."""
        self._stop_event.set()

    @property
    def is_stale(self) -> bool:
        """True if no frame has arrived in STALE_THRESHOLD_SEC seconds."""
        return (time.time() - self.last_frame_time) > STALE_THRESHOLD_SEC

    # ------------------------------------------------------------------
    # Internal reader loop
    # ------------------------------------------------------------------

    def _reader_loop(self):
        backoff = 1
        while not self._stop_event.is_set():
            if self.paused:
                time.sleep(0.5)
                continue

            cap = self._open_stream()
            if cap is None:
                self.online = False
                logger.warning(
                    f"[{self.cam_id}] Cannot open RTSP stream — retry in {backoff}s"
                )
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF)
                continue

            logger.info(f"[{self.cam_id}] RTSP stream opened successfully")
            self.online = True
            backoff = 1  # reset on successful connect

            while not self._stop_event.is_set() and not self.paused:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{self.cam_id}] Frame read failed — reconnecting")
                    self.online = False
                    break

                # Always overwrite — never queue. This is the core latency guarantee.
                with self._frame_lock:
                    self._frame = frame

                self.last_frame_time = time.time()
                self.online = True

            cap.release()
            logger.debug(f"[{self.cam_id}] Stream released")

        logger.info(f"[{self.cam_id}] Reader loop exited")

    def _open_stream(self):
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                # Limit decoder buffer to 1 frame so get_frame() always returns
                # the freshest image, not a frame that was buffered seconds ago.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        except Exception as exc:
            logger.error(f"[{self.cam_id}] Exception opening stream: {exc}")
        return None
