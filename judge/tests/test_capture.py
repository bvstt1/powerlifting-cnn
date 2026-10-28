"""Pruebas del módulo de captura con cámaras simuladas.

python -m unittest judge.tests.test_capture
"""

import time
import unittest

import numpy as np

from judge.capture import CameraError, Frame, MultiCameraCapture, select_set
from judge.capture.camera import backend_flag
from judge.config import VIEWS, CaptureConfig


class FakeCapture:
    """Imita cv2.VideoCapture: cada grab() tarda 1/fps y entrega una imagen numerada."""

    def __init__(self, fps, opened=True, fail_after=None):
        self.period = 1.0 / fps
        self.opened = opened
        self.fail_after = fail_after
        self.grabs = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def grab(self):
        time.sleep(self.period)
        if self.fail_after is not None and self.grabs >= self.fail_after:
            return False
        self.grabs += 1
        return True

    def retrieve(self):
        return True, np.full((4, 6, 3), self.grabs % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def capture_config(max_skew_ms=40.0):
    return CaptureConfig({"front": 0, "left": 1, "right": 2}, 640, 480, 30, "auto", max_skew_ms)


def factory_for(fakes):
    def make(index, _config):
        return fakes[index]
    return make


def frame(view, timestamp, seq):
    return Frame(view, np.zeros((2, 2, 3), dtype=np.uint8), timestamp, seq)


class SelectSetTest(unittest.TestCase):
    def setUp(self):
        self.delivered = {view: 0 for view in VIEWS}

    def test_waits_when_a_view_has_no_frame(self):
        latest = {"front": frame("front", 1.0, 1), "left": frame("left", 1.0, 1), "right": None}
        self.assertEqual(select_set(latest, self.delivered, 0.04), ("wait", None))

    def test_delivers_synchronized_frames(self):
        latest = {v: frame(v, 1.0 + 0.01 * i, 1) for i, v in enumerate(VIEWS)}
        action, frame_set = select_set(latest, self.delivered, 0.04)
        self.assertEqual(action, "deliver")
        self.assertAlmostEqual(frame_set.skew_ms, 20.0, places=3)

    def test_does_not_redeliver_the_same_frames(self):
        latest = {v: frame(v, 1.0, 3) for v in VIEWS}
        self.delivered = {v: 3 for v in VIEWS}
        self.assertEqual(select_set(latest, self.delivered, 0.04), ("wait", None))

    def test_waits_if_only_some_views_have_new_frames(self):
        latest = {v: frame(v, 1.0, 5) for v in VIEWS}
        self.delivered = {"front": 4, "left": 5, "right": 4}
        self.assertEqual(select_set(latest, self.delivered, 0.04)[0], "wait")

    def test_skewed_set_discards_the_oldest_view(self):
        latest = {"front": frame("front", 1.00, 1), "left": frame("left", 1.20, 1),
                  "right": frame("right", 1.21, 1)}
        self.assertEqual(select_set(latest, self.delivered, 0.04), ("skewed", "front"))

    def test_tolerance_is_inclusive(self):
        # Valores exactos en binario (1/8 s) para evitar errores de coma flotante.
        latest = {"front": frame("front", 1.0, 1), "left": frame("left", 1.125, 1),
                  "right": frame("right", 1.0625, 1)}
        self.assertEqual(select_set(latest, self.delivered, 0.125)[0], "deliver")
        self.assertEqual(select_set(latest, self.delivered, 0.1249)[0], "skewed")


class MultiCameraCaptureTest(unittest.TestCase):
    def test_delivers_one_frame_per_view_with_increasing_sequence(self):
        fakes = {i: FakeCapture(fps=200) for i in range(3)}
        with MultiCameraCapture(capture_config(), factory_for(fakes)) as capture:
            first = capture.read(timeout=2.0)
            second = capture.read(timeout=2.0)
        self.assertEqual(set(first.frames), set(VIEWS))
        self.assertLessEqual(first.skew_ms, 40.0)
        for view in VIEWS:
            self.assertGreater(second.frames[view].seq, first.frames[view].seq)
        self.assertEqual(first.images()["front"].shape, (4, 6, 3))

    def test_slow_camera_does_not_block_or_age_the_others(self):
        fakes = {0: FakeCapture(fps=200), 1: FakeCapture(fps=200), 2: FakeCapture(fps=25)}
        with MultiCameraCapture(capture_config(), factory_for(fakes)) as capture:
            sets = [capture.read(timeout=2.0) for _ in range(5)]
        self.assertTrue(all(s is not None for s in sets))
        self.assertTrue(all(s.skew_ms <= 40.0 for s in sets))
        # Las cámaras rápidas se leen sin acumular imágenes viejas: hay saltos de seq.
        self.assertGreater(sets[-1].frames["front"].seq - sets[0].frames["front"].seq, 5)

    def test_unsynchronized_cameras_drop_frames_and_still_deliver(self):
        fakes = {i: FakeCapture(fps=100) for i in range(3)}
        with MultiCameraCapture(capture_config(max_skew_ms=0.001), factory_for(fakes)) as capture:
            result = capture.read(timeout=0.5)
            dropped = capture.dropped_frames
        self.assertIsNone(result)
        self.assertGreater(dropped, 0)

    def test_read_times_out_when_a_camera_stalls(self):
        fakes = {0: FakeCapture(fps=200), 1: FakeCapture(fps=200), 2: FakeCapture(fps=2)}
        with MultiCameraCapture(capture_config(), factory_for(fakes)) as capture:
            started = time.monotonic()
            self.assertIsNone(capture.read(timeout=0.1))
            self.assertLess(time.monotonic() - started, 0.5)

    def test_camera_that_cannot_open_releases_the_others(self):
        fakes = {0: FakeCapture(fps=200), 1: FakeCapture(fps=200), 2: FakeCapture(fps=200, opened=False)}
        capture = MultiCameraCapture(capture_config(), factory_for(fakes))
        with self.assertRaises(CameraError) as context:
            capture.start()
        self.assertIn("cámara 2", str(context.exception).lower())
        self.assertTrue(all(fake.released for fake in fakes.values()))

    def test_camera_that_dies_raises_on_read(self):
        fakes = {0: FakeCapture(fps=200), 1: FakeCapture(fps=200, fail_after=3), 2: FakeCapture(fps=200)}
        with MultiCameraCapture(capture_config(), factory_for(fakes), max_failures=3) as capture:
            with self.assertRaises(CameraError) as context:
                for _ in range(200):
                    capture.read(timeout=0.2)
        self.assertIn("left", str(context.exception))

    def test_stop_releases_every_camera_and_threads(self):
        fakes = {i: FakeCapture(fps=200) for i in range(3)}
        capture = MultiCameraCapture(capture_config(), factory_for(fakes)).start()
        threads = [r._thread for r in capture._readers.values()]
        capture.stop()
        self.assertTrue(all(fake.released for fake in fakes.values()))
        self.assertFalse(any(t.is_alive() for t in threads))

    def test_read_before_start_is_an_error(self):
        with self.assertRaises(CameraError):
            MultiCameraCapture(capture_config(), factory_for({})).read()

    def test_cannot_start_twice(self):
        fakes = {i: FakeCapture(fps=200) for i in range(3)}
        with MultiCameraCapture(capture_config(), factory_for(fakes)) as capture:
            with self.assertRaises(CameraError):
                capture.start()


class BackendTest(unittest.TestCase):
    def test_backend_names(self):
        import cv2
        self.assertEqual(backend_flag("dshow"), cv2.CAP_DSHOW)
        self.assertEqual(backend_flag("msmf"), cv2.CAP_MSMF)
        self.assertEqual(backend_flag("any"), cv2.CAP_ANY)
        self.assertIn(backend_flag("auto"), (cv2.CAP_DSHOW, cv2.CAP_ANY))


if __name__ == "__main__":
    unittest.main()
