"""Pruebas sin ventana: ROI, continuidad, pausa, fallo, salto y cambio de vídeo."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from display.sq.bar_tracker import BarTracker, VideoViewport, draw_vertical_plot


class FakeTracker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.initial = None
        self.calls = 0

    def init(self, frame, box):
        self.initial = box
        return True

    def update(self, frame):
        self.calls += 1
        return next(self.outcomes)


class BarTests(unittest.TestCase):
    def test_mapping_with_letterbox_and_panel(self):
        viewport = VideoViewport(1080, 1920, 0, 0, 1080, 800)
        self.assertEqual((viewport.left, viewport.top, viewport.width, viewport.height),
                         (315, 0, 450, 800))
        self.assertIsNone(viewport.to_video(314, 400))  # banda izquierda
        self.assertIsNone(viewport.to_video(765, 400))  # banda derecha
        self.assertIsNone(viewport.to_video(1100, 400))  # panel
        self.assertEqual(viewport.roi_from_display((400, 200), (440, 240)),
                         (204, 480, 96, 96))
        self.assertIsNone(viewport.roi_from_display((300, 200), (440, 240)))
        self.assertIsNone(viewport.roi_from_display((400, 200), (402, 201)))
        # También controla bandas verticales de un vídeo horizontal.
        horizontal = VideoViewport(1920, 1080, 0, 0, 1080, 800)
        self.assertGreater(horizontal.top, 0)
        self.assertIsNone(horizontal.to_video(100, horizontal.top-1))

    def test_state_machine_sequential_pause_failure_reset_and_video_change(self):
        fake = FakeTracker([(True, (21, 30, 20, 12)),
                            (False, (0, 0, 0, 0))])
        bar = BarTracker(30, tracker_factory=lambda: fake)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        bar.set_enabled(True)
        self.assertTrue(bar.initialize(frame, 5, (20, 30, 20, 12)))
        self.assertEqual(bar.records[0]["center_y"], 36)
        # Pausar/redibujar nunca llama update; solo el siguiente frame.
        self.assertEqual(fake.calls, 0)
        self.assertTrue(bar.update(frame, 6))
        self.assertEqual(fake.calls, 1)
        self.assertFalse(bar.update(frame, 7))
        self.assertEqual(bar.state, "seguimiento perdido")
        self.assertIsNone(bar.current_center(7))
        self.assertIsNone(bar.records[-1]["center_y"])
        self.assertIsNone(bar.records[-1]["box_x"])
        self.assertFalse(bar.update(frame, 8))
        self.assertEqual(fake.calls, 2)
        bar.reset()
        self.assertEqual(bar.state, "sin inicializar")
        self.assertEqual(bar.records, [])
        self.assertTrue(bar.initialize(frame, 15, (20, 30, 20, 12)))
        self.assertFalse(bar.update(frame, 13))  # retroceso
        self.assertEqual(bar.records, [])
        self.assertTrue(bar.initialize(frame, 20, (20, 30, 20, 12)))
        self.assertFalse(bar.update(frame, 30))  # salto
        self.assertEqual(bar.state, "sin inicializar")
        bar.initialize(frame, 35, (20, 30, 20, 12))
        bar.change_video(60)
        self.assertTrue(bar.enabled)
        self.assertEqual(bar.records, [])
        self.assertEqual(bar.fps, 60)
        bar.set_enabled(False)
        self.assertFalse(bar.enabled)

    def test_plot_and_csv_do_not_invent_lost_positions(self):
        fake = FakeTracker([(False, (0, 0, 0, 0))])
        bar = BarTracker(25, tracker_factory=lambda: fake)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        bar.set_enabled(True)
        bar.initialize(frame, 4, (20, 30, 20, 12))
        bar.update(frame, 5)
        viewport = VideoViewport(120, 100, 0, 0, 120, 100)
        canvas = np.zeros((200, 350, 3), dtype=np.uint8)
        bar.draw(canvas, viewport, 5)
        draw_vertical_plot(canvas, bar, 5, 50, 100, (140, 10, 200, 160), smooth=True)
        self.assertTrue(canvas.any())
        with tempfile.TemporaryDirectory() as folder:
            path = bar.csv_export(Path(folder) / "diagnostico.csv")
            with path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["status"], "perdido")
        self.assertEqual(rows[-1]["center_y"], "")

    def test_visors_canvas_and_tracker_overlay(self):
        from display.sq import visualizar_keypoints_yolo_sq as visor
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = 30
        points = np.array([[.40,.55,.9],[.42,.35,.9],[.43,.80,.9],
                           [.44,.84,.9],[.41,.81,.9]], dtype=np.float32)
        viewport = VideoViewport(1920,1080,0,0,1080,800)
        canvas = visor.dibujar(frame,points,"left",10,"pose.pt",viewport)
        self.assertEqual(canvas.shape,(800,1500,3))
        self.assertTrue(np.all(canvas[0,10] == 0))  # banda negra arriba
        fake = FakeTracker([])
        bar = BarTracker(30,tracker_factory=lambda: fake)
        bar.set_enabled(True)
        bar.initialize(frame,10,(400,350,40,40))
        bar.draw(canvas,viewport,10)
        cx,cy = viewport.to_display(420,370)
        np.testing.assert_array_equal(canvas[cy,cx],(255,255,255))
        self.assertGreater(canvas[cy,cx+16,1],200)  # aro amarillo visible

    def test_visor_keyboard_mouse_playback_and_playlist_without_gui(self):
        from display.sq import visualizar_keypoints_yolo_sq as visor
        root = Path(__file__).resolve().parents[2]
        videos = [root / "dataset" / "sq" / "left" / f"sq_{i:03d}.mp4" for i in (7,225)]
        if not all(path.exists() for path in videos):
            self.skipTest("Se necesitan vídeos de desarrollo sq_007 y sq_225")
        capture_mouse = {}
        created = []
        fake_trackers = []

        def fake_factory():
            fake = FakeTracker([(True,(750,200,40,40))]*3)
            fake_trackers.append(fake)
            return fake

        def make_tracker(fps):
            tracker = BarTracker(fps, tracker_factory=fake_factory)
            created.append(tracker)
            return tracker

        # T desactiva/reactiva; S pausa, arrastre inicia; N avanza; B retrocede;
        # R reinicia; se re-selecciona; . y , saltan; ]/[ cambian vídeo;
        # Q termina. waitKey es simulado, VideoCapture lee vídeos reales.
        commands = iter((ord("t"), ord("t"), ord("s"), "mouse", ord("n"),
                         ord("b"), ord("r"), ord("s"), "mouse",
                         ord("."), ord(","), ord("]"), ord("["), ord("q")))
        images = []

        def key(delay):
            command = next(commands)
            if command == "mouse":
                callback = capture_mouse["callback"]
                callback(visor.cv2.EVENT_LBUTTONDOWN, 410, 180, 0, None)
                callback(visor.cv2.EVENT_MOUSEMOVE, 442, 210, 0, None)
                callback(visor.cv2.EVENT_LBUTTONUP, 442, 210, 0, None)
                return 255
            return command

        with patch.object(visor, "BarTracker", side_effect=make_tracker), \
             patch.object(visor.cv2, "namedWindow"), \
             patch.object(visor.cv2, "setMouseCallback", side_effect=lambda w, cb: capture_mouse.update(callback=cb)), \
             patch.object(visor.cv2, "imshow", side_effect=lambda w, image: images.append(image.copy())), \
             patch.object(visor.cv2, "waitKey", side_effect=key), \
             patch.object(visor.cv2, "destroyAllWindows"):
            visor.visualizar("left", videos[0], root / "models" / "sq" / "sq_side_skeleton_v3.pt",
                             videos=videos, tracking=True, usar_keypoints=True)
        self.assertEqual(len(created), 1)  # misma sesión, change_video limpia
        self.assertEqual(len(fake_trackers), 2)  # primera ROI y reselección
        self.assertEqual(fake_trackers[0].calls, 1)  # N, pausa no repite update
        self.assertEqual(fake_trackers[1].calls, 0)  # antes de cambiar vídeo
        self.assertTrue(created[0].enabled)
        self.assertEqual(created[0].records, [])
        self.assertGreaterEqual(len(images), 5)
        self.assertTrue(all(image.shape == (800,1500,3) for image in images))


if __name__ == "__main__":
    unittest.main()
