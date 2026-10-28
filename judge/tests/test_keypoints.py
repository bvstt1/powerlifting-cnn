"""Pruebas del módulo de keypoints.

python -m unittest judge.tests.test_keypoints
"""

import importlib.util
import unittest

import numpy as np

from judge import config as cfg
from judge.config import VIEWS
from judge.keypoints import KeypointResult, KeypointStage, draw_skeleton
from judge.keypoints.base import empty_points
from judge.keypoints.mediapipe_pose import MediaPipeFrontDeadliftExtractor, landmarks_to_points
from judge.keypoints.yolo import YoloKeypointExtractor, result_to_points


class Tensor:
    """Imita un tensor de torch: .cpu().numpy()."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeYoloResult:
    def __init__(self, people=None, confidences=None, box_conf=None):
        """people: lista de (K, 2) en coordenadas normalizadas."""
        if people is None:
            self.keypoints, self.boxes = None, None
            return
        self.keypoints = type("K", (), {})()
        self.keypoints.xyn = [Tensor(p) for p in people]
        self.keypoints.conf = None if confidences is None else [Tensor(c) for c in confidences]
        self.keypoints.__class__.__len__ = lambda s: len(s.xyn)
        self.boxes = None if box_conf is None else type("B", (), {"conf": Tensor(box_conf)})()
        if self.boxes is not None:
            self.boxes.__class__.__len__ = lambda s: len(s.conf.array)


class FakeYoloModel:
    def __init__(self, num_points, result_factory=None):
        self.model = type("M", (), {"yaml": {"kpt_shape": [num_points, 3]}})()
        self.num_points = num_points
        self.calls = []
        self.result_factory = result_factory or (lambda image: FakeYoloResult())

    def __call__(self, images, imgsz, device, verbose):
        self.calls.append({"n": len(images), "imgsz": imgsz, "device": device, "verbose": verbose})
        return [self.result_factory(image) for image in images]


class KeypointResultTest(unittest.TestCase):
    def test_empty_result_is_not_detected(self):
        self.assertFalse(KeypointResult(empty_points(5)).detected)

    def test_visible_excludes_yolo_missing_points_and_low_confidence(self):
        points = np.array([[0.2, 0.3, 0.9], [0.0, 0.0, 0.0], [0.5, 0.5, 0.2], [np.nan, np.nan, 0.0]],
                          dtype=np.float32)
        result = KeypointResult(points)
        self.assertTrue(result.detected)
        self.assertEqual(result.visible().tolist(), [True, False, True, False])
        self.assertEqual(result.visible(0.5).tolist(), [True, False, False, False])

    def test_to_pixels(self):
        result = KeypointResult(np.array([[0.5, 0.25, 1.0]], dtype=np.float32))
        self.assertEqual(result.to_pixels(200, 100).tolist(), [[100.0, 25.0]])


class YoloConversionTest(unittest.TestCase):
    def test_no_person_gives_all_nan(self):
        points = result_to_points(FakeYoloResult(), 4)
        self.assertEqual(points.shape, (4, 3))
        self.assertTrue(np.isnan(points).all())

    def test_picks_the_person_with_highest_box_confidence(self):
        first, second = np.full((3, 2), 0.1), np.full((3, 2), 0.7)
        result = FakeYoloResult([first, second], [np.full(3, 0.5), np.full(3, 0.9)], box_conf=[0.3, 0.8])
        points = result_to_points(result, 3)
        np.testing.assert_allclose(points[:, :2], second)
        np.testing.assert_allclose(points[:, 2], 0.9)
        self.assertEqual(points.dtype, np.float32)

    def test_missing_confidence_defaults_to_one(self):
        points = result_to_points(FakeYoloResult([np.full((2, 2), 0.4)], None, box_conf=[0.9]), 2)
        np.testing.assert_allclose(points[:, 2], 1.0)

    def test_undetected_points_keep_the_yolo_convention(self):
        xy = np.array([[0.3, 0.4], [0.0, 0.0]])
        points = result_to_points(FakeYoloResult([xy], [np.array([0.9, 0.0])], box_conf=[0.9]), 2)
        self.assertEqual(points[1].tolist(), [0.0, 0.0, 0.0])

    def test_wrong_point_count_is_an_error(self):
        result = FakeYoloResult([np.zeros((5, 2))], [np.zeros(5)], box_conf=[0.9])
        with self.assertRaises(RuntimeError):
            result_to_points(result, 3)


class YoloExtractorTest(unittest.TestCase):
    def test_rejects_model_with_different_keypoint_count(self):
        with self.assertRaises(ValueError):
            YoloKeypointExtractor("x.pt", 8, model=FakeYoloModel(6), device="cpu")

    def test_batches_images_in_one_call_with_training_imgsz(self):
        model = FakeYoloModel(2)
        extractor = YoloKeypointExtractor("x.pt", 2, model=model, device="cpu")
        images = [np.zeros((48, 64, 3), dtype=np.uint8)] * 2
        results = extractor.extract_many(images)
        self.assertEqual(len(results), 2)
        self.assertEqual(model.calls, [{"n": 2, "imgsz": 960, "device": "cpu", "verbose": False}])
        self.assertEqual(extractor.extract_many([]), [])

    def test_single_extract(self):
        extractor = YoloKeypointExtractor("x.pt", 2, imgsz=640, model=FakeYoloModel(2), device="cpu")
        self.assertFalse(extractor.extract(np.zeros((8, 8, 3), dtype=np.uint8)).detected)


class Landmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility


def random_landmarks(rng, hands_visible=True, knees_visible=True):
    landmarks = [Landmark(rng.random(), rng.random(), rng.random()) for _ in range(33)]
    for i in (15, 17, 19, 21, 16, 18, 20, 22):
        landmarks[i].visibility = rng.uniform(0.5, 1.0) if hands_visible else rng.uniform(0.0, 0.49)
    for i in (25, 26):
        landmarks[i].visibility = rng.uniform(0.5, 1.0) if knees_visible else rng.uniform(0.0, 0.49)
    return landmarks


def load_training_extractor():
    path = cfg.ROOT / "extract_keypoints" / "dl" / "extract_keypoints_mediapipe_dl.py"
    spec = importlib.util.spec_from_file_location("training_extract_dl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MediaPipeLayoutTest(unittest.TestCase):
    def test_matches_the_training_extractor(self):
        """Mismos 25 puntos que `extract_keypoints_mediapipe_dl.py`, con todas las ramas."""
        original = load_training_extractor().extraer_frame_front
        rng = np.random.default_rng(0)
        for hands in (True, False):
            for knees in (True, False):
                for _ in range(25):
                    landmarks = random_landmarks(rng, hands, knees)
                    np.testing.assert_array_equal(landmarks_to_points(landmarks), original(landmarks))
        # una sola mano visible
        landmarks = random_landmarks(rng)
        for i in (16, 18, 20, 22):
            landmarks[i].visibility = 0.1
        np.testing.assert_array_equal(landmarks_to_points(landmarks), original(landmarks))

    def test_layout(self):
        landmarks = random_landmarks(np.random.default_rng(1))
        points = landmarks_to_points(landmarks)
        self.assertEqual(points.shape, (25, 3))
        self.assertEqual(points.dtype, np.float32)
        self.assertAlmostEqual(float(points[0, 0]), landmarks[11].x, places=6)     # posición = landmark - 11
        self.assertAlmostEqual(float(points[21, 1]), landmarks[32].y, places=6)


class FakeLandmarker:
    def __init__(self, landmarks=None):
        self.landmarks = landmarks
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return type("R", (), {"pose_landmarks": [self.landmarks] if self.landmarks else []})()

    def close(self):
        self.closed = True


class MediaPipeExtractorTest(unittest.TestCase):
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    def test_no_pose_gives_all_nan(self):
        extractor = MediaPipeFrontDeadliftExtractor("x.task", landmarker=FakeLandmarker())
        result = extractor.extract(self.image, 1.0)
        self.assertEqual(result.points.shape, (25, 3))
        self.assertTrue(np.isnan(result.points).all())

    def test_pose_gives_25_points(self):
        landmarks = random_landmarks(np.random.default_rng(2))
        extractor = MediaPipeFrontDeadliftExtractor("x.task", landmarker=FakeLandmarker(landmarks))
        np.testing.assert_array_equal(extractor.extract(self.image, 1.0).points, landmarks_to_points(landmarks))

    def test_timestamps_are_strictly_increasing_even_if_clock_repeats(self):
        landmarker = FakeLandmarker()
        extractor = MediaPipeFrontDeadliftExtractor("x.task", landmarker=landmarker)
        for stamp in (1.0, 1.0, 1.0004, 0.5, None):
            extractor.extract(self.image, stamp)
        self.assertEqual(landmarker.timestamps, sorted(set(landmarker.timestamps)))
        self.assertEqual(len(landmarker.timestamps), 5)

    def test_close_closes_the_landmarker(self):
        landmarker = FakeLandmarker()
        MediaPipeFrontDeadliftExtractor("x.task", landmarker=landmarker).close()
        self.assertTrue(landmarker.closed)


class FakeExtractor:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.batches = []
        self.closed = False

    def extract_many(self, images, timestamps=None):
        self.batches.append(len(images))
        points = np.full((self.skeleton.num_points, 3), 0.5, dtype=np.float32)
        return [KeypointResult(points.copy()) for _ in images]

    def close(self):
        self.closed = True


class StageTest(unittest.TestCase):
    def stage(self, exercise):
        config = cfg.load_config(exercise=exercise, check_files=False)
        created = []

        def factory(skeleton):
            extractor = FakeExtractor(skeleton)
            created.append(extractor)
            return extractor

        return KeypointStage(config, factory), created

    def images(self):
        return {view: np.zeros((48, 64, 3), dtype=np.uint8) for view in VIEWS}

    def test_lateral_views_share_one_model_and_one_batched_call(self):
        for exercise in ("squat", "bench"):
            stage, created = self.stage(exercise)
            self.assertEqual(len(created), 2, exercise)             # frontal + un lateral compartido
            results = stage.process(self.images())
            self.assertEqual(set(results), set(VIEWS))
            self.assertEqual(sorted(sum((e.batches for e in created), [])), [1, 2], exercise)

    def test_deadlift_front_uses_its_own_mediapipe_extractor(self):
        stage, created = self.stage("deadlift")
        self.assertEqual(sorted(e.skeleton.name for e in created), ["dl_front", "dl_side"])
        results = stage.process(self.images())
        self.assertEqual(results["front"].points.shape, (25, 3))
        self.assertEqual(results["left"].points.shape, (2, 3))

    def test_processes_only_the_views_received(self):
        stage, created = self.stage("squat")
        images = self.images()
        results = stage.process({"left": images["left"]})
        self.assertEqual(set(results), {"left"})
        self.assertEqual(sorted(sum((e.batches for e in created), [])), [1])

    def test_annotate_returns_copies(self):
        stage, _ = self.stage("squat")
        images = self.images()
        annotated = stage.annotate(images, stage.process(images))
        self.assertEqual(set(annotated), set(VIEWS))
        self.assertTrue(all(img.sum() == 0 for img in images.values()))          # originales intactos
        self.assertTrue(all(img.sum() > 0 for img in annotated.values()))

    def test_close_closes_all_extractors(self):
        stage, created = self.stage("bench")
        stage.close()
        self.assertTrue(all(e.closed for e in created))


class DrawingTest(unittest.TestCase):
    def skeleton(self, name):
        config = cfg.load_config(check_files=False)
        for exercise in config.exercises.values():
            for view in exercise.values():
                if view.skeleton.name == name:
                    return view.skeleton

    def test_draws_visible_points_and_connection(self):
        skeleton = self.skeleton("dl_side")
        points = np.array([[0.25, 0.5, 0.9], [0.75, 0.5, 0.9]], dtype=np.float32)
        image = draw_skeleton(np.zeros((100, 200, 3), dtype=np.uint8), KeypointResult(points), skeleton)
        self.assertEqual(image[50, 50].tolist(), [0, 255, 0])        # punto 0
        self.assertEqual(image[50, 150].tolist(), [0, 255, 0])       # punto 1
        self.assertEqual(image[50, 100].tolist(), [255, 0, 0])       # línea entre ambos

    def test_low_confidence_points_and_their_lines_are_hidden(self):
        skeleton = self.skeleton("dl_side")
        points = np.array([[0.25, 0.5, 0.9], [0.75, 0.5, 0.2]], dtype=np.float32)
        image = draw_skeleton(np.zeros((100, 200, 3), dtype=np.uint8), KeypointResult(points), skeleton)
        self.assertEqual(image[50, 150].tolist(), [0, 0, 0])
        self.assertEqual(image[50, 100].tolist(), [0, 0, 0])

    def test_mediapipe_measure_point_is_never_drawn(self):
        skeleton = self.skeleton("dl_front")
        points = empty_points(25)
        points[24] = [0.5, 0.5, 1.0]
        image = draw_skeleton(np.zeros((100, 100, 3), dtype=np.uint8), KeypointResult(points), skeleton)
        self.assertEqual(int(image.sum()), 0)

    def test_empty_result_draws_nothing(self):
        skeleton = self.skeleton("sq_front")
        image = draw_skeleton(np.zeros((50, 50, 3), dtype=np.uint8), KeypointResult(empty_points(16)), skeleton)
        self.assertEqual(int(image.sum()), 0)


class RealModelsTest(unittest.TestCase):
    """Carga los modelos reales y los ejecuta sobre un frame negro.

    Solo se comprueba formato y ejecución: sobre un frame vacío bp_side_skeleton_v2 llega a
    "detectar" una persona (confianza de caja 0.38 > 0.25 por defecto), igual que durante la
    extracción de los datos de entrenamiento, así que la ausencia de detección no se exige.
    """

    def test_every_exercise_runs_end_to_end_on_a_blank_frame(self):
        for exercise in cfg.EXERCISES:
            config = cfg.load_config(exercise=exercise, check_files=False)
            if cfg.missing_files(config):
                self.skipTest("Modelos locales ausentes en esta copia del repositorio")
            with KeypointStage(config) as stage:
                images = {view: np.zeros((480, 640, 3), dtype=np.uint8) for view in VIEWS}
                results = stage.process(images, {view: 1.0 for view in VIEWS})
                for view in VIEWS:
                    expected = config.view(view).skeleton.num_points
                    self.assertEqual(results[view].points.shape, (expected, 3), (exercise, view))
                    self.assertEqual(results[view].points.dtype, np.float32)
                self.assertEqual(set(stage.annotate(images, results)), set(VIEWS))


if __name__ == "__main__":
    unittest.main()
