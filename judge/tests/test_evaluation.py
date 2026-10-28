"""Pruebas del módulo de evaluación.

python -m unittest judge.tests.test_evaluation
"""

import csv
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from judge import config as cfg
from judge.config import ROOT, VIEWS
from judge.evaluation import (INSUFFICIENT, INVALID, VALID, EvaluationError, InsufficientData, LiftEvaluator,
                              LiftRecorder, Recording, RecorderError, ViewEvaluator)
from judge.evaluation import recipes
from judge.evaluation.network import build_model, load_scaler
from judge.evaluation.stage import create_view_evaluator
from judge.keypoints import KeypointResult
from training.bp.experiments.bp_front_02_posecnn_common_v1.model import BPFrontPoseCNN

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


def synthetic(num_points, frames=80, seed=0):
    """Secuencia cruda plausible: puntos que oscilan suavemente, confianza alta."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, frames)[:, None]
    base = rng.uniform(0.3, 0.7, size=(1, num_points, 2))
    xy = base + 0.05 * np.sin(t[:, :, None] + rng.uniform(0, 3, size=(1, num_points, 2)))
    conf = np.full((frames, num_points, 1), 0.9)
    return np.concatenate([xy, conf], axis=2).astype(np.float32)


class Fake:
    def __init__(self, points):
        self.points = points


class RecorderTest(unittest.TestCase):
    def recorder(self, **kwargs):
        ticks = iter(range(0, 1000, 5))
        return LiftRecorder({"front": 6, "left": 8, "right": 8}, clock=lambda: float(next(ticks)), **kwargs)

    def frame(self, views=VIEWS):
        points = {"front": 6, "left": 8, "right": 8}
        return {v: Fake(np.full((points[v], 3), 0.5, dtype=np.float32)) for v in views}

    def test_records_between_start_and_stop(self):
        recorder = self.recorder()
        self.assertFalse(recorder.add(self.frame()))               # fuera de grabación se ignora
        recorder.start()
        self.assertTrue(recorder.recording)
        for _ in range(7):
            self.assertTrue(recorder.add(self.frame()))
        recording = recorder.stop()
        self.assertFalse(recorder.recording)
        self.assertEqual(recording.frames, {"front": 7, "left": 7, "right": 7})
        self.assertEqual(recording.sequences["left"].shape, (7, 8, 3))
        self.assertEqual(recording.sequences["left"].dtype, np.float32)
        self.assertEqual(recording.duration_s, 5.0)
        self.assertFalse(recording.truncated)

    def test_views_are_independent(self):
        recorder = self.recorder()
        recorder.start()
        recorder.add(self.frame())
        recorder.add(self.frame(("front",)))
        recorder.add(self.frame(("front", "left")))
        self.assertEqual(recorder.stop().frames, {"front": 3, "left": 2, "right": 1})

    def test_invalid_transitions(self):
        recorder = self.recorder()
        with self.assertRaises(RecorderError):
            recorder.stop()
        with self.assertRaises(RecorderError):
            recorder.cancel()
        recorder.start()
        with self.assertRaises(RecorderError):
            recorder.start()

    def test_cancel_discards_and_next_lift_starts_clean(self):
        recorder = self.recorder()
        recorder.start()
        recorder.add(self.frame())
        recorder.cancel()
        self.assertFalse(recorder.recording)
        recorder.start()
        recorder.add(self.frame())
        self.assertEqual(recorder.stop().frames, {"front": 1, "left": 1, "right": 1})

    def test_empty_recording_has_empty_sequences(self):
        recorder = self.recorder()
        recorder.start()
        recording = recorder.stop()
        self.assertEqual(recording.frames, {"front": 0, "left": 0, "right": 0})
        self.assertEqual(recording.sequences["front"].shape, (0, 6, 3))

    def test_rejects_wrong_point_count_and_unknown_view(self):
        recorder = self.recorder()
        recorder.start()
        with self.assertRaises(RecorderError):
            recorder.add({"front": Fake(np.zeros((5, 3), dtype=np.float32))})
        with self.assertRaises(RecorderError):
            recorder.add({"top": Fake(np.zeros((6, 3), dtype=np.float32))})

    def test_max_frames_truncates_instead_of_failing(self):
        recorder = self.recorder(max_frames=3)
        recorder.start()
        for _ in range(5):
            recorder.add(self.frame())
        recording = recorder.stop()
        self.assertTrue(recording.truncated)
        self.assertEqual(recording.frames["front"], 3)

    def test_requires_all_three_views(self):
        with self.assertRaises(ValueError):
            LiftRecorder({"front": 6})

    def test_recording_validates_dtype_and_shape(self):
        with self.assertRaises(ValueError):
            Recording({"front": np.zeros((3, 6, 3), dtype=np.float64)}, 1.0)


class RecipeTest(unittest.TestCase):
    CASES = [("bp_front", 6, (170, 6, 3)), ("bp_lateral", 8, (170, 8, 3)), ("sq_lateral", 5, (170, 5, 3)),
             ("dl_front", 25, (170, 25, 3)), ("dl_side", 2, (170, 2, 3))]

    def run_recipe(self, recipe, raw, num_points, view="left"):
        evaluator = type("E", (), {"recipe": recipe, "subset_indices": None})()
        skeleton = type("S", (), {"num_points": num_points})()
        return recipes.preprocess(evaluator, view, skeleton, raw)

    def test_output_shape_for_every_recipe(self):
        for recipe, points, shape in self.CASES:
            out = self.run_recipe(recipe, synthetic(points), points)
            self.assertEqual(out.shape, shape, recipe)
            self.assertEqual(out.dtype, np.float32)
            self.assertTrue(np.isfinite(out).all(), recipe)

    def test_sq_front_keeps_only_the_leg_points(self):
        evaluator = type("E", (), {"recipe": "sq_front_legs", "subset_indices": (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)})()
        out = recipes.preprocess(evaluator, "front", type("S", (), {"num_points": 16})(), synthetic(16))
        self.assertEqual(out.shape, (170, 10, 3))

    def test_any_length_is_resampled_to_170_frames(self):
        for frames in (10, 45, 170, 400):
            out = self.run_recipe("bp_lateral", synthetic(8, frames), 8)
            self.assertEqual(out.shape, (170, 8, 3))

    def test_left_view_is_mirrored_in_x_for_deadlift_side(self):
        raw = synthetic(2)
        left = self.run_recipe("dl_side", raw, 2, "left")
        right = self.run_recipe("dl_side", raw, 2, "right")
        np.testing.assert_allclose(left[..., 0], -right[..., 0], atol=1e-6)
        np.testing.assert_allclose(left[..., 1:], right[..., 1:], atol=1e-6)

    def test_insufficient_data(self):
        for recipe, points, _ in self.CASES:
            for name, raw in (("todo NaN", np.full((60, points, 3), np.nan, dtype=np.float32)),
                              ("pocos frames", synthetic(points, 5)),
                              ("forma incorrecta", synthetic(points + 1)),
                              ("infinitos", np.where(np.arange(60)[:, None, None] == 3, np.inf, synthetic(points, 60)))):
                with self.assertRaises(InsufficientData, msg=f"{recipe}: {name}"):
                    self.run_recipe(recipe, raw, points)

    def test_lateral_rejects_missing_boundaries_and_too_many_gaps(self):
        raw = synthetic(8, 60)
        raw[0, 3, :2] = np.nan                                  # falta el punto en el primer frame
        with self.assertRaises(InsufficientData):
            self.run_recipe("bp_lateral", raw, 8)
        raw = synthetic(8, 60)
        raw[20:30, :, :2] = np.nan                              # hueco de 10 frames (máximo 5)
        with self.assertRaises(InsufficientData):
            self.run_recipe("bp_lateral", raw, 8)

    def test_small_gaps_are_interpolated(self):
        raw = synthetic(8, 60)
        raw[20:23, 2, :2] = np.nan
        self.assertTrue(np.isfinite(self.run_recipe("bp_lateral", raw, 8)).all())

    def test_yolo_missing_points_are_treated_as_gaps(self):
        raw = synthetic(8, 60)
        raw[10:12, 4] = 0.0                                     # YOLO: (0, 0) con confianza 0
        self.assertTrue(np.isfinite(self.run_recipe("bp_lateral", raw, 8)).all())


class NetworkLoadingTest(unittest.TestCase):
    def test_scaler_with_wrong_feature_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scaler.pkl"
            scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(20, 100)))
            path.write_bytes(pickle.dumps(scaler))
            with self.assertRaises(EvaluationError) as context:
                load_scaler(path, "standard", 170, 8)
            self.assertIn("4080", str(context.exception))

    def test_unknown_scaler_type_and_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "s.pkl"
            path.write_bytes(pickle.dumps(StandardScaler().fit(np.zeros((3, 2)) + [[0, 1], [1, 2], [2, 3]])))
            with self.assertRaises(EvaluationError):
                load_scaler(path, "minmax", 170, 8)
        with self.assertRaises(EvaluationError):
            build_model("ResNet", 170, 8)
        with self.assertRaises(EvaluationError):
            build_model("BPFrontPoseCNN", 170, 8)

    def test_checkpoint_of_another_architecture_is_rejected(self):
        config = cfg.load_config(exercise="bench", check_files=False)
        if cfg.missing_files(config):
            self.skipTest("Modelos locales ausentes")
        item = config.view("left")
        with tempfile.TemporaryDirectory() as folder:
            wrong = Path(folder) / "front.pth"
            torch.save(BPFrontPoseCNN().state_dict(), wrong)
            evaluator = type(item.evaluator)(**{**item.evaluator.__dict__, "checkpoint": wrong})
            with self.assertRaises(EvaluationError) as context:
                ViewEvaluator(evaluator, "left", item.skeleton)
            self.assertIn("checkpoint", str(context.exception))


def saved_probabilities(exercise, view):
    """Probabilidades que guardó el entrenamiento de cada red, por video."""
    def read(path):
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    if exercise == "deadlift":
        run = "legacy_default" if view == "front" else "baseline_v3_manual_audit"
        path = ROOT / f"training/results/dl/{view}/posecnn/runs/{run}/predictions.csv"
        return {r["video_id"]: float(r.get("probability") or r.get("probabilidad")) for r in read(path)}
    code = {"squat": "sq", "bench": "bp"}[exercise]
    path = ROOT / f"training/results/{code}/{view}/posecnn/runs/source_cmp_v1_yolo/test_predictions.csv"
    return {r["video_id"]: float(r["probability_valid"]) for r in read(path)}


def raw_keypoints(exercise, view, video_id):
    base = ROOT / "keypoints"
    if exercise == "deadlift":
        number = int(video_id.split("_")[1])
        if view == "front":
            return base / "dl/mediapipe/front" / f"{video_id}.npy"
        if view == "right":
            return base / "dl/yolo/right" / f"{video_id}.npy"
        return base / "dl/yolo/side" / f"dl_{number + 413:03d}.npy"        # side M >= 414 es left M - 413
    code = {"squat": "sq", "bench": "bp"}[exercise]
    return base / code / "yolo" / view / f"{video_id}.npy"


VERIFIED = [(e, v) for e in cfg.EXERCISES for v in VIEWS if (e, v) != ("deadlift", "front")]


class TrainingParityTest(unittest.TestCase):
    """El keypoint crudo, pasado por el evaluador, da la misma probabilidad que guardó el entrenamiento."""

    def check(self, exercise, view, tolerance, limit=10):
        config = cfg.load_config(exercise=exercise, check_files=False)
        if cfg.missing_files(config):
            self.skipTest("Modelos locales ausentes")
        evaluator = create_view_evaluator(config, view)
        compared = 0
        for video_id, saved in list(saved_probabilities(exercise, view).items())[:limit]:
            path = raw_keypoints(exercise, view, video_id)
            if not path.is_file():
                continue
            verdict = evaluator.evaluate(np.load(path))
            self.assertNotEqual(verdict.label, INSUFFICIENT, f"{video_id}: {verdict.reason}")
            self.assertAlmostEqual(verdict.probability, saved, delta=tolerance, msg=f"{exercise}/{view}/{video_id}")
            compared += 1
        if compared == 0:
            self.skipTest("Sin keypoints crudos locales")
        return compared

    def test_bench_and_squat_networks_reproduce_training_probabilities(self):
        for exercise, view in VERIFIED:
            if exercise != "deadlift":
                with self.subTest(exercise=exercise, view=view):
                    self.assertGreaterEqual(self.check(exercise, view, 1e-5), 1)

    def test_deadlift_side_networks_reproduce_training_probabilities(self):
        for view in ("left", "right"):
            with self.subTest(view=view):
                self.assertGreaterEqual(self.check("deadlift", view, 1e-3), 1)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.config = cfg.load_config(exercise="squat", check_files=False)
        if cfg.missing_files(self.config):
            self.skipTest("Modelos locales ausentes")
        sample = next(iter(saved_probabilities("squat", "left")))
        path = raw_keypoints("squat", "left", sample)
        if not path.is_file():
            self.skipTest("Sin keypoints crudos locales")
        self.raw = np.load(path)

    def test_threshold_decides_the_label(self):
        item = self.config.view("left")
        low = ViewEvaluator(item.evaluator, "left", item.skeleton, threshold=1e-9)
        high = ViewEvaluator(item.evaluator, "left", item.skeleton, threshold=1.0)
        self.assertEqual(low.evaluate(self.raw).label, VALID)
        self.assertEqual(high.evaluate(self.raw).label, INVALID)

    def test_verdict_for_unusable_sequence_is_insufficient_with_reason(self):
        verdict = create_view_evaluator(self.config, "left").evaluate(np.full((60, 5, 3), np.nan, dtype=np.float32))
        self.assertEqual(verdict.label, INSUFFICIENT)
        self.assertIsNone(verdict.probability)
        self.assertTrue(verdict.reason)
        self.assertFalse(verdict.decided)

    def test_lift_evaluator_end_to_end_through_the_recorder(self):
        """Grabar frame a frame y evaluar da la misma probabilidad que la secuencia completa."""
        lifts = LiftEvaluator(self.config)
        direct = lifts.evaluate({"left": self.raw})["left"]
        recorder = lifts.new_recorder()
        recorder.start()
        for frame in self.raw:
            recorder.add({"left": KeypointResult(frame)})
        recording = recorder.stop()
        verdicts = lifts.evaluate(recording)
        self.assertEqual(verdicts["left"].label, direct.label)
        self.assertAlmostEqual(verdicts["left"].probability, direct.probability, places=7)
        self.assertEqual(verdicts["left"].frames, len(self.raw))
        for view in ("front", "right"):                          # sin frames: no decide, pero no rompe nada
            self.assertEqual(verdicts[view].label, INSUFFICIENT)

    def test_one_failing_view_does_not_affect_the_others(self):
        lifts = LiftEvaluator(self.config)
        verdicts = lifts.evaluate({"left": self.raw, "right": np.full((60, 5, 3), np.nan, dtype=np.float32)})
        self.assertTrue(verdicts["left"].decided)
        self.assertEqual(verdicts["right"].label, INSUFFICIENT)
        self.assertEqual(verdicts["front"].label, INSUFFICIENT)
        self.assertEqual(verdicts["front"].reason, "sin secuencia para esta vista")

    def test_recorder_matches_the_exercise_point_counts(self):
        for exercise, expected in (("squat", (16, 5, 5)), ("bench", (6, 8, 8)), ("deadlift", (25, 2, 2))):
            config = cfg.load_config(exercise=exercise, check_files=False)
            recorder = LiftEvaluator(config, evaluator_factory=lambda *a: None).new_recorder()
            self.assertEqual(tuple(recorder._num_points[v] for v in VIEWS), expected, exercise)


class UnverifiedNetworkTest(unittest.TestCase):
    def setUp(self):
        self.config = cfg.load_config(exercise="deadlift", check_files=False)
        if cfg.missing_files(self.config):
            self.skipTest("Modelos locales ausentes")

    def test_deadlift_front_is_flagged_unverified_with_a_reason(self):
        evaluator = self.config.view("front").evaluator
        self.assertFalse(evaluator.verified)
        self.assertIn("AUC", evaluator.verification_note)
        for view in ("left", "right"):
            self.assertTrue(self.config.view(view).evaluator.verified)

    def test_unverified_network_does_not_issue_a_verdict_by_default(self):
        verdict = create_view_evaluator(self.config, "front").evaluate(synthetic(25, 80))
        self.assertEqual(verdict.label, INSUFFICIENT)
        self.assertIn("no verificada", verdict.reason)

    def test_allow_unverified_lets_the_network_run(self):
        item = self.config.view("front")
        evaluator = ViewEvaluator(item.evaluator, "front", item.skeleton, allow_unverified=True)
        verdict = evaluator.evaluate(synthetic(25, 80))
        self.assertIn(verdict.label, (VALID, INVALID, INSUFFICIENT))
        self.assertNotIn("no verificada", verdict.reason or "")


if __name__ == "__main__":
    unittest.main()
