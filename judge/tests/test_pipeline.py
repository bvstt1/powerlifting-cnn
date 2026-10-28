"""Pruebas del pipeline principal, la fuente de video, la interfaz y el punto de entrada.

python -m unittest judge.tests.test_pipeline
"""

import dataclasses
import json
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

from judge import config as cfg
from judge import main as judge_main
from judge.attempt_log import AttemptLog
from judge.capture import CameraError, Frame, FrameSet
from judge.capture.video_source import VideoSource
from judge.config import VIEWS, PipelineConfig
from judge.evaluation import (INSUFFICIENT, INVALID, VALID, LiftEvaluator, RecorderError, ViewVerdict)
from judge.fusion import FusionRule
from judge.keypoints import KeypointResult
from judge.pipeline import JudgePipeline, PipelineError
from judge.ui import compose_mosaic, status_text

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
POINTS = {"squat": {"front": 16, "left": 5, "right": 5}}
SIZE = (48, 64)


def frame_set(index=0, value=0):
    return FrameSet({view: Frame(view, np.full((*SIZE, 3), value, dtype=np.uint8), index / 30, index) for view in VIEWS})


class FakeKeypoints:
    def __init__(self, delay=0.0, fail_on=()):
        self.delay, self.fail_on = delay, set(fail_on)
        self.calls = 0
        self.closed = False

    def process(self, images, timestamps=None):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("fallo de extracción simulado")
        time.sleep(self.delay)
        return {view: KeypointResult(np.full((POINTS["squat"][view], 3), 0.5, dtype=np.float32)) for view in images}

    def annotate(self, images, results, **kwargs):
        return {view: image + 1 for view, image in images.items()}

    def close(self):
        self.closed = True


class FakeSegmentation:
    def __init__(self):
        self.closed = False

    def process(self, images):
        return {view: None for view in images}

    def annotate(self, images, results, **kwargs):
        return {view: image + 10 for view, image in images.items()}

    def close(self):
        self.closed = True


class ScriptedViewEvaluator:
    def __init__(self, view, labels):
        self.view, self.labels = view, labels

    def evaluate(self, raw):
        label = self.labels[self.view]
        return ViewVerdict(self.view, label, None if label == INSUFFICIENT else (0.9 if label == VALID else 0.1),
                           "sin datos" if label == INSUFFICIENT else None, len(raw))


def make_pipeline(directory, labels=None, preview=False, threaded=False, min_interval=0.0, keypoints=None,
                  segmentation=None, log=True, clock=time.perf_counter):
    labels = labels or {view: VALID for view in VIEWS}
    base = cfg.load_config(exercise="squat", check_files=False)
    config = dataclasses.replace(base, pipeline=PipelineConfig(preview, Path(directory)))
    evaluator = LiftEvaluator(config, evaluator_factory=lambda c, view, device: ScriptedViewEvaluator(view, labels))
    return JudgePipeline(config, keypoints or FakeKeypoints(), evaluator, FusionRule.from_config(config), segmentation,
                         AttemptLog(directory, config) if log else None, threaded, min_interval, clock), config


class LiftCycleTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_frames_are_ignored_until_the_judge_starts_the_lift(self):
        pipeline, _ = make_pipeline(self.folder.name)
        self.assertFalse(pipeline.offer(frame_set()))
        self.assertEqual(pipeline.keypoints.calls, 0)
        pipeline.start_lift()
        self.assertEqual(pipeline.state, "recording")
        self.assertTrue(pipeline.offer(frame_set(1)))
        self.assertEqual(pipeline.frame_counts(), {"front": 1, "left": 1, "right": 1})

    def test_finish_returns_the_fused_decision_and_logs_the_attempt(self):
        pipeline, config = make_pipeline(self.folder.name, {"front": VALID, "left": VALID, "right": INVALID})
        pipeline.start_lift()
        for i in range(6):
            pipeline.offer(frame_set(i))
        attempt = pipeline.finish_lift(mosaic=np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(attempt.decision.decision, "manual_review")          # exactamente 1 nulo
        self.assertEqual(attempt.recording.frames, {"front": 6, "left": 6, "right": 6})
        self.assertFalse(pipeline.recording)
        record = json.loads((attempt.path / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(record["decision"], "manual_review")
        self.assertEqual(record["views"]["right"]["label"], INVALID)
        self.assertAlmostEqual(record["views"]["front"]["probability"], 0.9)
        self.assertEqual(record["recording"]["frames"]["left"], 6)
        self.assertEqual(record["networks"]["front"]["name"], "sq_front")
        self.assertEqual(len(record["networks"]["front"]["checkpoint_sha256"]), 64)
        self.assertEqual(np.load(attempt.path / "keypoints_left.npy").shape, (6, 5, 3))
        self.assertTrue((attempt.path / "config.yaml").is_file())
        self.assertTrue((attempt.path / "last_frames.jpg").is_file())

    def test_each_attempt_gets_its_own_folder(self):
        pipeline, _ = make_pipeline(self.folder.name)
        paths = []
        for _ in range(3):
            pipeline.start_lift()
            pipeline.offer(frame_set())
            paths.append(pipeline.finish_lift().path)
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len(list(Path(self.folder.name).iterdir())), 3)

    def test_an_insufficient_view_sends_the_attempt_to_manual_review(self):
        pipeline, _ = make_pipeline(self.folder.name, {"front": VALID, "left": VALID, "right": INSUFFICIENT})
        pipeline.start_lift()
        pipeline.offer(frame_set())
        decision = pipeline.finish_lift().decision
        self.assertEqual(decision.decision, "manual_review")
        self.assertEqual(decision.insufficient_views, ("right",))

    def test_cancel_discards_the_lift(self):
        pipeline, _ = make_pipeline(self.folder.name)
        pipeline.start_lift()
        pipeline.offer(frame_set())
        pipeline.cancel_lift()
        self.assertEqual(pipeline.state, "idle")
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])
        pipeline.start_lift()
        self.assertEqual(pipeline.frame_counts(), {"front": 0, "left": 0, "right": 0})

    def test_invalid_transitions(self):
        pipeline, _ = make_pipeline(self.folder.name)
        with self.assertRaises(PipelineError):
            pipeline.finish_lift()
        pipeline.start_lift()
        with self.assertRaises(RecorderError):
            pipeline.start_lift()

    def test_without_log_nothing_is_written(self):
        pipeline, _ = make_pipeline(self.folder.name, log=False)
        pipeline.start_lift()
        pipeline.offer(frame_set())
        self.assertIsNone(pipeline.finish_lift().path)
        self.assertEqual(list(Path(self.folder.name).iterdir()), [])

    def test_extraction_failures_are_counted_and_do_not_stop_the_lift(self):
        pipeline, _ = make_pipeline(self.folder.name, keypoints=FakeKeypoints(fail_on={2, 4}))
        pipeline.start_lift()
        for i in range(5):
            pipeline.offer(frame_set(i))
        attempt = pipeline.finish_lift()
        self.assertEqual(attempt.errors, 2)
        self.assertEqual(attempt.recording.frames["front"], 3)
        self.assertIn("simulado", str(pipeline.last_error))

    def test_preview_while_idle_processes_but_does_not_record(self):
        pipeline, _ = make_pipeline(self.folder.name, preview=True)
        self.assertTrue(pipeline.offer(frame_set()))
        self.assertEqual(pipeline.frame_counts(), {"front": 0, "left": 0, "right": 0})
        self.assertIsNotNone(pipeline.last_annotated)

    def test_rate_cap_and_force(self):
        now = [0.0]
        pipeline, _ = make_pipeline(self.folder.name, min_interval=0.5, clock=lambda: now[0])
        pipeline.start_lift()
        self.assertTrue(pipeline.offer(frame_set()))
        now[0] = 0.2
        self.assertFalse(pipeline.offer(frame_set()))
        self.assertTrue(pipeline.offer(frame_set(), force=True))
        now[0] = 0.8
        self.assertTrue(pipeline.offer(frame_set()))

    def test_annotation_layers_stack_skeleton_then_segmentation(self):
        pipeline, _ = make_pipeline(self.folder.name, segmentation=FakeSegmentation())
        pipeline.start_lift()
        pipeline.offer(frame_set())
        self.assertEqual(int(pipeline.last_annotated["front"][0, 0, 0]), 11)       # +1 esqueleto, +10 barra

    def test_close_closes_every_stage(self):
        segmentation = FakeSegmentation()
        pipeline, _ = make_pipeline(self.folder.name, segmentation=segmentation)
        pipeline.close()
        self.assertTrue(pipeline.keypoints.closed and segmentation.closed)


class ThreadedPipelineTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_busy_worker_skips_frames_instead_of_blocking(self):
        pipeline, _ = make_pipeline(self.folder.name, threaded=True, keypoints=FakeKeypoints(delay=0.3))
        self.addCleanup(pipeline.close)
        pipeline.start_lift()
        started = time.perf_counter()
        results = [pipeline.offer(frame_set(i)) for i in range(5)]
        self.assertLess(time.perf_counter() - started, 0.25)                  # offer() nunca espera al procesamiento
        self.assertTrue(results[0])
        time.sleep(0.1)
        accepted = pipeline.offer(frame_set(9))
        self.assertFalse(accepted)                                           # ocupado procesando el primero
        self.assertGreaterEqual(pipeline.skipped, 1)

    def test_finish_waits_for_the_frame_in_flight(self):
        pipeline, _ = make_pipeline(self.folder.name, threaded=True, keypoints=FakeKeypoints(delay=0.3))
        self.addCleanup(pipeline.close)
        pipeline.start_lift()
        pipeline.offer(frame_set(0))
        time.sleep(0.1)                                                       # el hilo ya está procesando
        attempt = pipeline.finish_lift()
        self.assertEqual(attempt.recording.frames["front"], 1)                # el frame en curso no se pierde

    def test_close_stops_the_worker_thread(self):
        pipeline, _ = make_pipeline(self.folder.name, threaded=True)
        thread = pipeline._thread
        pipeline.close()
        self.assertFalse(thread.is_alive())

    def test_cancel_during_processing_does_not_leak_frames_into_the_next_lift(self):
        pipeline, _ = make_pipeline(self.folder.name, threaded=True, keypoints=FakeKeypoints(delay=0.3))
        self.addCleanup(pipeline.close)
        pipeline.start_lift()
        pipeline.offer(frame_set(0))
        time.sleep(0.1)
        pipeline.cancel_lift()
        pipeline.start_lift()
        self.assertEqual(pipeline.frame_counts(), {"front": 0, "left": 0, "right": 0})


def write_video(path, frames, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 255, dtype=np.uint8))
    writer.release()


class VideoSourceTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.paths = {}
        for view, frames in zip(VIEWS, (20, 16, 18)):
            self.paths[view] = Path(self.folder.name) / f"{view}.avi"
            write_video(self.paths[view], frames)

    def test_reads_in_lockstep_and_ends_with_the_shortest_video(self):
        with VideoSource(self.paths) as source:
            sets = []
            while (current := source.read()) is not None:
                sets.append(current)
        self.assertEqual(len(sets), 16)
        self.assertEqual([f.seq for f in sets[3].frames.values()], [3, 3, 3])
        self.assertAlmostEqual(sets[3].frames["front"].timestamp, 0.3, places=6)
        self.assertEqual(sets[0].frames["left"].image.shape, (48, 64, 3))

    def test_stride_skips_frames(self):
        with VideoSource(self.paths, stride=4) as source:
            sets = []
            while (current := source.read()) is not None:
                sets.append(current)
        self.assertEqual([s.frames["front"].seq for s in sets], [0, 4, 8, 12])

    def test_errors(self):
        with self.assertRaises(CameraError):
            VideoSource({"front": self.paths["front"]})
        with self.assertRaises(ValueError):
            VideoSource(self.paths, stride=0)
        missing = {**self.paths, "right": Path(self.folder.name) / "no_existe.avi"}
        with self.assertRaises(CameraError) as context:
            VideoSource(missing).start()
        self.assertIn("right", str(context.exception))
        with self.assertRaises(CameraError):
            VideoSource(self.paths).read()

    def test_batch_run_evaluates_the_whole_video_as_one_lift(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        pipeline, _ = make_pipeline(folder.name)
        messages = []
        with VideoSource(self.paths, stride=2) as source:
            attempt = judge_main.run_batch(pipeline, source, messages.append)
        self.assertEqual(attempt.recording.frames, {"front": 8, "left": 8, "right": 8})
        self.assertEqual(attempt.decision.decision, "valid")
        self.assertTrue(any("DECISIÓN: VÁLIDO" in m for m in messages))


class InteractiveLoopTest(unittest.TestCase):
    """El juez maneja el levantamiento con el teclado."""

    class Source:
        def __init__(self):
            self.index = 0

        def read(self, timeout=None):
            self.index += 1
            return frame_set(self.index)

    def run_keys(self, keys, labels=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        pipeline, config = make_pipeline(folder.name, labels)
        queue, shown, messages = list(keys), [], []
        result = judge_main.run_interactive(config, pipeline, self.Source(), lambda window, mosaic: shown.append(mosaic),
                                            lambda delay: ord(queue.pop(0)), messages.append, max_iterations=len(keys) + 1)
        return result, pipeline, shown, messages

    def test_space_starts_and_ends_the_lift(self):
        attempt, pipeline, shown, messages = self.run_keys(["x", " ", "x", "x", " ", "q"])
        self.assertIsNotNone(attempt)
        self.assertEqual(attempt.decision.decision, "valid")
        self.assertGreaterEqual(attempt.recording.frames["front"], 2)
        self.assertFalse(pipeline.recording)
        self.assertIn("Levantamiento iniciado.", messages)
        self.assertTrue(any("DECISIÓN: VÁLIDO" in m for m in messages))
        self.assertTrue(all(m.ndim == 3 for m in shown))

    def test_c_cancels_without_evaluating(self):
        attempt, pipeline, _, messages = self.run_keys([" ", "x", "c", "q"])
        self.assertIsNone(attempt)
        self.assertIn("Levantamiento cancelado.", messages)
        self.assertFalse(pipeline.recording)

    def test_quit_during_a_lift_discards_it(self):
        attempt, pipeline, _, _ = self.run_keys([" ", "x", "q"])
        self.assertIsNone(attempt)
        self.assertFalse(pipeline.recording)

    def test_cancel_key_does_nothing_outside_a_lift(self):
        _, _, _, messages = self.run_keys(["c", "x", "q"])
        self.assertNotIn("Levantamiento cancelado.", messages)

    def test_a_new_lift_clears_the_previous_decision_banner(self):
        _, _, shown, _ = self.run_keys([" ", "x", " ", "x", " ", "q"])
        heights = {m.shape[0] for m in shown}
        self.assertEqual(len(heights), 2)                       # con y sin banner de decisión


class UiTest(unittest.TestCase):
    def test_mosaic_has_three_tiles_and_optional_banner(self):
        images = {view: np.zeros((*SIZE, 3), dtype=np.uint8) for view in VIEWS}
        controls = cfg.load_config(check_files=False).controls
        plain = compose_mosaic(images, "estado", "idle")
        self.assertEqual(plain.shape[0], 36 + 360)
        self.assertEqual(plain.shape[1], 3 * round(SIZE[1] * 360 / SIZE[0]))
        decision = FusionRule({0: "valid", 1: "manual_review", 2: "invalid", 3: "invalid"}).fuse(
            {v: ViewVerdict(v, VALID, 0.9, None, 10) for v in VIEWS})
        banner = compose_mosaic(images, "estado", "recording", decision)
        self.assertEqual(banner.shape[0], plain.shape[0] + 40 + 32)
        self.assertIn(status_text("squat", "idle", {v: 0 for v in VIEWS}, controls), "SQUAT  en espera   [space] iniciar  [c] cancelar  [q] salir")

    def test_missing_view_image_gives_a_black_tile(self):
        mosaic = compose_mosaic({"front": np.full((*SIZE, 3), 200, dtype=np.uint8)}, "estado")
        self.assertGreater(mosaic.shape[1], 0)

    def test_recording_status_shows_frame_counts(self):
        controls = cfg.load_config(check_files=False).controls
        text = status_text("bench", "recording", {"front": 3, "left": 2, "right": 4}, controls, skipped=7)
        self.assertIn("GRABANDO", text)
        self.assertIn("3/2/4", text)
        self.assertIn("omitidos: 7", text)
        self.assertIn("terminar", text)


class MainTest(unittest.TestCase):
    def test_invalid_config_returns_a_clear_error(self):
        messages = []
        code = judge_main.main(["--config", str(Path(tempfile.gettempdir()) / "no_existe_config.yaml")], messages.append)
        self.assertEqual(code, judge_main.EXIT_CONFIG)
        self.assertTrue(any("Configuración inválida" in m for m in messages))

    def test_missing_video_returns_a_capture_error(self):
        config = cfg.load_config(exercise="squat", check_files=False)
        if cfg.missing_files(config):
            self.skipTest("Modelos locales ausentes")
        messages = []
        code = judge_main.main(["--exercise", "squat", "--videos", "a.mp4", "b.mp4", "c.mp4", "--no-log"], messages.append)
        self.assertEqual(code, judge_main.EXIT_CAPTURE)
        self.assertTrue(any("Error de captura" in m for m in messages))

    def test_unknown_exercise_is_rejected_by_the_parser(self):
        with self.assertRaises(SystemExit):
            judge_main.parse_args(["--exercise", "clean"])

    def test_startup_warning_for_the_unverified_deadlift_front_network(self):
        config = cfg.load_config(exercise="deadlift", check_files=False)
        notes = judge_main.warnings_for(config)
        self.assertEqual(len(notes), 1)
        self.assertIn("front", notes[0])
        self.assertEqual(judge_main.warnings_for(cfg.load_config(exercise="squat", check_files=False)), [])
        allowed = dataclasses.replace(config, allow_unverified=True)
        self.assertEqual(judge_main.warnings_for(allowed), [])


class PlaybackKeypoints:
    """Entrega los keypoints crudos guardados, un frame por llamada, como si los hubiera extraído el esqueleto."""

    def __init__(self, sequences):
        self.sequences = sequences
        self.calls = 0

    def process(self, images, timestamps=None):
        index, self.calls = self.calls, self.calls + 1
        return {view: KeypointResult(self.sequences[view][index]) for view in images}

    def annotate(self, images, results, **kwargs):
        return images

    def close(self):
        pass


class SyntheticSource:
    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def read(self, timeout=None):
        if self.index >= self.frames:
            return None
        self.index += 1
        return frame_set(self.index - 1)


class RealNetworksEndToEndTest(unittest.TestCase):
    """Frames sintéticos + keypoints reales reproducidos + las 3 redes reales + fusión + registro."""

    def test_decision_matches_evaluating_the_raw_sequences_directly(self):
        from judge.tests.test_evaluation import raw_keypoints, saved_probabilities
        config = cfg.load_config(exercise="squat", check_files=False)
        if cfg.missing_files(config):
            self.skipTest("Modelos locales ausentes")
        raws = {}
        for view in VIEWS:
            for video_id in saved_probabilities("squat", view):
                path = raw_keypoints("squat", view, video_id)
                if path.is_file():
                    raws[view] = np.load(path)
                    break
        if set(raws) != set(VIEWS):
            self.skipTest("Sin keypoints crudos locales")
        frames = min(len(r) for r in raws.values())

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = dataclasses.replace(config, pipeline=PipelineConfig(False, Path(folder.name)))
        evaluator = LiftEvaluator(config)
        pipeline = JudgePipeline(config, PlaybackKeypoints(raws), evaluator, FusionRule.from_config(config), None,
                                 AttemptLog(folder.name, config))
        attempt = judge_main.run_batch(pipeline, SyntheticSource(frames), lambda *_: None)

        direct = evaluator.evaluate({view: raws[view][:frames] for view in VIEWS})
        for view in VIEWS:
            self.assertEqual(attempt.decision.verdicts[view].label, direct[view].label)
            self.assertAlmostEqual(attempt.decision.verdicts[view].probability, direct[view].probability, places=7)
        self.assertEqual(attempt.recording.frames, {view: frames for view in VIEWS})
        saved = json.loads((attempt.path / "decision.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["decision"], attempt.decision.decision)
        self.assertTrue(all(n["verified"] for n in saved["networks"].values()))


if __name__ == "__main__":
    unittest.main()
