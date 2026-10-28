"""Pruebas de judge/config.yaml y judge/config.py.

python -m unittest judge.tests.test_config
"""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from judge import config as cfg


def raw_config():
    return yaml.safe_load(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def load_raw(raw, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        return cfg.load_config(path, **kwargs)


class DefaultConfigTest(unittest.TestCase):
    def test_structure_covers_three_exercises_and_views(self):
        config = cfg.load_config(check_files=False)
        self.assertEqual(config.exercise, "bench")
        for exercise in cfg.EXERCISES:
            self.assertEqual(tuple(config.exercises[exercise]), cfg.VIEWS)

    def test_lateral_views_share_skeleton_but_not_evaluator(self):
        config = cfg.load_config(check_files=False)
        for exercise, views in config.exercises.items():
            self.assertEqual(views["left"].skeleton.name, views["right"].skeleton.name, exercise)
            self.assertNotEqual(views["left"].evaluator.name, views["right"].evaluator.name, exercise)

    def test_only_bench_has_segmentation(self):
        config = cfg.load_config(check_files=False)
        for exercise, views in config.exercises.items():
            for view in views.values():
                self.assertEqual(view.segmentation is not None, exercise == "bench", exercise)
        bench = config.exercises["bench"]
        self.assertEqual(bench["left"].segmentation.name, bench["right"].segmentation.name)

    def test_input_points_match_trained_networks(self):
        views = cfg.load_config(check_files=False).exercises
        expected = {("squat", "front"): 10, ("squat", "left"): 5, ("squat", "right"): 5,
                    ("bench", "front"): 6, ("bench", "left"): 8, ("bench", "right"): 8,
                    ("deadlift", "front"): 25, ("deadlift", "left"): 2, ("deadlift", "right"): 2}
        for (exercise, view), points in expected.items():
            self.assertEqual(views[exercise][view].input_points(), points, (exercise, view))

    def test_fusion_rule(self):
        config = cfg.load_config(check_files=False)
        self.assertEqual([config.decide(n) for n in range(4)],
                         ["valid", "manual_review", "invalid", "invalid"])

    def test_exercise_override(self):
        self.assertEqual(cfg.load_config(exercise="squat", check_files=False).exercise, "squat")

    def test_referenced_files_exist(self):
        for exercise in cfg.EXERCISES:
            config = cfg.load_config(exercise=exercise, check_files=False)
            absent = cfg.missing_files(config)
            if absent and not (cfg.ROOT / "models").is_dir():
                self.skipTest("Modelos locales ausentes en esta copia del repositorio")
            self.assertEqual(absent, [], exercise)


class InvalidConfigTest(unittest.TestCase):
    def assertRejected(self, raw, fragment):
        with self.assertRaises(cfg.ConfigError) as context:
            load_raw(raw, check_files=False)
        self.assertIn(fragment, str(context.exception))

    def test_repeated_camera_index(self):
        raw = raw_config()
        raw["capture"]["cameras"]["left"] = raw["capture"]["cameras"]["front"]
        self.assertRejected(raw, "índices repetidos")

    def test_fusion_must_cover_all_null_counts(self):
        raw = raw_config()
        del raw["fusion"]["decisions_by_null_count"][3]
        self.assertRejected(raw, "debe cubrir exactamente")

    def test_fusion_rejects_unknown_decision(self):
        raw = raw_config()
        raw["fusion"]["decisions_by_null_count"][0] = "maybe"
        self.assertRejected(raw, "no es una de")

    def test_bench_requires_segmentation(self):
        raw = raw_config()
        del raw["exercises"]["bench"]["views"]["front"]["segmentation"]
        self.assertRejected(raw, "exige 'segmentation'")

    def test_squat_rejects_segmentation(self):
        raw = raw_config()
        raw["exercises"]["squat"]["views"]["front"]["segmentation"] = "bp_front"
        self.assertRejected(raw, "solo el press de banca")

    def test_lateral_views_cannot_use_different_skeletons(self):
        raw = raw_config()
        raw["exercises"]["squat"]["views"]["right"]["skeleton"] = "sq_front"
        self.assertRejected(raw, "deben compartir el modelo de skeleton")

    def test_lateral_views_cannot_share_evaluator(self):
        raw = raw_config()
        raw["exercises"]["squat"]["views"]["right"]["evaluator"] = "sq_left"
        self.assertRejected(raw, "propia red de evaluación")

    def test_unknown_reference(self):
        raw = raw_config()
        raw["exercises"]["squat"]["views"]["front"]["evaluator"] = "nope"
        self.assertRejected(raw, "no existe en evaluators")

    def test_subset_beyond_points(self):
        raw = copy.deepcopy(raw_config())
        raw["evaluators"]["sq_front"]["subset_indices"] = [0, 99]
        self.assertRejected(raw, "excede")

    def test_segmentation_inference_parameters(self):
        raw = raw_config()
        raw["segmentation_models"]["bp_side"]["conf"] = 1.5
        self.assertRejected(raw, "segmentation_models.bp_side.conf")
        views = cfg.load_config(check_files=False).views()
        self.assertEqual((views["front"].segmentation.imgsz, views["front"].segmentation.conf,
                          views["front"].segmentation.iou), (960, 0.25, 0.70))

    def test_controls_and_pipeline_sections(self):
        config = cfg.load_config(check_files=False)
        self.assertEqual(config.key_codes(), {"start_stop": 32, "cancel": ord("c"), "quit": ord("q")})
        self.assertEqual(config.capture.process_fps, 2.0)
        self.assertFalse(config.pipeline.preview_while_idle)
        self.assertEqual(config.pipeline.log_directory, cfg.ROOT / "judge" / "runs")

    def test_keys_must_be_distinct_and_known(self):
        raw = raw_config()
        raw["controls"]["cancel"] = "space"
        self.assertRejected(raw, "tecla distinta")
        raw = raw_config()
        raw["controls"]["quit"] = "escape"
        self.assertRejected(raw, "Tecla desconocida")

    def test_process_fps_must_be_positive(self):
        raw = raw_config()
        raw["capture"]["process_fps"] = 0
        self.assertRejected(raw, "process_fps")

    def test_unverified_network_must_explain_why(self):
        raw = raw_config()
        del raw["evaluators"]["dl_front"]["verification_note"]
        self.assertRejected(raw, "verification_note")

    def test_allow_unverified_must_be_boolean(self):
        raw = raw_config()
        raw["evaluation"]["allow_unverified"] = "si"
        self.assertRejected(raw, "allow_unverified")

    def test_connection_index_out_of_range(self):
        raw = raw_config()
        raw["skeleton_models"]["dl_side"]["connections"] = [[0, 5]]
        self.assertRejected(raw, "connections")

    def test_mediapipe_measure_point_is_not_drawable(self):
        raw = raw_config()
        raw["skeleton_models"]["dl_front"]["connections"].append([4, 24])
        self.assertRejected(raw, "connections")

    def test_unknown_exercise(self):
        raw = raw_config()
        raw["exercise"] = "clean"
        self.assertRejected(raw, "no es uno de")

    def test_missing_file_is_reported(self):
        raw = raw_config()
        raw["skeleton_models"]["bp_front"]["model"] = "models/bp/no_existe.pt"
        with self.assertRaises(cfg.ConfigError) as context:
            load_raw(raw, check_files=True)
        self.assertIn("no_existe.pt", str(context.exception))


if __name__ == "__main__":
    unittest.main()
