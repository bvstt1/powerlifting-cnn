"""Controles del pipeline PoseCNN de squat (front K=16, laterales K=5)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training.sq.train_posecnn as pipeline
from training.sq.train_posecnn import FRAMES, paths, preprocess_front, preprocess_left, apply_subset, resolve_subset
from training.common.train_utils import augment_dataset


def make_sample(keypoints):
    sequence = np.zeros((20, keypoints, 3), dtype=np.float32)
    for i in range(keypoints):
        sequence[:, i, 0] = 0.2 + 0.02 * i
        sequence[:, i, 1] = 0.4 + 0.01 * i
    sequence[..., 2] = 0.8
    return sequence


class SQFrontTests(unittest.TestCase):
    def test_shape_and_scale_reference(self):
        sequence = make_sample(16)
        sequence[:, 0, 0] = 0.2
        sequence[:, 0, 1] = 0.4
        sequence[:, 1, 0] = 0.4
        sequence[:, 1, 1] = 0.4
        # Descenso de cadera simulado: la cadera baja respecto a rodillas.
        sequence[:, 2, 1] = np.linspace(0.5, 0.7, 20)
        normalized, _, scale = preprocess_front(sequence)
        self.assertEqual(normalized.shape, (FRAMES, 16, 3))
        self.assertAlmostEqual(scale, 0.2, places=5)
        self.assertTrue(np.isfinite(normalized).all())
        # El movimiento relativo cadera-rodilla se conserva tras normalizar.
        self.assertGreater(normalized[-1, 2, 1], normalized[0, 2, 1])

    def test_gap_and_confidence(self):
        sequence = make_sample(16)
        sequence[5:7, 2, :2] = np.nan
        sequence[5:7, 2, 2] = np.nan
        normalized, _, _ = preprocess_front(sequence)
        self.assertEqual(normalized[0, 2, 2], 0.8)
        self.assertEqual(normalized[round(6 * (FRAMES - 1) / 19), 2, 2], 0)
        bad = make_sample(16)
        bad[4:10, 2, :2] = np.nan
        with self.assertRaises(ValueError):
            preprocess_front(bad)


class SQLeftTests(unittest.TestCase):
    def test_shank_scale_and_hip_drop(self):
        sequence = make_sample(5)
        sequence[:, 0, 0] = 0.3
        sequence[:, 0, 1] = 0.6  # rodilla
        sequence[:, 2, 0] = 0.3
        sequence[:, 2, 1] = 0.9  # tobillo -> tibia = 0.3
        sequence[:, 1, 1] = np.linspace(0.4, 0.55, 20)  # cadera desciende
        normalized, center, scale = preprocess_left(sequence)
        self.assertEqual(normalized.shape, (FRAMES, 5, 3))
        self.assertAlmostEqual(scale, 0.3, places=5)
        self.assertGreater(normalized[-1, 1, 1], normalized[0, 1, 1])
        self.assertTrue(np.isfinite(normalized).all())

    def test_boundary_gap_rejected(self):
        sequence = make_sample(5)
        sequence[0:2, 1, :2] = np.nan
        with self.assertRaises(ValueError):
            preprocess_left(sequence)


class SQSharedTests(unittest.TestCase):
    def test_right_hip_knee_signal_and_confidence_scaler(self):
        from training.sq.controlled_front import PointXYScaler
        sequence = make_sample(5)
        sequence[:, 0, :2] = [0.4, 0.6]
        sequence[:, 2, :2] = [0.4, 0.9]
        sequence[:, 1, 1] = np.linspace(0.4, 0.65, 20)
        normalized, _, scale = pipeline.PREPROCESSORS["right"](sequence)
        self.assertAlmostEqual(scale, 0.3, places=5)
        self.assertAlmostEqual(normalized[0, 1, 1] - normalized[0, 0, 1], (0.4-0.6)/0.3, places=5)
        self.assertAlmostEqual(normalized[-1, 1, 1] - normalized[-1, 0, 1], (0.65-0.6)/0.3, places=5)
        X = np.stack([normalized, normalized + np.array([0.02, 0.01, 0], dtype=np.float32)])
        scaler = PointXYScaler().fit(X[:1])
        np.testing.assert_array_equal(scaler.transform(X)[..., 2], X[..., 2])
        self.assertEqual(scaler.n_samples_seen_, 1)

    def test_augmentation_keeps_confidence(self):
        sequence, _, _ = preprocess_left(make_sample(5))
        original = np.stack([sequence])
        augmented, labels = augment_dataset(original, np.asarray([1]), np.random.RandomState(42),
                                            3, transforms=("translation", "scaling", "noise"))
        self.assertEqual(labels.tolist(), [1] * 4)
        np.testing.assert_array_equal(augmented[..., 2], np.repeat(original[..., 2], 4, axis=0))

    def test_front_legs_subset(self):
        sequence, _, _ = preprocess_front(make_sample(16))
        name, indices = resolve_subset("front", "legs")
        self.assertEqual(name, "legs")
        self.assertEqual(len(indices), 10)
        subset = apply_subset(np.stack([sequence]), indices)
        self.assertEqual(subset.shape, (1, FRAMES, 10, 3))
        np.testing.assert_array_equal(subset[0, :, :, :2], sequence[:, list(indices), :2])
        with self.assertRaises(ValueError):
            resolve_subset("left", "legs")

    def test_scaler_fit_and_smoke(self):
        sequence, _, _ = preprocess_left(make_sample(5))
        X = np.stack([sequence + i * 0.01 for i in range(8)]).astype(np.float32)
        y = np.asarray([0, 1] * 4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            augmented, labels, early, scaler = pipeline.scaled_arrays(
                X[:6], y[:6], X[6:], 42, folder / "scaler.pkl")
            self.assertEqual(scaler.n_samples_seen_, 6)
            self.assertEqual(len(augmented), 24)
            with patch.object(pipeline, "EPOCHS", 1):
                model, best_epoch, history = pipeline.fit(augmented, labels, early, y[6:], 42, folder)
            self.assertEqual(best_epoch, 1)
            metrics, predictions, _ = pipeline.evaluate(model, early, y[6:], 42)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(metrics["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
