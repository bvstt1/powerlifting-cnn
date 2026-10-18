"""Controles de interpolacion, canal confianza y particion BP lateral."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training.bp.train_posecnn as pipeline
from training.bp.train_posecnn import FRAMES, preprocess, paths
from training.common.train_utils import augment_dataset


class BPSidePipelineTests(unittest.TestCase):
    def sample(self):
        sequence = np.zeros((20, 8, 3), dtype=np.float32)
        sequence[..., 0] = 0.3
        sequence[..., 1] = 0.3
        sequence[..., 2] = 0.8
        sequence[:, 2, 1] = 0.6
        sequence[:, 3, 1] = 0.4
        sequence[:, 4, 1] = np.linspace(0.4, 0.2, 20)
        return sequence

    def test_preprocess_preserves_motion_and_confidence(self):
        sequence = self.sample()
        sequence[5:7, 4, :2] = np.nan
        sequence[5:7, 4, 2] = np.nan
        normalized, _, scale = preprocess(sequence)
        self.assertEqual(normalized.shape, (FRAMES, 8, 3))
        self.assertAlmostEqual(scale, 0.2, places=5)
        self.assertGreater(normalized[0, 4, 1], normalized[-1, 4, 1])
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(normalized[0, 4, 2], 0.8)
        self.assertEqual(normalized[round(6 * (FRAMES - 1) / 19), 4, 2], 0)

    def test_long_or_boundary_gap_rejected(self):
        for start, end in ((4, 10), (0, 2), (18, 20)):
            with self.subTest(start=start):
                sequence = self.sample()
                sequence[start:end, 4, :2] = np.nan
                with self.assertRaises(ValueError):
                    preprocess(sequence)

    def test_augmentation_changes_geometry_not_confidence(self):
        sequence, _, _ = preprocess(self.sample())
        original = np.stack([sequence])
        augmented, labels = augment_dataset(original, np.asarray([1]), np.random.RandomState(42),
                                             3, transforms=("translation", "scaling", "noise"))
        self.assertEqual(labels.tolist(), [1] * 4)
        np.testing.assert_array_equal(augmented[..., 2], np.repeat(original[..., 2], 4, axis=0))
        self.assertTrue(np.any(augmented[1:, :, :, :2] != original[0, :, :, :2]))

    def test_prepared_splits_have_exclusive_ids(self):
        for view in ("left", "right"):
            path = paths(view)["data"] / "split.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                roles = [set(data[f"ids_{role}"].tolist()) for role in ("train", "validation", "test")]
                self.assertEqual(sum(len(role) for role in roles), len(set.union(*roles)))
                for role in ("train", "validation", "test"):
                    self.assertEqual(data[f"X_{role}"].shape[1:], (FRAMES, 8, 3))
                    self.assertEqual(set(data[f"y_{role}"].tolist()), {0, 1})

    def test_scaler_fit_only_on_original_and_model_smoke(self):
        sequence, _, _ = preprocess(self.sample())
        X = np.stack([sequence + i * 0.01 for i in range(8)]).astype(np.float32)
        y = np.asarray([0, 1] * 4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            augmented, labels, early, scaler = pipeline.scaled_arrays(
                X[:6], y[:6], X[6:], 42, folder / "scaler.pkl")
            self.assertEqual(scaler.n_samples_seen_, 6)
            self.assertEqual(len(augmented), 24)
            self.assertEqual(len(early), 2)
            with patch.object(pipeline, "EPOCHS", 1):
                model, best_epoch, history = pipeline.fit(augmented, labels, early, y[6:], 42, folder)
            self.assertEqual(best_epoch, 1)
            self.assertEqual(len(history), 1)
            metrics, predictions, _ = pipeline.evaluate(model, early, y[6:], 42)
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(metrics["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
