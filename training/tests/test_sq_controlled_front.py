"""Regresiones de normalización, aislamiento y scaler controlado SQ front."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.sq import controlled_front as controlled
from training.sq import train_posecnn as sq


class ControlledFrontTests(unittest.TestCase):
    def sample(self):
        arr = np.zeros((20, 16, 3), dtype=np.float32)
        arr[..., 0] = 0.4
        arr[..., 1] = 0.4
        arr[..., 2] = .9
        arr[:, 0, 0], arr[:, 1, 0] = .25, .55
        arr[:, 4, 1], arr[:, 9, 1] = .6, .85
        arr[:, 5, 1], arr[:, 6, 1] = .6, .85
        arr[:, 2, 1] = np.linspace(.4, .7, len(arr))
        return arr

    def test_tibia_scale_and_center_fixed(self):
        original = self.sample()
        result, center, scale = sq.preprocess_front(original, target_frames=96, scale_mode="shanks")
        self.assertEqual(result.shape, (96, 16, 3))
        self.assertAlmostEqual(scale, .25, places=5)
        self.assertAlmostEqual(center[0], .4, places=5)
        self.assertGreater(result[-1, 2, 1] - result[0, 2, 1], .5)
        np.testing.assert_allclose(result[..., 2], .9, atol=1e-6)
        # Un par ausente no debe forzar que se use una distancia inventada.
        original[:, 4, :2] = np.nan
        with self.assertRaises(ValueError):
            sq.preprocess_front(original, scale_mode="shanks")

    def test_scaler_confidence_and_train_only(self):
        base, _, _ = sq.preprocess_front(self.sample())
        x = sq.apply_subset(np.stack([base + i * .01 for i in range(6)]).astype(np.float32), controlled.IDX)
        x[..., 2] = .9
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            train, labels, val, scaler = controlled.scale("D", x[:4], np.array([0,1,0,1]), x[4:], 42,
                                                         Path(temp) / "scaler.pkl")
            self.assertEqual(scaler.n_samples_seen_, 4)
            self.assertEqual(train.shape, (16, 170, 10, 3))
            np.testing.assert_allclose(train[..., 2], .9, atol=1e-6)
            np.testing.assert_allclose(val[..., 2], .9, atol=1e-6)
            np.testing.assert_array_equal(labels, np.tile([0,1,0,1], 4).reshape(4,4).T.flatten())

    def test_reference_membership_and_params(self):
        model = sq.PoseCNN(T=170, K=10, C=3)
        model_96 = sq.PoseCNN(T=96, K=10, C=3)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in model_96.parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 102017)
        manifest, prov = controlled.reference()
        with np.load(prov / "split.npz") as split:
            ids = np.concatenate([split["ids_train"],split["ids_validation"]]).astype(str)
            y = np.concatenate([split["y_train"],split["y_validation"]])
        self.assertEqual(len(list(controlled.fold_indices(ids, y, prov))), 5)
        self.assertEqual(set(ids) & set(manifest["split_ids"]["test"]), set())


if __name__ == "__main__":
    unittest.main()
