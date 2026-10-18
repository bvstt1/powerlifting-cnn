"""Regresiones: scaler SQ left y folds congelados sin test."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.sq import controlled_left as left
from training.sq import train_posecnn as sq
from training.sq.controlled_front import PointXYScaler, transform


class LeftControlledTests(unittest.TestCase):
    def test_scaler_confidence_and_training_only(self):
        X=np.zeros((6,170,5,3),dtype=np.float32)
        X[...,0]=np.linspace(0,1,170)[None,:,None]
        X[...,1]=np.linspace(1,0,170)[None,:,None]
        X[...,2]=.87
        X[:, :, 1, 0]+=np.arange(6)[:,None]*.1
        X[:, :, 3, 1]+=np.arange(6)[:,None]*.1
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            train,labels,val,scaler=left.scale_fit(X[:4],np.array([0,1,0,1]),X[4:],43,Path(temp)/"scaler.pkl")
        self.assertIsInstance(scaler,PointXYScaler)
        self.assertEqual(scaler.n_samples_seen_,4)
        self.assertEqual(scaler.mean_.shape,(5,2))
        self.assertEqual(train.shape,(16,170,5,3))
        np.testing.assert_allclose(train[...,2],.87,atol=1e-6)
        np.testing.assert_allclose(val[...,2],.87,atol=1e-6)
        np.testing.assert_array_equal(labels, np.repeat([0,1,0,1],4))

    def test_frozen_folds_and_clases(self):
        manifest=left.frozen()
        with np.load(left.PROVENANCE/"split.npz") as data:
            ids=np.concatenate([data["ids_train"],data["ids_validation"]]).astype(str)
            labels=np.concatenate([data["y_train"],data["y_validation"]])
            self.assertEqual(set(data["y_test"]),{0,1})
            self.assertEqual(data["X_train"].shape[1:],(170,5,3))
        folds=list(left.fold_indices(ids,labels,left.PROVENANCE))
        self.assertEqual(len(folds),5)
        self.assertEqual(sum(len(outer) for _,_,_,outer in folds),len(ids))
        self.assertFalse(set(ids)&set(manifest["split_ids"]["test"]))


if __name__=="__main__":
    unittest.main()
