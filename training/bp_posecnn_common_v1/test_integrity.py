"""Pruebas de contrato monovista y adaptador sin modificar pesos históricos."""
import unittest

import numpy as np
import torch

from training.bp.train_posecnn import preprocess
from training.common.train_utils import PoseCNN
from .core import PoseEmbeddingAdapter, get_members


class PoseLateralCompatibility(unittest.TestCase):
    def test_cohort_exact_membership(self):
        left,*_=get_members("left")
        right,*_=get_members("right")
        self.assertEqual([len(left[s]) for s in ("train","validation")],[422,91])
        self.assertEqual([len(right[s]) for s in ("train","validation")],[431,93])
        self.assertIn("bp_027",{r["video_id"] for r in right["train"]})
        self.assertNotIn("bp_354",{r["video_id"] for r in right["train"]})

    def test_embeddings_restore_original_probability(self):
        torch.manual_seed(42)
        model=PoseCNN(T=170,K=8,C=3).eval()
        adapter=PoseEmbeddingAdapter(model).eval()
        x=torch.rand(2,3,170,8)
        with torch.no_grad():
            z=adapter.extract_embeddings(x)
            p=model(x)
            recovered=adapter.classifier_from_embeddings(z)
        self.assertEqual(tuple(z.shape),(2,64))
        self.assertTrue(torch.equal(p,recovered))

    def test_historical_pose_gaps_are_bounded(self):
        sequence=np.zeros((12,8,3),np.float32)
        sequence[:,:,0]=np.arange(12)[:,None]*.001+.5
        sequence[:,:,1]=.4
        sequence[:,2,1]=.6  # cadera frente a hombro índice 3
        sequence[:,:,2]=.9
        sequence[4:6,:,:]=0
        result,center,scale=preprocess(sequence)
        self.assertEqual(result.shape,(170,8,3))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(scale,0)
        sequence[2:9,:,:]=0
        with self.assertRaises(ValueError):
            preprocess(sequence)


if __name__=="__main__":unittest.main()
