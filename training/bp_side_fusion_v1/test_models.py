"""Garantías de forward y pesos fijos de la fusión lateral."""
import unittest

import numpy as np
import torch

from training.bp_front_segcnn_v1.model import SegCNNTemporal
from training.common.train_utils import PoseCNN
from .models import FusionHead, late_fusion, verify_adapters


class FusionContract(unittest.TestCase):
    def test_head_parameters_and_adapters(self):
        torch.set_num_threads(2)
        pose=PoseCNN(T=170,K=8,C=3).eval().requires_grad_(False)
        seg=SegCNNTemporal(90).eval().requires_grad_(False)
        xp=np.random.RandomState(42).randn(2,3,170,8).astype(np.float32)
        xs=np.random.RandomState(1337).randn(2,90,170).astype(np.float32)
        report=verify_adapters(pose,seg,xp,xs,torch.device("cpu"))
        self.assertEqual(report["concat_dimension"],192)
        self.assertEqual(report["head_trainable_parameters"],6209)
        self.assertEqual(tuple(FusionHead()(torch.zeros(2,192)).shape),(2,))
        self.assertFalse(any(p.requires_grad for p in list(pose.parameters())+list(seg.parameters())))

    def test_late_fixed_weight(self):
        p=np.array([.1,.9,.5]);q=np.array([.3,.3,.5])
        self.assertTrue(np.array_equal(late_fusion(p,q),.5*p+.5*q))
        with self.assertRaises(ValueError):late_fusion(p,q[:2])


if __name__=="__main__":unittest.main()
