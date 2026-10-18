"""Pruebas de folds disjuntos y scaler condicionado solo a inner Train."""
import unittest

import numpy as np

from training.bp_side_segcnn_v1.data import cohort, csv_rows
from .protocol import folds, frozen_contract, seg_fold_transform


class CVProtocol(unittest.TestCase):
    def test_real_development_folds_and_exclusion(self):
        for view,n in (("left",513),("right",524)):
            groups,_,p,_=cohort(view)
            ids=[r["video_id"] for role in ("train","validation") for r in groups[role]]
            labels={r["video_id"]:int(r["label"]) for r in csv_rows(p["labels"]) if r["video_id"] in set(ids)}
            y=np.array([labels[vid] for vid in ids])
            result=folds(ids,y,view)
            self.assertEqual(len(result),5)
            self.assertEqual(sum(len(f["outer_eval"]) for f in result),n)
            self.assertTrue(all(len(f["inner_validation"])>=50 for f in result))
            if view=="right":self.assertNotIn("bp_354",ids)

    def test_duplicate_hash_blocks_fold_construction(self):
        ids=[f"bp_{i:03d}" for i in range(1,514)]
        labels=np.array([i%2 for i in range(513)])
        hashes={vid:vid for vid in ids};hashes[ids[1]]=hashes[ids[0]]
        with self.assertRaises(RuntimeError):folds(ids,labels,"left",hashes)

    def test_outer_extreme_never_changes_seg_inner_stats_or_topology(self):
        contract=frozen_contract("left")
        x=np.zeros((3,170,62),np.float32)
        a=np.zeros_like(x,np.uint8);prov=np.zeros_like(x,np.uint8)
        feature=contract["selected_feature_indices"][0]
        x[0,:,feature]=1;x[1,:,feature]=3;x[2,:,feature]=1000000.
        a[:,:,feature]=1;prov[:,:,feature]=1
        train,outer,stats=seg_fold_transform(x,a,prov,contract,np.array([0,1]),np.array([2]))
        self.assertEqual((train.shape,outer.shape),((2,138,170),(1,138,170)))
        self.assertEqual(stats[0]["mean"],2.)
        self.assertEqual(stats[0]["median"],2.)
        self.assertEqual(stats[0]["n_observed_inner_train"],340)
        self.assertTrue(np.isfinite(outer).all())


if __name__=="__main__":unittest.main()
