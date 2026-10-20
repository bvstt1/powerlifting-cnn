"""Pruebas de grid, abstención y aislamiento de cohortes."""
import unittest
from pathlib import Path

import numpy as np

from .data import T_TARGET, RELATION_EXCLUDED, cohort, paths, prepare, resample


class TemporalLateral(unittest.TestCase):
    def test_source_cohorts_and_pose_historical_duplicate(self):
        for view,ntr,nval in (("left",422,91),("right",431,93)):
            groups,_,_,src=cohort(view)
            self.assertEqual((len(groups["train"]),len(groups["validation"])),(ntr,nval))
            self.assertEqual(src["test_npz_opened"],0)
            self.assertNotIn("bp_354",{r["video_id"] for r in groups["train"]} if view=="right" else set())
            if view=="right":self.assertTrue(src["pose_compatibility"]["right_historical_train_includes_bp_354"])

    def test_ausencias_y_proyeccion_no_se_convierten_en_observacion(self):
        names=["bar_cx_w","sep_signed_h","bar_vy_h_s","q_bar_local_association","q_relation_direct"]
        x=np.array([[1.,.1,np.nan,0.,0.],[np.nan,np.nan,np.nan,0.,0.],[2.,.2,np.nan,0.,1.]],np.float32)
        a=np.isfinite(x).astype(np.uint8)
        origin=np.where(a,1,0).astype(np.uint8)
        origin[0,1]=origin[2,1]=4
        z={"features":x,"availability":a,"provenance":origin,
           "bar_state":np.array([2,0,2]),"profile_state":np.array([1,1,1]),
           "relation_state":np.array([2,0,1]),"raw_class_counts":np.zeros((3,3),np.int16),
           "raw_instance_ids":np.full((3,3),-1,np.int32),"frame_index":np.arange(3),
           "feature_names":np.asarray(names)}
        y,m,p=resample(z,names)
        self.assertEqual(y.shape,(T_TARGET,5))
        self.assertEqual(m[0,1],1);self.assertEqual(p[0,1],4)
        self.assertFalse(m[1:-1,1].any())
        self.assertFalse(m[:,2].any())
        self.assertTrue(np.array_equal(m,np.isfinite(y)))

    def test_real_train_provenance(self):
        p=paths("right");groups,names,_,_=cohort("right")
        vid="bp_020"
        self.assertIn(vid,{r["video_id"] for r in groups["train"]})
        with np.load(p["source"]/"sequences"/f"{vid}_temporal.npz",allow_pickle=False) as z:
            vals,mask,orig=resample(z,names)
        j=names.index("sep_signed_h")
        self.assertEqual(vals.shape,(170,62))
        self.assertTrue(np.isin(orig[:,j],[0,2,3,4,5]).all())
        self.assertFalse(np.isinf(vals).any())
        self.assertTrue(np.array_equal(mask,np.isfinite(vals)))

    def test_imputacion_ajustada_solo_train_y_ablacion_sin_proxy(self):
        names=["bar_cx_w","sep_signed_h","bar_area_frame"]
        train=np.array([[[1.,.3,1.],[3.,np.nan,2.]]],dtype=np.float32)
        val=np.array([[[1000000.,.4,2.],[np.nan,.5,3.]]],dtype=np.float32)
        data={"train":(train,np.isfinite(train).astype(np.uint8),
                       np.where(np.isfinite(train),1,0).astype(np.uint8),np.array([0]),["t"]),
              "validation":(val,np.isfinite(val).astype(np.uint8),
                            np.where(np.isfinite(val),1,0).astype(np.uint8),np.array([1]),["v"])}
        _,full=prepare(data,names,"full")
        _,without=prepare(data,names,"sin_relacion")
        self.assertEqual(full["train_statistics"]["bar_cx_w"]["median"],2.)
        self.assertEqual(full["train_statistics"]["bar_cx_w"]["mean"],2.)
        self.assertNotIn("sep_signed_h",without["selected_feature_names"])
        self.assertTrue(all("sep_signed_h" not in c for c in without["input_channel_names"]))
        self.assertIn("sep_signed_h",RELATION_EXCLUDED)


if __name__=="__main__":unittest.main()
