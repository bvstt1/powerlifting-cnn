"""Pruebas con máscaras sintéticas y un caso RAW de desarrollo existente."""
import unittest
from pathlib import Path

import numpy as np

from . import geometry as g
from . import raw_reader as rr
from .features import extraer
from .schema import F, INDEX


def instancia(mask, x=0, y=0, clase="bar", conf=.8):
    ys, xs=np.nonzero(mask)
    return {"crop":mask.astype(bool),"x":x,"y":y,"class":clase,"conf":conf,
            "box":np.asarray([x,y,x+mask.shape[1],y+mask.shape[0]],float),"area":int(mask.sum()),"id":0}


class GeometriaLateral(unittest.TestCase):
    def setUp(self):
        self.pose=np.full((8,3),np.nan,dtype=float)

    def test_mascara_pendiente_y_reflexion(self):
        m=np.zeros((100,180),bool)
        for x in range(20,160):
            y=30+round(.1*x)
            m[y:y+5,x]=1
        a=g.pendiente(instancia(m,120,200),1080)
        b=g.pendiente(instancia(np.fliplr(m),120,200),1080)
        self.assertIsNotNone(a);self.assertIsNotNone(b)
        self.assertAlmostEqual(a[0],-b[0],delta=.02)
        self.assertIsNone(g.pendiente(instancia(m[:,:25]),1080))

    def test_contorno_pose_y_respaldo_banco(self):
        m=np.ones((80,220),bool)
        p=instancia(m,300,320,"person")
        b=instancia(np.ones((12,110),bool),380,240)
        self.pose[2]=[350/1000,400/600,.95]
        self.pose[3]=[500/1000,350/600,.95]
        perfil,estado=g.torso(p,self.pose,np.array([340,400,540,470]),b,1000,600)
        self.assertEqual(estado,1)
        self.assertAlmostEqual(perfil["y"],320)
        self.pose[:]=np.nan
        perfil,estado=g.torso(p,self.pose,np.array([340,400,540,470]),b,1000,600)
        self.assertEqual(estado,2)
        self.assertIsNone(g.torso(None,self.pose,None,b,1000,600)[0])
        self.assertEqual(g.torso(p,self.pose,None,b,1000,600)[1],0)

    def test_sep_firmada_directa_corta_lejana_abstener(self):
        m=np.ones((8,100),bool);b=instancia(m,100,200)
        for x,state in ((150,1),(240,2),(450,3),(650,0)):
            p={"x":x,"y":240.,"coverage":1.}
            r=g.relacion(b,p,(0.,204.,0.),600)
            self.assertEqual(r["state"],state)
            if state:
                self.assertGreater(r["sep"],0)
        self.assertEqual(g.relacion(b,{"x":650.,"y":240.},None,600)["state"],0)
        # El eje vertical no cambia de significado al reflejar horizontalmente.
        self.assertGreater(g.relacion(instancia(np.fliplr(m),100,200),{"x":150.,"y":240.},(0.,204.,0.),600)["sep"],0)

    def test_multiples_ausentes_y_asociacion_conservadora(self):
        b1=instancia(np.ones((8,70),bool),100,100)
        b2=instancia(np.ones((8,70),bool),110,105)
        self.assertEqual(g.elegir_barra([],None,None,1000,600)[1],0)
        self.assertEqual(g.elegir_barra([b1,b2],None,np.array([140,107]),1000,600)[1],3)
        self.assertEqual(g.elegir_barra([b1],None,None,1000,600)[1],4)
        self.assertFalse(g.continuidad(b1,b2,2,2,1000,600))
        self.assertFalse(g.continuidad(b1,instancia(np.ones((8,70),bool),400,100),1,1,1000,600))

    def test_real_train_left(self):
        root=Path(__file__).resolve().parents[3]
        raw=root/"segmentation/bp/left/raw/bp_001_segmentation.npz"
        pose=root/"keypoints/bp/yolo/left/bp_001.npy"
        if not raw.exists() or not pose.exists():
            self.skipTest("Dataset local no disponible")
        d,meta=rr.cargar(raw)
        r=extraer(d,meta,np.load(pose,allow_pickle=False))
        self.assertEqual(r["features"].shape,(len(d["frame_index"]),F))
        self.assertTrue(np.array_equal(r["raw_class_counts"],d["frame_class_counts"]))
        self.assertTrue(np.array_equal(np.isfinite(r["features"]),r["availability"]))
        self.assertFalse(np.isinf(r["features"]).any())
        self.assertTrue(np.array_equal(r["bar_state"]==0,d["frame_class_counts"][:,0]==0))
        self.assertTrue(np.all(r["availability"][:,INDEX["bar_vx_w_s"]] <= r["features"][:,INDEX["q_bar_local_association"]]))

    def test_real_train_right(self):
        root=Path(__file__).resolve().parents[3]
        raw=root/"segmentation/bp/right/raw/bp_020_segmentation.npz"
        pose=root/"keypoints/bp/yolo/right/bp_020.npy"
        if not raw.exists() or not pose.exists():
            self.skipTest("Dataset local no disponible")
        d,meta=rr.cargar(raw)
        r=extraer(d,meta,np.load(pose,allow_pickle=False))
        self.assertEqual(r["features"].shape,(len(d["frame_index"]),F))
        self.assertTrue(np.array_equal(np.isfinite(r["features"]),r["availability"]))
        self.assertTrue((r["bar_state"]==2).any())
        self.assertFalse(r["availability"][:,INDEX["bar_vx_w_s"]].any())

    def test_perfil_persona_con_barra_ausente(self):
        root=Path(__file__).resolve().parents[3]
        raw=root/"segmentation/bp/right/raw/bp_242_segmentation.npz"
        pose=root/"keypoints/bp/yolo/right/bp_242.npy"
        if not raw.exists() or not pose.exists():
            self.skipTest("Dataset local no disponible")
        d,meta=rr.cargar(raw)
        result=extraer(d,meta,np.load(pose,allow_pickle=False))
        ausente=result["bar_state"]==0
        self.assertTrue(ausente.any())
        self.assertTrue(result["availability"][ausente,INDEX["torso_profile_y_h"]].any())
        self.assertFalse(result["availability"][ausente,INDEX["sep_signed_h"]].any())


if __name__=="__main__":unittest.main()
