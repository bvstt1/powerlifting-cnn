"""Inferencia reproducible por video de desarrollo (sin Test en esta fase).

Reconstruye exactamente Pose y Seg desde NPY/NPZ originales monovista.
`allow_test` no existe aquí: el protocolo final debe implementarse por separado.
"""
import json
import pickle

import numpy as np
import torch

from training.bp.train_posecnn import preprocess, transform
from training.bp.experiments.bp_front_01_segcnn_v1.model import SegCNNTemporal
from training.bp.experiments.bp_side_segcnn_v1.data import cohort, resample, sha
from training.common.train_utils import PoseCNN
from training.bp.experiments.bp_posecnn_common_v1.core import paths, rows_csv
from .data import fusion_out, snapshot
from .models import FusionHead, PoseEmbeddingAdapter, SegEmbeddingAdapter, late_fusion


class FusionInference:
    def __init__(self,view,variant,kind,device=None):
        if view not in ("left","right") or variant not in ("full","sin_relacion") or kind not in ("late","embedding"):
            raise ValueError("Vista/variante/tipo no reconocido")
        self.view=view;self.variant=variant;self.kind=kind
        self.p=paths(view);self.out=fusion_out(view)
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        src=json.loads((self.out/"fusion_source_manifest.json").read_text(encoding="utf-8"))
        if src["source_sha256"]!=snapshot(view):raise RuntimeError("Fuente protegida mutada")
        co,names,_,_=cohort(view)
        self.roles={r["video_id"]:role for role in ("train","validation") for r in co[role]}
        self.names=names
        pose_path=self.p["historical_ckpt"] if view=="left" else self.p["right_out"]/"seed_1337/best_model.pth"
        if sha(pose_path)!=src["source_sha256"]["pose_checkpoint"]["sha256"]:
            raise RuntimeError("PoseCNN checkpoint alterado")
        self.pose=PoseCNN(T=170,K=8,C=3).to(self.device)
        self.pose.load_state_dict(torch.load(pose_path,map_location=self.device,weights_only=True),strict=True)
        self.pose.eval().requires_grad_(False)
        pose_scaler=self.p["historical_scaler"] if view=="left" else self.p["right_out"]/"scaler_train_original.pkl"
        with open(pose_scaler,"rb") as file:self.pose_scaler=pickle.load(file)
        if int(self.pose_scaler.n_samples_seen_)!=(422 if view=="left" else 431):
            raise RuntimeError("Pose scaler no corresponde a Train")
        root=self.p["seg_train"]/variant
        selected=json.loads((root/"selected_checkpoint.json").read_text(encoding="utf-8"))
        file=self.p["seg_train"]/selected["checkpoint"]
        if sha(file)!=src["source_sha256"][f"seg_{variant}_checkpoint"]["sha256"]:
            raise RuntimeError("SegCNN checkpoint alterado")
        self.contract=json.loads((root/"channels_and_train_preprocessing.json").read_text(encoding="utf-8"))
        state=torch.load(file,map_location=self.device,weights_only=True)
        if state["input_channels"]!=self.contract["input_channel_names"]:
            raise RuntimeError("Canales SegCNN no coinciden")
        self.seg=SegCNNTemporal(self.contract["C"]).to(self.device)
        self.seg.load_state_dict(state["state_dict"],strict=True)
        self.seg.eval().requires_grad_(False)
        self.head=None
        if kind=="embedding":
            parent=self.out/f"embedding_fusion_{variant}"
            selected=json.loads((parent/"selected_checkpoint.json").read_text(encoding="utf-8"))
            checkpoint=parent/selected["checkpoint"]
            if sha(checkpoint)!=selected["checkpoint_sha256"]:raise RuntimeError("Cabeza de fusión alterada")
            record=torch.load(checkpoint,map_location=self.device,weights_only=True)
            with np.load(parent/"embedding_scaler_train.npz",allow_pickle=False) as sc:
                self.emb_mean=sc["mean"].copy();self.emb_std=sc["std"].copy()
            if (record["embedding_scaler_sha256"]!=sha(parent/"embedding_scaler_train.npz") or
                self.emb_mean.shape!=(192,) or self.emb_std.shape!=(192,)):
                raise RuntimeError("Scaler embedding alterado")
            self.head=FusionHead(64,128).to(self.device).eval()
            self.head.load_state_dict(record["head_state_dict"],strict=True)
            self.head.requires_grad_(False)

    def from_sources(self,vid):
        if self.roles.get(vid) not in ("train","validation"):
            raise ValueError("Video fuera de desarrollo; protocolo Test separado")
        pose_file=self.p["keypoints"]/f"{vid}.npy"
        seq_file=self.p["source"]/"sequences"/f"{vid}_temporal.npz"
        x,_,_=preprocess(np.load(pose_file,allow_pickle=False))
        xp=np.ascontiguousarray(transform(self.pose_scaler,x[None]).transpose(0,3,1,2))
        with np.load(seq_file,allow_pickle=False) as z:
            if str(z["view"])!=self.view or str(z["video_id"])!=vid:
                raise RuntimeError("Secuencia de vista/ID ajeno")
            values,available,origin=resample(z,self.names)
        indices=self.contract["selected_feature_indices"]
        stats=self.contract["train_statistics"]
        chosen=self.contract["selected_feature_names"]
        xx=values[:,indices];aa=available[:,indices].astype(bool);pp=origin[:,indices]
        med=np.array([stats[n]["median"] for n in chosen],np.float32)
        mean=np.array([stats[n]["mean"] for n in chosen],np.float32)
        std=np.array([stats[n]["std"] for n in chosen],np.float32)
        numeric=(np.where(aa,xx,med[None,:])-mean[None,:])/std[None,:]
        flags=[np.isin(pp,key).astype(np.float32) for key in (2,(3,6),4,5,7)]
        combined=np.concatenate((numeric,aa.astype(np.float32),*flags),axis=1).astype(np.float32)
        keep=self.contract["input_channel_indices_before_constant_filter"]
        xs=np.ascontiguousarray(combined[:,keep].T[None])
        if xs.shape!=(1,self.contract["C"],170) or not np.isfinite(xs).all():
            raise RuntimeError("Transform de SegCNN no reproducible")
        return xp,xs

    @torch.no_grad()
    def predict_processed(self,xp,xs):
        if xp.ndim!=4 or xp.shape[1:]!=(3,170,8) or xs.ndim!=3 or xs.shape[1:]!=(self.contract["C"],170) or len(xp)!=len(xs):
            raise ValueError("Input fusion incompatible")
        p=torch.as_tensor(xp,device=self.device)
        s=torch.as_tensor(xs,device=self.device)
        self.pose.eval();self.seg.eval()
        if self.kind=="late":
            pose=torch.as_tensor(self.pose(p).reshape(-1)).float()
            with torch.autocast(device_type=self.device.type,enabled=self.device.type=="cuda"):
                logits=self.seg(s)
            return late_fusion(pose.cpu().numpy(),torch.sigmoid(logits.float()).cpu().numpy())
        self.head.eval()
        zp=PoseEmbeddingAdapter(self.pose).extract_embeddings(p)
        with torch.autocast(device_type=self.device.type,enabled=self.device.type=="cuda"):
            zs=SegEmbeddingAdapter(self.seg).extract_embeddings(s)
        z=torch.cat((zp.float(),zs.float()),dim=1).cpu().numpy()
        standardized=(z-self.emb_mean)/self.emb_std
        logits=self.head(torch.as_tensor(standardized,device=self.device))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)
