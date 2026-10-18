"""Adaptadores externos; los estados PoseCNN y SegCNN permanecen inalterados."""
import torch
from torch import nn

from training.bp_posecnn_common_v1.core import PoseEmbeddingAdapter


class SegEmbeddingAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone=backbone

    def extract_embeddings(self,x):
        return self.backbone.temporal(x).flatten(1)

    def forward(self,x):
        return self.backbone.clasificador(self.extract_embeddings(x)).squeeze(-1)


class FusionHead(nn.Module):
    def __init__(self, pose_dim=64, seg_dim=128):
        super().__init__()
        self.pose_dim=pose_dim;self.seg_dim=seg_dim
        self.net=nn.Sequential(nn.Linear(pose_dim+seg_dim,32),nn.ReLU(),nn.Dropout(.40),nn.Linear(32,1))

    def forward(self,z):
        if z.ndim!=2 or z.shape[1]!=self.pose_dim+self.seg_dim:
            raise ValueError("Embedding concatenado incompatible")
        return self.net(z).squeeze(-1)


def late_fusion(p_pose,p_seg):
    if p_pose.shape!=p_seg.shape:
        raise ValueError("Probabilidades de distinta cohorte")
    return .5*p_pose+.5*p_seg


def verify_adapters(pose_model,seg_model,pose_x,seg_x,device):
    pose_model.eval();seg_model.eval()
    pose=PoseEmbeddingAdapter(pose_model).to(device).eval()
    seg=SegEmbeddingAdapter(seg_model).to(device).eval()
    with torch.no_grad():
        xp=torch.as_tensor(pose_x[:8],device=device)
        xs=torch.as_tensor(seg_x[:8],device=device)
        p=pose_model(xp)
        z_pose=pose.extract_embeddings(xp)
        recovered_pose=pose.classifier_from_embeddings(z_pose)
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
            original_seg=seg_model(xs)
            z_seg=seg.extract_embeddings(xs)
            recovered_seg=seg_model.clasificador(z_seg).squeeze(-1)
    if not torch.equal(p,recovered_pose) or not torch.equal(original_seg,recovered_seg):
        raise RuntimeError("Adaptador modifica forward original")
    if z_pose.shape!=(len(xp),64) or z_seg.shape!=(len(xs),128):
        raise RuntimeError(f"Embeddings inesperados {z_pose.shape} {z_seg.shape}")
    head=FusionHead(z_pose.shape[1],z_seg.shape[1])
    n=sum(p.numel() for p in head.parameters() if p.requires_grad)
    if n!=6209:raise RuntimeError(f"Parámetros cabeza {n} != 6209")
    return {"pose_input_shape":list(xp.shape),"seg_input_shape":list(xs.shape),
            "pose_embedding_shape":list(z_pose.shape),"seg_embedding_shape":list(z_seg.shape),
            "concat_dimension":int(z_pose.shape[1]+z_seg.shape[1]),"head_trainable_parameters":n,
            "pose_forward_exact":True,"seg_forward_exact":True}
