"""Carga congelada de backbones BP_FRONT y cabeza de fusion.

No modifica checkpoints, scalers ni preprocesamientos originales.
La interfaz de embedding SegCNN se verifica contra el forward original.
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parents[2]
POSE_DIR = RAIZ / "training/bp_front_posecnn_common_v1"
SEG_DIR = RAIZ / "training/bp_front_segcnn_v1"
FUSION_DIR = Path(__file__).resolve().parent


def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def cargar_pose(dispositivo):
    mod = _modulo("pose_model", POSE_DIR / "model.py")
    ckpt_path = POSE_DIR / "seed_42/best_validation_loss.pt"
    modelo, contenido = mod.cargar_checkpoint(ckpt_path, str(dispositivo))
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    return modelo, contenido


def cargar_seg(variante, dispositivo):
    mod = _modulo("seg_model", SEG_DIR / "model.py")
    sub = "full" if variante == "full" else "sin_relacion"
    ckpt_path = SEG_DIR / sub / (
        "seed_1337/best_validation_loss.pt" if variante == "full"
        else "seed_42/best_validation_loss.pt")
    contenido = torch.load(ckpt_path, map_location=dispositivo,
                           weights_only=False)
    canales = int(contenido["channels_total"])
    if (variante == "full" and canales != 79) or (
            variante == "sin_relacion" and canales != 65):
        raise ValueError(f"Canales SegCNN inesperados: {canales}")
    modelo = mod.SegCNNTemporal(canales).to(dispositivo)
    modelo.load_state_dict(contenido["state_dict"], strict=True)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    return modelo, contenido


class SegEmbeddingAdapter(nn.Module):
    """Expone el embedding temporal (N,128) sin alterar el checkpoint."""

    def __init__(self, seg_model):
        super().__init__()
        self.backbone = seg_model

    def embed(self, x):
        return self.backbone.temporal(x).flatten(1)

    def forward(self, x):
        return self.backbone.clasificador(self.embed(x)).squeeze(-1)


class FusionHead(nn.Module):
    """Cabeza compacta sobre concatenacion (N,256). Solo esto se entrena."""

    def __init__(self, entrada=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(entrada, 32), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def late_fusion(p_pose, p_seg):
    return 0.5 * p_pose + 0.5 * p_seg


def parametros_entrenables(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
