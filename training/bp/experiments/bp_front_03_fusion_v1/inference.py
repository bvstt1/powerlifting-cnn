"""Inferencia reproducible de fusion BP_FRONT sin Test.

Tardia: reproduce ambos preprocesamientos, promedia 50/50.
Embeddings: concatena Z_pose+Z_seg estandarizados (media/std Train) y cabeza.
Devuelve probabilidad de valido + disponibilidad/advertencias.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from .fusion_models import (FUSION_DIR, POSE_DIR, SEG_DIR, FusionHead,
                           SegEmbeddingAdapter, cargar_pose, cargar_seg,
                           late_fusion)


def _scaler_pose():
    with np.load(POSE_DIR / "train_scaler.npz", allow_pickle=False) as sc:
        return sc["mean"].copy(), sc["scale"].copy()


def _canales_seg(variante):
    import json
    sub = "full" if variante == "full" else "sin_relacion"
    with open(SEG_DIR / sub / "channels_and_train_preprocessing.json",
              encoding="utf-8") as f:
        return json.load(f)


def preprocesar(pose_path, seg_path, variante, nombres):
    import importlib.util

    def mod(nombre, ruta):
        spec = importlib.util.spec_from_file_location(nombre, ruta)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    posedata = mod("inf_pose", POSE_DIR / "data.py")
    segdata = mod("inf_seg", SEG_DIR / "data.py")
    cfg = _canales_seg(variante)
    pose = np.load(pose_path, allow_pickle=False)
    xp, mask_p, _, _ = posedata.secuencia_a_170(pose)
    media, escala = _scaler_pose()
    xp = ((xp.reshape(1, -1) - media) / escala).reshape(
        1, 170, 6, 3).astype(np.float32).transpose(0, 3, 1, 2)
    with np.load(seg_path, allow_pickle=False) as sec:
        v, m, p = segdata.remuestrear(sec)
    idx = cfg["indices"]
    xx, mm = v[:, idx], m[:, idx]
    mediana = np.asarray(cfg["mediana_train"], dtype=np.float32)
    imp = np.where(mm, xx, mediana[None, :])
    xs = ((imp - np.asarray(cfg["media_train"], dtype=np.float32))
          / np.asarray(cfg["std_train"], dtype=np.float32))
    proc = p if variante == "full" else p[:, :3]
    xs = np.concatenate(
        [xs, mm.astype(np.float32), proc], axis=1)[None].transpose(0, 2, 1)
    advertencias = {
        "pose_keypoints_ausentes": int((~mask_p.astype(bool)).sum()),
        "seg_canales_ausentes": int((~mm).sum()),
        "t_original_pose": int(len(pose)),
        "t_original_seg": int(v.shape[0]) if False else None,
    }
    with np.load(seg_path, allow_pickle=False) as sec:
        advertencias["t_original_seg"] = int(len(sec["features"]))
    return (xp.astype(np.float32), xs.astype(np.float32), advertencias,
            cfg["nombres"])


@torch.no_grad()
def inferir_tardia(pose_path, seg_path, variante, dispositivo="cpu"):
    dev = torch.device(dispositivo)
    xp, xs, adv, _ = preprocesar(
        pose_path, seg_path, variante,
        _canales_seg(variante)["nombres"])
    pose, _ = cargar_pose(dev)
    seg, _ = cargar_seg(variante, dev)
    pose.eval()
    seg.eval()
    p_pose = float(pose(torch.as_tensor(xp, device=dev)).squeeze(-1)[0])
    logits = seg(torch.as_tensor(xs, device=dev))
    p_seg = float(torch.sigmoid(logits.float().reshape(-1))[0])
    return {"p_pose": p_pose, "p_seg": p_seg,
            "p_fusion": float(late_fusion(
                np.asarray([p_pose]), np.asarray([p_seg]))[0]),
            "advertencias": adv}


@torch.no_grad()
def inferir_embeddings(pose_path, seg_path, variante, cabeza_ckpt,
                       dispositivo="cpu"):
    dev = torch.device(dispositivo)
    xp, xs, adv, _ = preprocesar(
        pose_path, seg_path, variante,
        _canales_seg(variante)["nombres"])
    pose, _ = cargar_pose(dev)
    seg, _ = cargar_seg(variante, dev)
    adapt = SegEmbeddingAdapter(seg).to(dev).eval()
    pose.eval()
    zp = pose.extract_embeddings(torch.as_tensor(xp, device=dev))
    zs = adapt.embed(torch.as_tensor(xs, device=dev))
    z = torch.cat([zp, zs], dim=1).float().cpu().numpy()
    estado = torch.load(cabeza_ckpt, map_location="cpu", weights_only=False)
    z = (z - estado["emb_mean"]) / estado["emb_std"]
    cabeza = FusionHead(256).to(dev)
    cabeza.load_state_dict(estado["head_state_dict"])
    cabeza.eval()
    logit = cabeza(torch.as_tensor(z, device=dev))
    return {"p_fusion": float(torch.sigmoid(logit.float()).reshape(-1)[0]),
            "advertencias": adv}
