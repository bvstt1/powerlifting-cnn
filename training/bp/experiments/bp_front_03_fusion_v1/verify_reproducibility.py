"""FASE A: verifica recarga exacta de los 3 modelos base en Validation.

Compara probabilidades recalculadas contra los CSV congelados.
No escribe sobre archivos originales. No abre Test.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .fusion_dataset import cargar_fusion
from .fusion_models import (FUSION_DIR, POSE_DIR, SEG_DIR, SegEmbeddingAdapter,
                           cargar_pose, cargar_seg)

TOLERANCIA = 1e-5


def _csv(ruta):
    with open(ruta, encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if len(filas) != 99:
        raise ValueError(f"CSV inesperado: {ruta}")
    return {f["video_id"]: (int(f["label"]), float(f["probability_valid"]),
                            int(f["predicted"])) for f in filas}


@torch.no_grad()
def _probs(modelo, x, dispositivo, sigmoid_seg=False, amp=False):
    Ps = []
    modelo.eval()
    for i in range(0, len(x), 128):
        xb = torch.as_tensor(x[i:i + 64], device=dispositivo)
        with torch.autocast(device_type=dispositivo.type, enabled=amp):
            out = modelo(xb)
            if sigmoid_seg:
                out = torch.sigmoid(out.float())
        out = out.squeeze(-1) if out.dim() > 1 else out
        Ps.extend(out.float().cpu().numpy().tolist())
    return np.asarray(Ps, dtype=np.float64)


def main():
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    informe = {"device": (torch.cuda.get_device_name(0)
                           if dispositivo.type == "cuda" else "cpu"),
               "tolerancia": TOLERANCIA, "modelos": {}}
    datos_full = cargar_fusion("full")
    datos_sin = cargar_fusion("sin_relacion")
    xv_pose = datos_full["validation"]["pose"]
    yv = datos_full["validation"]["y"]
    ids = datos_full["validation"]["ids"]
    if datos_sin["validation"]["ids"] != ids:
        raise ValueError("IDs Pose/Seg no emparejan")

    pose, _ = cargar_pose(dispositivo)
    seg_full, _ = cargar_seg("full", dispositivo)
    seg_sin, _ = cargar_seg("sin_relacion", dispositivo)
    adapt_full = SegEmbeddingAdapter(seg_full).to(dispositivo).eval()
    adapt_sin = SegEmbeddingAdapter(seg_sin).to(dispositivo).eval()
    # La interfaz no debe cambiar las probabilidades originales.
    xs_full = datos_full["validation"]["seg"]
    xs_sin = datos_sin["validation"]["seg"]
    for nombre, adapt, original, x in (
            ("full", adapt_full, seg_full, xs_full),
            ("sin", adapt_sin, seg_sin, xs_sin)):
        a = _probs(adapt, x, dispositivo, sigmoid_seg=True)
        b = _probs(original, x, dispositivo, sigmoid_seg=True)
        if float(np.max(np.abs(a - b))) > 1e-7:
            raise ValueError(f"Adaptador {nombre} altera probabilidades")

    p_pose = _probs(pose, xv_pose, dispositivo)
    # SegCNN original infiere con AMP en CUDA: replicarlo para comparar.
    amp = dispositivo.type == "cuda"
    p_full = _probs(seg_full, xs_full, dispositivo, sigmoid_seg=True, amp=amp)
    p_sin = _probs(seg_sin, xs_sin, dispositivo, sigmoid_seg=True, amp=amp)

    esperados = {
        "pose": _csv(POSE_DIR / "seed_42/validation_predictions.csv"),
        "full": _csv(SEG_DIR / "full/seed_1337/validation_predictions.csv"),
        "sin": _csv(SEG_DIR / "sin_relacion/seed_42/validation_predictions.csv"),
    }
    for nombre, probs in (("pose", p_pose), ("full", p_full), ("sin", p_sin)):
        ref = esperados[nombre]
        if [v for v in ids] != [v for v in ref.keys()] and set(ids) != set(ref):
            raise ValueError(f"IDs {nombre} no coinciden con CSV")
        diffs = []
        for vid, p in zip(ids, probs):
            label, p_ref, pred_ref = ref[vid]
            diffs.append(abs(p - p_ref))
            if int(p >= 0.5) != pred_ref:
                raise ValueError(f"Prediccion {nombre}/{vid} difiere del CSV")
        informe["modelos"][nombre] = {
            "n": len(ids), "max_abs_diff": float(np.max(diffs)),
            "ok": bool(np.max(diffs) <= TOLERANCIA),
        }
    # Embedding shapes reales.
    with torch.no_grad():
        zp = pose.extract_embeddings(
            torch.as_tensor(xv_pose[:4], device=dispositivo)).cpu().numpy()
        zf = adapt_full.embed(
            torch.as_tensor(xs_full[:4], device=dispositivo)).cpu().numpy()
        zs = adapt_sin.embed(
            torch.as_tensor(xs_sin[:4], device=dispositivo)).cpu().numpy()
    informe["embeddings"] = {
        "pose": list(zp.shape), "seg_full": list(zf.shape),
        "seg_sin": list(zs.shape),
        "fusion_concat": [4, int(zp.shape[1] + zf.shape[1])],
    }
    if zp.shape[1] != 128 or zf.shape[1] != 128 or zs.shape[1] != 128:
        raise ValueError("Dimensiones de embedding inesperadas")
    with open(FUSION_DIR / "base_reproducibility.json", "w",
              encoding="utf-8") as archivo:
        json.dump(informe, archivo, indent=2, ensure_ascii=False)
    print(json.dumps(informe, indent=2))
    if not all(m["ok"] for m in informe["modelos"].values()):
        raise SystemExit("Reproduccion base fuera de tolerancia: detener")


if __name__ == "__main__":
    main()
