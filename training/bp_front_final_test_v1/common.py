"""Constantes finales congelables BP_FRONT; sin entrenamiento ni acceso implícito a Test."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

RAIZ = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
POSE = RAIZ / "training/bp_front_posecnn_common_v1"
SEG = RAIZ / "training/bp_front_segcnn_v1"
FUSION = RAIZ / "training/bp_front_fusion_v1"
COHORTE = (RAIZ / "segmentation/bp/front/temporal_features_v1"
           / "bp_front_features_common_cohort.csv")
POSE_CKPT = POSE / "seed_42/best_validation_loss.pt"
SEG_CKPT = SEG / "sin_relacion/seed_42/best_validation_loss.pt"
SCALER_POSE = POSE / "train_scaler.npz"
SCALER_SEG = SEG / "sin_relacion/channels_and_train_preprocessing.json"
UMBRAL = 0.5
TOLERANCIA_VAL = 1e-5


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(4 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cohortes():
    with open(COHORTE, encoding="utf-8-sig", newline="") as arch:
        filas = list(csv.DictReader(arch))
    if len(filas) != 663 or len({r["video_id"] for r in filas}) != 663:
        raise RuntimeError("Inventario alterado")
    exc = {r["video_id"] for r in filas if r["excluido"] == "True"}
    if exc != {"bp_349", "bp_352"}:
        raise RuntimeError(f"Exclusiones alteradas: {exc}")
    particiones = {s: [r for r in filas
                       if r["excluido"] == "False" and r["common_split"] == s]
                   for s in ("train", "validation", "test")}
    if [len(particiones[s]) for s in ("train", "validation", "test")] != [462, 99, 100]:
        raise RuntimeError("Partición no coincide 462/99/100")
    ids = [r["video_id"] for s in particiones for r in particiones[s]]
    if len(set(ids)) != 661 or any(r["label"] not in {"0", "1"}
                                  for s in particiones for r in particiones[s]):
        raise RuntimeError("ID duplicado o label no binaria")
    return particiones


def ruta_seg_test(r):
    return RAIZ / "segmentation/bp/front/temporal_features_v1/sequences" / (
        r["video_id"] + "_temporal.npz")


def ruta_pose_test(r):
    return RAIZ / "keypoints/bp/yolo/front" / (r["video_id"] + ".npy")


def cargar_modelos(dispositivo):
    posemod = modulo("final_pose_model", POSE / "model.py")
    segmod = modulo("final_seg_model", SEG / "model.py")
    pose, info_pose = posemod.cargar_checkpoint(POSE_CKPT, str(dispositivo))
    estado_seg = torch.load(SEG_CKPT, map_location=dispositivo,
                            weights_only=True)
    if (info_pose["seed"], info_pose["threshold"],
            estado_seg["seed"], estado_seg["channels_total"],
            estado_seg["threshold"]) != (42, 0.5, 42, 65, 0.5):
        raise RuntimeError("Checkpoints/umbral/canales no congelados")
    seg = segmod.SegCNNTemporal(65).to(dispositivo)
    seg.load_state_dict(estado_seg["state_dict"], strict=True)
    pose.eval()
    seg.eval()
    for red in (pose, seg):
        if any(p.requires_grad for p in red.parameters()):
            red.requires_grad_(False)
    return pose, seg


def metrica(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pr = (p >= UMBRAL).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pr, labels=[0, 1]).ravel())
    return {"accuracy": float(accuracy_score(y, pr)),
            "precision": float(precision_score(y, pr, zero_division=0)),
            "recall": float(recall_score(y, pr, zero_division=0)),
            "f1": float(f1_score(y, pr, zero_division=0)),
            "macro_f1": float(f1_score(y, pr, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pr)),
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "pr_auc_average_precision": float(average_precision_score(y, p)),
            "sensibilidad_valido": tp / (tp + fn) if tp + fn else None,
            "especificidad_invalido": tn / (tn + fp) if tn + fp else None,
            "fpr": fp / (tn + fp) if tn + fp else None,
            "fnr": fn / (fn + tp) if fn + tp else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "matriz_confusion_0_1": [[tn, fp], [fn, tp]]}


def guardar_json(ruta, objeto):
    Path(ruta).write_text(json.dumps(objeto, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def guardar_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as arch:
        w = csv.DictWriter(arch, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
