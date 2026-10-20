"""FASE B: fusion tardia 50/50 fija sobre CSV congelados. Sin entrenamiento.

No toca checkpoints ni preprocesamientos. Umbral 0.5 sin optimizar.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

FUSION_DIR = Path(__file__).resolve().parent
RAIZ = FUSION_DIR.parents[3]


def _csv(ruta):
    with open(ruta, encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def metrica(y, p, umbral=0.5):
    pred = (p >= umbral).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "sensibilidad_valido": float(tp / (tp + fn)),
        "especificidad_invalido": float(tn / (tn + fp)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "matriz_confusion_0_1": cm.tolist(),
    }


def main():
    pose = _csv(RAIZ / "training/bp/experiments/bp_front_02_posecnn_common_v1/seed_42/validation_predictions.csv")
    full = _csv(RAIZ / "training/bp/experiments/bp_front_01_segcnn_v1/full/seed_1337/validation_predictions.csv")
    sin = _csv(RAIZ / "training/bp/experiments/bp_front_01_segcnn_v1/sin_relacion/seed_42/validation_predictions.csv")
    if [r["video_id"] for r in pose] != [r["video_id"] for r in full] or [
            r["video_id"] for r in pose] != [r["video_id"] for r in sin]:
        raise ValueError("Orden de IDs Validation difiere entre CSV base")
    for nombre, seg in (("late_fusion_full", full),
                        ("late_fusion_sin_relacion", sin)):
        carpeta = FUSION_DIR / nombre
        carpeta.mkdir(parents=True, exist_ok=True)
        y = np.asarray([int(r["label"]) for r in pose], dtype=np.int64)
        pp = np.asarray([float(r["probability_valid"]) for r in pose])
        ps = np.asarray([float(r["probability_valid"]) for r in seg])
        pf = 0.5 * pp + 0.5 * ps
        m = metrica(y, pf)
        with open(carpeta / "validation_predictions.csv", "w",
                  encoding="utf-8", newline="") as archivo:
            w = csv.DictWriter(archivo, fieldnames=[
                "video_id", "label", "p_pose", "p_seg", "p_fusion", "predicted"])
            w.writeheader()
            for r, a, b, c in zip(pose, pp, ps, pf):
                w.writerow({"video_id": r["video_id"], "label": r["label"],
                            "p_pose": float(a), "p_seg": float(b),
                            "p_fusion": float(c), "predicted": int(c >= 0.5)})
        with open(carpeta / "metrics.json", "w", encoding="utf-8") as archivo:
            json.dump({"experimento": nombre, "regla": "0.5*p_pose+0.5*p_seg",
                       "umbral": 0.5, "entrenamiento": False,
                       "validation_metrics": m}, archivo, indent=2)
        print(nombre, {k: round(m[k], 4) for k in (
            "accuracy", "precision", "recall", "f1", "macro_f1",
            "balanced_accuracy", "roc_auc", "pr_auc",
            "sensibilidad_valido", "especificidad_invalido")},
            (m["tn"], m["fp"], m["fn"], m["tp"]), flush=True)


if __name__ == "__main__":
    main()
