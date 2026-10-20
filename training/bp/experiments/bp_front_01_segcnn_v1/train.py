"""Primer baseline SegCNN BP_FRONT: FULL y ablación sin barra–silueta.

Ejecutar: python -m training.bp.experiments.bp_front_01_segcnn_v1.train
No carga NPZ del split Test ni usa su evaluación para ninguna decisión.
"""
import csv
import hashlib
import json
import os
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from .data import (COHORTE, ESQUEMA, FUENTE, QUALITY_POSE_ASSISTED,
                  RELACION_EXCLUIDA, canales_constantes_train, cargar_datos,
                  leer_cohorte, preparar_variante)
from .model import SegCNNTemporal

SALIDA = Path(__file__).resolve().parent
CONFIG = {
    "experimentos": ["full", "sin_relacion"], "semillas": [42, 1337],
    "T": 170, "batch_size": 32, "max_epochs": 200,
    "early_stopping_patience": 15, "learning_rate": 0.001,
    "weight_decay": 0.0001, "scheduler_patience": 5,
    "scheduler_factor": 0.5, "umbral": 0.5,
    "checkpoint_selection": "menor loss Validation, desempate menor epoch",
    "seed_selection": "menor loss Validation del mejor checkpoint por semilla",
    "threshold_selection": "0.5 fijado antes de entrenar; sin ajuste en Validation",
    "early_stopping": "patience 15 en loss Validation",
    "temporal_grid": "linspace(0,T_original-1,170), igual PoseCNN; no puentear NaN/procedencia/track",
    "schema_warning": "q_bar_associated/q_bar_unverified/q_bar_multiple/q_projected derivan de seleccion Pose-assisted aunque aparecen QUALITY",
    "test_policy": "no se cargan NPZ Test; no se evalua Test",
}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as arch:
        for bloque in iter(lambda: arch.read(4 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def escribir_json(ruta, obj):
    Path(ruta).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def escribir_csv(ruta, filas):
    if not filas:
        return
    with open(ruta, "w", encoding="utf-8", newline="") as arch:
        writer = csv.DictWriter(arch, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)


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
        "sensibilidad_valido": tp / (tp + fn),
        "especificidad_invalido": tn / (tn + fp),
        "recall_invalido": tn / (tn + fp),
        "precision_invalido": tn / (tn + fn) if tn + fn else 0.0,
        "matriz_confusion_0_1": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def inferir(modelo, x, y, dispositivo, loss_fn, amp):
    modelo.eval()
    acumulado = 0.0
    ps = []
    with torch.no_grad():
        for a in range(0, len(y), 128):
            xb = torch.as_tensor(x[a:a + 128], device=dispositivo)
            yb = torch.as_tensor(y[a:a + 128], dtype=torch.float32,
                                 device=dispositivo)
            with torch.autocast(device_type=dispositivo.type, enabled=amp):
                logits = modelo(xb)
                loss = loss_fn(logits.float(), yb)
            acumulado += float(loss.item()) * len(yb)
            ps.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return acumulado / len(y), np.asarray(ps, dtype=np.float64)


def dibujar_historia(historia, ruta):
    ep = [r["epoch"] for r in historia]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for clave in ("train_loss", "validation_loss"):
        axes[0].plot(ep, [r[clave] for r in historia], label=clave)
    for clave in ("train_f1", "validation_f1"):
        axes[1].plot(ep, [r[clave] for r in historia], label=clave)
    axes[0].set_ylabel("BCE loss")
    axes[1].set_ylabel("F1 por video")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def dibujar_validacion(y, p, cm, carpeta):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p[y == 0], bins=np.linspace(0, 1, 21), alpha=0.6,
            label="Inválido (0)")
    ax.hist(p[y == 1], bins=np.linspace(0, 1, 21), alpha=0.6,
            label="Válido (1)")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set(xlabel="Probabilidad de válido", ylabel="Videos Validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "validation_probabilities.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    ax.set(xlabel="Predicción", ylabel="Etiqueta", xticks=[0, 1], yticks=[0, 1])
    fig.tight_layout()
    fig.savefig(carpeta / "validation_confusion.png", dpi=140)
    plt.close(fig)


def entrenar_variante(variante, preparado, canales, dispositivo):
    carpeta = SALIDA / variante
    carpeta.mkdir(parents=True, exist_ok=True)
    escribir_json(carpeta / "channels_and_train_preprocessing.json", canales)
    xtr, ytr, idtr = preparado["train"]
    xva, yva, idva = preparado["validation"]
    assert xtr.shape == (462, canales["canales_total"], 170)
    assert xva.shape == (99, canales["canales_total"], 170)
    resultados = []
    semilla_elegida = None
    mejor_val_loss = float("inf")
    usa_amp = dispositivo.type == "cuda"
    for semilla in CONFIG["semillas"]:
        fijar_semilla(semilla)
        salida_sem = carpeta / f"seed_{semilla}"
        salida_sem.mkdir(parents=True, exist_ok=True)
        modelo = SegCNNTemporal(canales["canales_total"]).to(dispositivo)
        parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        optim = torch.optim.AdamW(modelo.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", patience=CONFIG["scheduler_patience"],
            factor=CONFIG["scheduler_factor"])
        criterio = torch.nn.BCEWithLogitsLoss()
        escalador = torch.amp.GradScaler("cuda", enabled=usa_amp)
        dt = TensorDataset(torch.from_numpy(xtr),
                           torch.from_numpy(ytr.astype(np.float32)))
        generador = torch.Generator().manual_seed(semilla)
        dl = DataLoader(dt, batch_size=CONFIG["batch_size"], shuffle=True,
                        generator=generador, num_workers=0)
        historial = []
        mejor = float("inf")
        mejor_ep = 0
        espera = 0
        ckpt = salida_sem / "best_validation_loss.pt"
        for ep in range(1, CONFIG["max_epochs"] + 1):
            modelo.train()
            for xb, yb in dl:
                xb, yb = xb.to(dispositivo), yb.to(dispositivo)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dispositivo.type, enabled=usa_amp):
                    logits = modelo(xb)
                    loss = criterio(logits.float(), yb)
                escalador.scale(loss).backward()
                escalador.step(optim)
                escalador.update()
            loss_tr, p_tr = inferir(modelo, xtr, ytr, dispositivo, criterio, usa_amp)
            loss_va, p_va = inferir(modelo, xva, yva, dispositivo, criterio, usa_amp)
            scheduler.step(loss_va)
            h = {"epoch": ep, "train_loss": loss_tr,
                 "validation_loss": loss_va,
                 "train_f1": metrica(ytr, p_tr)["f1"],
                 "validation_f1": metrica(yva, p_va)["f1"],
                 "lr": optim.param_groups[0]["lr"]}
            historial.append(h)
            if loss_va < mejor - 1e-7:
                mejor = loss_va
                mejor_ep = ep
                espera = 0
                torch.save({"state_dict": modelo.state_dict(),
                            "channels": canales["nombres"],
                            "channels_total": canales["canales_total"],
                            "epoch": ep, "seed": semilla,
                            "validation_loss": loss_va,
                            "threshold": CONFIG["umbral"]}, ckpt)
            else:
                espera += 1
            if ep == 1 or ep % 10 == 0:
                print(f"{variante} seed={semilla} epoch={ep} "
                      f"train_loss={loss_tr:.4f} val_loss={loss_va:.4f} "
                      f"val_F1={h['validation_f1']:.4f}", flush=True)
            if espera >= CONFIG["early_stopping_patience"]:
                break
        # Evaluación del único checkpoint elegido por Validation loss.
        estado = torch.load(ckpt, map_location=dispositivo, weights_only=True)
        modelo.load_state_dict(estado["state_dict"])
        lt, pt = inferir(modelo, xtr, ytr, dispositivo, criterio, usa_amp)
        lv, pv = inferir(modelo, xva, yva, dispositivo, criterio, usa_amp)
        mt, mv = metrica(ytr, pt), metrica(yva, pv)
        resumen = {"variante": variante, "seed": semilla,
                   "parametros": parametros, "C": canales["canales_total"],
                   "epoch_best": mejor_ep, "epochs_run": ep,
                   "train_loss_best": lt, "validation_loss_best": lv,
                   "train_metrics": mt, "validation_metrics": mv,
                   "checkpoint": str(ckpt.relative_to(SALIDA))}
        escribir_json(salida_sem / "metrics.json", resumen)
        escribir_csv(salida_sem / "learning_curves.csv", historial)
        escribir_csv(salida_sem / "validation_predictions.csv", [
            {"video_id": video_id, "label": int(y), "probability_valid": float(p),
             "predicted": int(p >= CONFIG["umbral"])}
            for video_id, y, p in zip(idva, yva, pv)])
        np.save(salida_sem / "validation_confusion.npy",
                np.asarray(mv["matriz_confusion_0_1"], dtype=np.int64))
        dibujar_historia(historial, salida_sem / "learning_curves.png")
        dibujar_validacion(yva, pv, mv["matriz_confusion_0_1"], salida_sem)
        resultados.append(resumen)
        print(f"{variante} seed={semilla}: best_ep={mejor_ep}, "
              f"val_loss={lv:.4f}, F1={mv['f1']:.4f}, "
              f"macroF1={mv['macro_f1']:.4f}, AUC={mv['roc_auc']:.4f}", flush=True)
        if lv < mejor_val_loss:
            mejor_val_loss = lv
            semilla_elegida = semilla
    escribir_json(carpeta / "seeds_metrics.json", resultados)
    escribir_json(carpeta / "selected_checkpoint.json", {
        "seed": semilla_elegida,
        "checkpoint": f"seed_{semilla_elegida}/best_validation_loss.pt",
        "selection": CONFIG["seed_selection"], "threshold": CONFIG["umbral"]})
    return resultados, semilla_elegida


def main():
    torch.set_num_threads(4)
    SALIDA.mkdir(parents=True, exist_ok=True)
    grupos, nombres = leer_cohorte()
    datos = cargar_datos(grupos, nombres)
    constantes = canales_constantes_train(*datos["train"][:2], nombres)
    if "profile_ambiguous" not in constantes:
        raise RuntimeError("profile_ambiguous no constante en Train; parar y revisar")
    print("Cohorte train/val/test (test sin abrir):",
          len(grupos["train"]), len(grupos["validation"]), len(grupos["test"]))
    print("Train constantes:", constantes, flush=True)
    print("Calidad Train:", Counter(f["quality_status_seg"] for f in grupos["train"]))
    # Congelar configuración y fuentes antes del entrenamiento.
    manifiesto = {
        "cohorte_counts": {k: len(v) for k, v in grupos.items()},
        "splits": "bp_front_features_common_cohort.csv; no modificaciones",
        "constant_train": constantes,
        "excluded_relation_channels_predefined": sorted(RELACION_EXCLUIDA),
        "schema_quality_pose_dependency": list(QUALITY_POSE_ASSISTED),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": (torch.cuda.get_device_name(0)
                                             if torch.cuda.is_available() else "cpu"),
        "fuentes_sha256": {str(p.relative_to(SALIDA.parents[3])): sha256(p)
                           for p in [COHORTE, ESQUEMA,
                                     FUENTE / "bp_front_features_config.json",
                                     FUENTE / "bp_front_features_validation.json",
                                     SALIDA.parents[3] / "segmentation/bp/front/manifests/bp_front_segmentation_manifest.csv"]},
        "train_validation_sequences_sha256": {
            f["video_id"]: sha256(Path(f["features_path"]))
            for split in ("train", "validation") for f in grupos[split]},
        "test_sequences_opened": 0,
    }
    escribir_json(SALIDA / "training_config.json", CONFIG)
    escribir_json(SALIDA / "source_manifest.json", manifiesto)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resultados = {}
    for variante in CONFIG["experimentos"]:
        preparado, canales = preparar_variante(datos, nombres, variante, constantes)
        resultados[variante], seleccion = entrenar_variante(variante, preparado,
                                                              canales, dispositivo)
    train_y = datos["train"][3]
    val_y = datos["validation"][3]
    clase = int(np.mean(train_y) >= 0.5)
    p_base = np.full(len(val_y), float(np.mean(train_y)), dtype=np.float64)
    referencia = metrica(val_y, p_base)
    escribir_json(SALIDA / "comparison_validation.json", {
        "baseline_mayoritaria": {"clase_train": clase,
                                 "prior_train_valid": float(np.mean(train_y)),
                                 "validation": referencia},
        "experimentos": resultados,
        "test_untouched": True,
    })
    print("BASELINE MAYORITARIA Validation:", referencia, flush=True)
    print("FIN. Test sin evaluar.", flush=True)


if __name__ == "__main__":
    main()
