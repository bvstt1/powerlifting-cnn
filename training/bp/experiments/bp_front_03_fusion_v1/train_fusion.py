"""FASE C+D: cabezas de fusion sobre embeddings congelados (B1/B2 x 2 semillas).

Backbones en eval sin gradientes. Solo la cabeza Linear(256,32)->ReLU->
Dropout(0.40)->Linear(32,1) se entrena con BCEWithLogitsLoss.
Estandarizacion de embeddings ajustada solo con Train por experimento.
No abre Test. No modifica checkpoints base.
"""
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .fusion_dataset import cargar_fusion
from .fusion_models import (FUSION_DIR, FusionHead, SegEmbeddingAdapter,
                           cargar_pose, cargar_seg, parametros_entrenables)
from .late_fusion import metrica

CONFIG = {"semillas": [42, 1337], "batch_size": 32, "max_epochs": 100,
          "patience": 15, "lr": 0.001, "weight_decay": 0.001,
          "scheduler_patience": 5, "scheduler_factor": 0.5, "umbral": 0.5}


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def extraer(pose, adapt, x_pose, x_seg, dispositivo):
    pose.eval()
    adapt.eval()
    Zs = []
    for i in range(0, len(x_pose), 128):
        zp = pose.extract_embeddings(
            torch.as_tensor(x_pose[i:i + 128], device=dispositivo))
        zs = adapt.embed(torch.as_tensor(x_seg[i:i + 128], device=dispositivo))
        Zs.append(torch.cat([zp, zs], dim=1).float().cpu().numpy())
    z = np.concatenate(Zs, axis=0)
    if z.shape[1] != 256:
        raise ValueError(f"Concat inesperada: {z.shape}")
    return z.astype(np.float32)


def evaluar_cabeza(cabeza, z, y, dispositivo, criterion):
    cabeza.eval()
    tot, ps = 0.0, []
    with torch.no_grad():
        for i in range(0, len(y), 256):
            zb = torch.as_tensor(z[i:i + 256], device=dispositivo)
            yb = torch.as_tensor(y[i:i + 256], dtype=torch.float32,
                                 device=dispositivo)
            logits = cabeza(zb)
            tot += float(criterion(logits, yb).item()) * len(yb)
            ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return tot / len(y), np.asarray(ps, dtype=np.float64)


def dibujar(hist, ruta):
    ep = [r["epoch"] for r in hist]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k in ("train_loss", "validation_loss"):
        axes[0].plot(ep, [r[k] for r in hist], label=k)
    for k in ("train_f1", "validation_f1"):
        axes[1].plot(ep, [r[k] for r in hist], label=k)
    axes[0].set_ylabel("BCE loss cabeza")
    axes[1].set_ylabel("F1 por video")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def entrenar(variante, datos, dispositivo):
    carpeta = FUSION_DIR / (
        "embedding_fusion_full" if variante == "full" else
        "embedding_fusion_sin_relacion")
    carpeta.mkdir(parents=True, exist_ok=True)
    pose, _ = cargar_pose(dispositivo)
    seg, _ = cargar_seg(variante, dispositivo)
    adapt = SegEmbeddingAdapter(seg).to(dispositivo).eval()
    for p in list(pose.parameters()) + list(seg.parameters()):
        if p.requires_grad:
            raise ValueError("Backbone no congelado")
    dtr, dva = datos["train"], datos["validation"]
    ztr = extraer(pose, adapt, dtr["pose"], dtr["seg"], dispositivo)
    zva = extraer(pose, adapt, dva["pose"], dva["seg"], dispositivo)
    ytr, yva = dtr["y"], dva["y"]
    media = ztr.mean(axis=0).astype(np.float32)
    desv = ztr.std(axis=0).astype(np.float32)
    desv[desv < 1e-6] = 1e-6
    np.savez_compressed(carpeta / "embedding_scaler_train.npz",
                        mean=media, std=desv)
    ztr_s, zva_s = (ztr - media) / desv, (zva - media) / desv
    if not np.isfinite(ztr_s).all() or not np.isfinite(zva_s).all():
        raise ValueError("Embeddings no finitos")
    resultados, mejor_global, semilla_ganadora = [], float("inf"), None
    for semilla in CONFIG["semillas"]:
        fijar_semilla(semilla)
        sal = carpeta / f"seed_{semilla}"
        sal.mkdir(parents=True, exist_ok=True)
        cabeza = FusionHead(256).to(dispositivo)
        n_params = parametros_entrenables(cabeza)
        opt = torch.optim.AdamW(cabeza.parameters(), lr=CONFIG["lr"],
                                weight_decay=CONFIG["weight_decay"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=CONFIG["scheduler_factor"],
            patience=CONFIG["scheduler_patience"])
        crit = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(semilla)
        dl = DataLoader(TensorDataset(torch.from_numpy(ztr_s),
                                      torch.from_numpy(ytr.astype(np.float32))),
                        batch_size=CONFIG["batch_size"], shuffle=True,
                        generator=gen, num_workers=0)
        hist, mejor, mejor_ep, espera = [], float("inf"), 0, 0
        ckpt = sal / "best_validation_loss.pt"
        for epoca in range(1, CONFIG["max_epochs"] + 1):
            cabeza.train()
            for zb, yb in dl:
                zb, yb = zb.to(dispositivo), yb.to(dispositivo)
                opt.zero_grad(set_to_none=True)
                loss = crit(cabeza(zb), yb)
                loss.backward()
                opt.step()
            ltr, ptr = evaluar_cabeza(cabeza, ztr_s, ytr, dispositivo, crit)
            lva, pva = evaluar_cabeza(cabeza, zva_s, yva, dispositivo, crit)
            sch.step(lva)
            hist.append({"epoch": epoca, "train_loss": ltr,
                         "validation_loss": lva,
                         "train_f1": metrica(ytr, ptr)["f1"],
                         "validation_f1": metrica(yva, pva)["f1"],
                         "lr": float(opt.param_groups[0]["lr"])})
            if lva < mejor - 1e-7:
                mejor, mejor_ep, espera = lva, epoca, 0
                torch.save({"head_state_dict": cabeza.state_dict(),
                            "emb_mean": media, "emb_std": desv,
                            "epoch": epoca, "seed": semilla,
                            "validation_loss": lva,
                            "threshold": CONFIG["umbral"]}, ckpt)
            else:
                espera += 1
            if epoca == 1 or epoca % 10 == 0:
                print(f"{variante} seed={semilla} ep={epoca} "
                      f"train={ltr:.4f} val={lva:.4f} "
                      f"F1={hist[-1]['validation_f1']:.4f}", flush=True)
            if espera >= CONFIG["patience"]:
                break
        estado = torch.load(ckpt, map_location=dispositivo, weights_only=False)
        cabeza.load_state_dict(estado["head_state_dict"])
        ltr, ptr = evaluar_cabeza(cabeza, ztr_s, ytr, dispositivo, crit)
        lva, pva = evaluar_cabeza(cabeza, zva_s, yva, dispositivo, crit)
        resumen = {"variante": variante, "seed": semilla,
                   "parametros_cabeza": n_params,
                   "epoch_best": mejor_ep, "epochs_run": epoca,
                   "train_loss_best": ltr, "validation_loss_best": lva,
                   "train_metrics": metrica(ytr, ptr),
                   "validation_metrics": metrica(yva, pva),
                   "checkpoint": str(ckpt.relative_to(FUSION_DIR))}
        with open(sal / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(resumen, f, indent=2)
        with open(sal / "learning_curves.csv", "w", encoding="utf-8",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader()
            w.writerows(hist)
        with open(sal / "validation_predictions.csv", "w", encoding="utf-8",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "label",
                                              "probability_valid", "predicted"])
            w.writeheader()
            for vid, yy, pp in zip(dva["ids"], yva, pva):
                w.writerow({"video_id": vid, "label": int(yy),
                            "probability_valid": float(pp),
                            "predicted": int(pp >= CONFIG["umbral"])})
        np.save(sal / "validation_confusion.npy",
                np.asarray(resumen["validation_metrics"]["matriz_confusion_0_1"]))
        dibujar(hist, sal / "learning_curves.png")
        resultados.append(resumen)
        print(f"{variante} seed={semilla}: best={mejor_ep} val_loss={lva:.4f} "
              f"F1={resumen['validation_metrics']['f1']:.4f} "
              f"macro={resumen['validation_metrics']['macro_f1']:.4f} "
              f"AUC={resumen['validation_metrics']['roc_auc']:.4f}", flush=True)
        if lva < mejor_global:
            mejor_global, semilla_ganadora = lva, semilla
    with open(carpeta / "seeds_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    with open(carpeta / "selected_checkpoint.json", "w",
              encoding="utf-8") as f:
        json.dump({"seed": semilla_ganadora,
                   "checkpoint": f"seed_{semilla_ganadora}/best_validation_loss.pt",
                   "selection": "menor loss Validation entre semillas",
                   "threshold": CONFIG["umbral"]}, f, indent=2)
    return resultados


def main():
    torch.set_num_threads(4)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variante in ("full", "sin_relacion"):
        entrenar(variante, cargar_fusion(variante), dispositivo)
    print("FIN cabezas B1/B2. Test sin evaluar.", flush=True)


if __name__ == "__main__":
    main()
