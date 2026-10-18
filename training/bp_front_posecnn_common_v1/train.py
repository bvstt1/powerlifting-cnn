"""PoseCNN BP_FRONT reentrenada desde cero sobre cohorte común (sin Test).

python training/bp_front_posecnn_common_v1/train.py
"""
import csv
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from data import (RAIZ, POSE_DIR, COHORTE, leer_cohorte, cargar_desarrollo,
                  ajustar_scaler_train, augment_train, transformar)
from model import BPFrontPoseCNN, cargar_checkpoint

OUT = Path(__file__).resolve().parent
HISTORICO = RAIZ / "training/bp/bp_front_posecnn.py"
SEG = RAIZ / "training/bp_front_segcnn_v1"
CONFIG = {
    "arquitectura": "identica a BPFrontPoseCNN historica, sigmoid final y BCELoss",
    "fuente_arquitectura": "training/bp/bp_front_posecnn.py",
    "semillas": [42, 1337], "T": 170, "K": 6, "C": 3,
    "batch_size": 16, "max_epochs": 200, "patience": 15,
    "lr": 0.001, "weight_decay": 1e-4, "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau(val_loss, factor=0.5, patience=5)",
    "criterion": "BCELoss; sigmoid integrado; FP32, sin AMP",
    "threshold": 0.5, "checkpoint_rule": "mínima validation loss por seed",
    "seed_rule": "mínima validation loss entre checkpoints de las dos semillas",
    "augmentation": {
        "factor": 3, "probability_each": 0.5,
        "shift_xy": [-0.05, 0.05], "scale_xy": [0.9, 1.1],
        "noise_xy_std": 0.01, "rotation_deg": [-10, 10],
        "temporal_shift_steps": [-3, 3], "flip": False,
        "source": "training/bp/bp_front_posecnn.py: augment_sequence"
    },
    "preprocessing": "centro mediano hombros / distancia interhombros por video; "+
                     "170=linspace(0,T-1,170); scaler StandardScaler fit(train original), "+
                     "augmentation(train original) antes de transform; sin interpolar hueco bp_406",
    "test_access": "solo conteo/IDs del inventario; no se abre ningún .npy Test",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for bloque in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def json_file(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metricas(y, probs):
    pred = (probs >= 0.5).astype(int)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "valid_sensitivity": tp / (tp + fn),
        "invalid_specificity": tn / (tn + fp),
        "invalid_precision": tn / (tn + fn) if (tn + fn) else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "matrix_0_1": matrix.tolist(),
    }


def reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(x, y, shuffle=False, seed=42):
    t = TensorDataset(torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))),
                      torch.from_numpy(np.asarray(y, dtype=np.float32)))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(t, batch_size=CONFIG["batch_size"], shuffle=shuffle,
                      generator=g if shuffle else None, num_workers=0)


def evaluar(model, dataloader, device, criterion):
    model.eval()
    losses = 0.0
    total = 0
    pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            losses += float(loss.item()) * len(y)
            total += len(y)
            pred.extend(out.cpu().numpy().tolist())
    return losses / total, np.asarray(pred, dtype=np.float64)


def graficas(carpeta, history, y, probs, cm):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [r[name] for r in history], label=name)
    for name in ("train_f1", "validation_f1"):
        axes[1].plot(epochs, [r[name] for r in history], label=name)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BCELoss")
    axes[1].set_ylabel("F1/video")
    fig.tight_layout()
    fig.savefig(carpeta / "learning_curves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(probs[y == 0], bins=bins, alpha=0.65, label="Inválido (0)")
    ax.hist(probs[y == 1], bins=bins, alpha=0.65, label="Válido (1)")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set(xlabel="Probabilidad válido", ylabel="Videos Validation")
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


def entrenar(seed, xtrain, mtrain, ytrain, idtrain, xval, yval, idval, scaler, device):
    reloj = time.perf_counter()
    reproducible(seed)
    folder = OUT / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    augmented, aug_y = augment_train(xtrain, mtrain, ytrain, seed)
    train_loader = loader(transformar(augmented, scaler), aug_y, shuffle=True, seed=seed)
    orig_loader = loader(transformar(xtrain, scaler), ytrain)
    val_loader = loader(transformar(xval, scaler), yval)
    model = BPFrontPoseCNN().to(device)  # siempre inicialización nueva
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.BCELoss()
    best_loss = float("inf")
    best_epoch = 0
    waiting = 0
    history = []
    checkpoint = folder / "best_validation_loss.pt"
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        loss_train, prob_train = evaluar(model, orig_loader, device, criterion)
        loss_val, prob_val = evaluar(model, val_loader, device, criterion)
        scheduler.step(loss_val)
        lr = float(opt.param_groups[0]["lr"])
        fila = {"epoch": epoch, "train_loss": loss_train,
                "validation_loss": loss_val,
                "train_f1": metricas(ytrain, prob_train)["f1"],
                "validation_f1": metricas(yval, prob_val)["f1"],
                "learning_rate": lr}
        history.append(fila)
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            waiting = 0
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "epoch": epoch, "validation_loss": loss_val,
                        "threshold": CONFIG["threshold"],
                        "preprocessing": "train_scaler.npz"}, checkpoint)
        else:
            waiting += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Pose seed={seed} epoch={epoch} "
                  f"train={loss_train:.4f} val={loss_val:.4f} F1={fila['validation_f1']:.4f}",
                  flush=True)
        if waiting >= CONFIG["patience"]:
            break
    model, contenido = cargar_checkpoint(checkpoint, device)
    loss_train, prob_train = evaluar(model, orig_loader, device, criterion)
    loss_val, prob_val = evaluar(model, val_loader, device, criterion)
    resumen = {
        "seed": seed, "params": params, "best_epoch": best_epoch,
        "epochs_run": epoch, "train_time_sec": round(time.perf_counter() - reloj, 1),
        "learning_rate_final": lr, "train_loss": loss_train,
        "validation_loss": loss_val,
        "train_metrics": metricas(ytrain, prob_train),
        "validation_metrics": metricas(yval, prob_val),
        "checkpoint": str(checkpoint.relative_to(OUT)),
        "train_original_videos": len(ytrain),
        "train_optimization_sequences": len(aug_y),
        "validation_videos": len(yval),
    }
    json_file(folder / "metrics.json", resumen)
    csv_file(folder / "learning_curves.csv", history)
    csv_file(folder / "validation_predictions.csv", [
        {"video_id": video_id, "label": int(y), "probability_valid": float(p),
         "predicted": int(p >= CONFIG["threshold"])}
        for video_id, y, p in zip(idval, yval, prob_val)])
    np.save(folder / "validation_confusion.npy",
            np.asarray(resumen["validation_metrics"]["matrix_0_1"], dtype=np.int64))
    graficas(folder, history, yval, prob_val,
             resumen["validation_metrics"]["matrix_0_1"])
    print(f"Pose seed={seed} best={best_epoch} loss_val={loss_val:.4f} "
          f"F1={resumen['validation_metrics']['f1']:.4f} "
          f"AUC={resumen['validation_metrics']['roc_auc']:.4f} ", flush=True)
    return resumen


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    grupos = leer_cohorte()
    datos, auditoria = cargar_desarrollo(grupos)
    xtr, mtr, ytr, idtr = datos["train"]
    xva, mva, yva, idva = datos["validation"]
    if xtr.shape != (462, 170, 6, 3) or xva.shape != (99, 170, 6, 3):
        raise RuntimeError("Tensor inesperado")
    scaler = ajustar_scaler_train(xtr)
    np.savez_compressed(OUT / "train_scaler.npz",
                        mean=scaler.mean_.astype(np.float64),
                        scale=scaler.scale_.astype(np.float64),
                        var=scaler.var_.astype(np.float64),
                        n_samples=np.asarray([scaler.n_samples_seen_]))
    csv_file(OUT / "preprocessing_development_manifest.csv", auditoria)
    json_file(OUT / "training_config.json", CONFIG)
    fuentes = [COHORTE, HISTORICO,
               RAIZ / "models/bp/bp_front_skeleton_v8.pt",
               SEG / "full/selected_checkpoint.json",
               SEG / "sin_relacion/selected_checkpoint.json",
               SEG / "full/seed_1337/validation_predictions.csv",
               SEG / "sin_relacion/seed_42/validation_predictions.csv"]
    origen = {
        "fuentes_sha256": {str(p.relative_to(RAIZ)): sha256(p) for p in fuentes},
        "pose_development_sha256": {
            vid: sha256(POSE_DIR / f"{vid}.npy") for vid in idtr + idva},
        "pose_dir": str(POSE_DIR), "test_npys_opened": 0,
        "splits": {k: len(v) for k, v in grupos.items()},
        "python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "missing_original_total": sum(r["nan_values"] for r in auditoria),
        "missing_videos": [r["video_id"] for r in auditoria if r["nan_values"]],
    }
    json_file(OUT / "source_manifest.json", origen)
    print("Train 462 Val 99 Test 100 (sin abrir Test); Pose con NaN:",
          origen["missing_videos"], flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resultados = [entrenar(seed, xtr, mtr, ytr, idtr, xva, yva, idva,
                           scaler, device) for seed in CONFIG["semillas"]]
    json_file(OUT / "results_by_seed.json", resultados)
    elegido = min(resultados, key=lambda r: (r["validation_loss"], r["seed"]))
    json_file(OUT / "selected_checkpoint.json", {
        "seed": elegido["seed"], "path": elegido["checkpoint"],
        "selection": CONFIG["seed_rule"], "threshold": CONFIG["threshold"],
        "embedding_layer": "pool antes de classifier.Flatten", "embedding_dim": 128})
    # Hashes después de entrenar: se verifica que ninguna fuente se alteró.
    errores = [name for name, original in origen["fuentes_sha256"].items()
               if sha256(RAIZ / name) != original]
    errores += [v for v, original in origen["pose_development_sha256"].items()
                if sha256(POSE_DIR / f"{v}.npy") != original]
    json_file(OUT / "integrity_after_training.json", {
        "sources_unchanged": len(errores) == 0, "changed": errores,
        "test_npys_opened": 0,
        "checkpoints_sha256": {r["checkpoint"]: sha256(OUT / r["checkpoint"])
                                for r in resultados},
    })
    if errores:
        raise RuntimeError(f"Fuentes alteradas: {errores[:5]}")
    print("Seleccionado", elegido["checkpoint"], "Test reservado", flush=True)


if __name__ == "__main__":
    main()
