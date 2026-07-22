import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "normalized" / "sequences_170.npz"
OUTPUT_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 15
VAL_SPLIT = 0.15
AUG_FACTOR = 3
RANDOM_SEED = 42


class PoseCNN(nn.Module):
    def __init__(self, T=170, K=25, C=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())
    return (total_loss / len(loader),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def augment_data(X, y, rng):
    X_aug, y_aug = [X.copy()], [y.copy()]
    for _ in range(AUG_FACTOR):
        X_new = X.copy()
        for i in range(len(X)):
            seq = X_new[i]
            if rng.rand() > 0.5:
                shift = rng.uniform(-0.05, 0.05, size=2)
                seq[..., 0] += shift[0]
                seq[..., 1] += shift[1]
            if rng.rand() > 0.5:
                scale = rng.uniform(0.9, 1.1)
                seq[..., :2] *= scale
            if rng.rand() > 0.5:
                noise = rng.normal(0, 0.01, size=seq[..., :2].shape).astype(np.float32)
                seq[..., :2] += noise
            if rng.rand() > 0.5:
                angle = np.deg2rad(rng.uniform(-10, 10))
                c, s = np.cos(angle), np.sin(angle)
                x_coords = seq[..., 0].copy()
                y_coords = seq[..., 1].copy()
                seq[..., 0] = x_coords * c - y_coords * s
                seq[..., 1] = x_coords * s + y_coords * c
            if rng.rand() > 0.5:
                shift_t = rng.randint(-3, 4)
                if shift_t != 0:
                    seq = np.roll(seq, shift_t, axis=0)
                    if shift_t > 0:
                        seq[:shift_t] = seq[shift_t]
                    else:
                        seq[shift_t:] = seq[-1]
            X_new[i] = seq
        X_aug.append(X_new)
        y_aug.append(y.copy())
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def train_fold_model(X_train_fold, y_train_fold, X_val, y_val):
    rng = np.random.RandomState(RANDOM_SEED)
    X_train_aug, y_train_aug = augment_data(X_train_fold, y_train_fold, rng)

    X_train_t = torch.from_numpy(X_train_aug).float().permute(0, 3, 1, 2)
    y_train_t = torch.from_numpy(y_train_aug).float()
    X_val_t = torch.from_numpy(X_val).float().permute(0, 3, 1, 2)
    y_val_t = torch.from_numpy(y_val).float()

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = PoseCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimizer)
        val_loss, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def run_cross_validation():
    data = np.load(INPUT_PATH)
    X = data["keypoints"]
    y = data["labels"]

    print(f"\n{'='*60}")
    print(f"VALIDACIÓN CRUZADA {N_FOLDS}-FOLD")
    print(f"{'='*60}")
    print(f"Total muestras: {len(X)}")
    print(f"Válidos: {sum(y==1)}, Inválidos: {sum(y==0)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"{'='*50}")

        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]

        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_fold, y_train_fold, test_size=VAL_SPLIT,
            stratify=y_train_fold, random_state=RANDOM_SEED
        )

        print(f"  Train: {len(X_tr)} (válidos={sum(y_tr==1)}, inválidos={sum(y_tr==0)})")
        print(f"  Val:   {len(X_val)} (válidos={sum(y_val==1)}, inválidos={sum(y_val==0)})")
        print(f"  Test:  {len(X_test_fold)} (válidos={sum(y_test_fold==1)}, inválidos={sum(y_test_fold==0)})")

        model = train_fold_model(X_tr, y_tr, X_val, y_val)

        X_test_t = torch.from_numpy(X_test_fold).float().permute(0, 3, 1, 2)
        y_test_t = torch.from_numpy(y_test_fold).float()
        test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

        criterion = nn.BCELoss()
        _, preds, labels, probs = evaluate(model, test_loader, criterion)

        metrics = {
            "fold": fold,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }
        if len(np.unique(labels)) > 1:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        else:
            metrics["roc_auc"] = None

        fold_metrics.append(metrics)
        print(f"  → Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
              f"ROC-AUC={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")

    print(f"\n{'='*60}")
    print(f"RESULTADOS {N_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for m in fold_metrics:
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        print(f"{m['fold']:<6} {m['accuracy']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {auc_str:<10}")

    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    avg = {k: np.mean([m[k] for m in fold_metrics if m[k] is not None]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    std = {k: np.std([m[k] for m in fold_metrics if m[k] is not None]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    print(f"{'Promedio':<6} {avg['accuracy']:<10.4f} {avg['precision']:<10.4f} {avg['recall']:<10.4f} {avg['f1']:<10.4f} {avg['roc_auc']:<10.4f}")
    print(f"{'Std':<6} {std['accuracy']:<10.4f} {std['precision']:<10.4f} {std['recall']:<10.4f} {std['f1']:<10.4f} {std['roc_auc']:<10.4f}")

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE ESTABILIDAD")
    print(f"{'='*60}")

    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in metrics_names:
        vals = [m[metric] for m in fold_metrics if m[metric] is not None]
        if not vals:
            continue
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        cv = std_v / mean_v if mean_v > 0 else 0
        print(f"\n{metric.upper()}: {mean_v:.4f} ± {std_v:.4f} (CV={cv:.4f})")
        if cv < 0.05:
            print(f"  → Baja variabilidad. El modelo es estable para {metric}.")
        elif cv < 0.10:
            print(f"  → Variabilidad moderada. Aceptable.")
        else:
            print(f"  → Alta variabilidad. El modelo depende del split.")

    min_f1 = min([m["f1"] for m in fold_metrics])
    max_f1 = max([m["f1"] for m in fold_metrics])
    range_f1 = max_f1 - min_f1
    print(f"\nRango F1-score: {min_f1:.4f} - {max_f1:.4f} (Δ={range_f1:.4f})")
    if range_f1 < 0.05:
        print("→ Rango muy estrecho: el modelo es consistente entre folds.")
    elif range_f1 < 0.10:
        print("→ Rango aceptable: cierta dependencia del split.")
    else:
        print("→ Rango amplio: el modelo depende demasiado del split.")

    overall_cv = np.mean([np.std([m[k] for m in fold_metrics if m[k] is not None]) /
                          (np.mean([m[k] for m in fold_metrics if m[k] is not None]) + 1e-8)
                          for k in metrics_names if any(m[k] is not None for m in fold_metrics)])
    if overall_cv < 0.05:
        print("\n CONCLUSIÓN: El modelo generaliza bien y es estable entre folds.")
    elif overall_cv < 0.10:
        print("\n CONCLUSIÓN: El modelo generaliza aceptablemente, pero con cierta variabilidad.")
    else:
        print("\n CONCLUSIÓN: Existe alta variabilidad entre folds. Considerar aumentar el dataset.")

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = []
    labels_plot = []
    for metric in metrics_names:
        vals = [m[metric] for m in fold_metrics if m[metric] is not None]
        if vals:
            data_to_plot.append(vals)
            labels_plot.append(metric.upper())

    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, vals in enumerate(data_to_plot):
        y = vals
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, color="black", alpha=0.6, zorder=5)

    ax.set_ylabel("Score")
    ax.set_title(f"Validación Cruzada {N_FOLDS}-Fold", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    boxplot_path = RESULTS_DIR / "cross_validation_boxplot.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    print(f"\nBoxplot guardado en: {boxplot_path}")
    plt.close()

    return fold_metrics, avg, std


if __name__ == "__main__":
    run_cross_validation()
