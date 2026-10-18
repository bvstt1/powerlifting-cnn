import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import copy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def best_group_split(y, groups, test_size, random_seed, attempts=200):
    indices = np.arange(len(y))
    global_ratio = float(np.mean(y))
    best_split = None
    best_score = float("inf")

    for attempt in range(attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_seed + attempt,
        )
        train_idx, test_idx = next(splitter.split(indices, y, groups))
        size_error = abs((len(test_idx) / len(y)) - test_size)
        ratio_error = abs(float(np.mean(y[test_idx])) - global_ratio)
        score = size_error + ratio_error

        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)

    return best_split


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
            nn.Dropout2d(0.4),
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
    return total_loss / len(loader), correct / total


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


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, patience, output_dir=None, fold_name="", verbose=True):
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None

    prefix = f"[{fold_name}] " if fold_name else ""

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        val_acc = accuracy_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"{prefix}Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            if output_dir is not None:
                save_name = "best_model.pth" if not fold_name else f"best_model_{fold_name}.pth"
                torch.save(model.state_dict(), output_dir / save_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"{prefix}Early stopping en época {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {k: np.array(v) for k, v in history.items()}
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    return model, history, best_epoch


# ─── Augmentation ────────────────────────────────────────────────────────────

def add_translation(seq, rng, max_shift=0.05):
    shift_x = rng.uniform(-max_shift, max_shift)
    shift_y = rng.uniform(-max_shift, max_shift)
    aug = seq.copy()
    aug[..., 0] += shift_x
    aug[..., 1] += shift_y
    return aug


def add_scaling(seq, rng, scale_range=(0.9, 1.1)):
    scale = rng.uniform(*scale_range)
    aug = seq.copy()
    aug[..., :2] *= scale
    return aug


def add_gaussian_noise(seq, rng, noise_std=0.01):
    noise = rng.normal(0, noise_std, size=seq[..., :2].shape).astype(np.float32)
    aug = seq.copy()
    aug[..., :2] += noise
    return aug


def add_rotation(seq, rng, max_angle=10):
    angle = rng.uniform(-max_angle, max_angle)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    aug = seq.copy()
    x = aug[..., 0].copy()
    y = aug[..., 1].copy()
    aug[..., 0] = x * cos_a - y * sin_a
    aug[..., 1] = x * sin_a + y * cos_a
    return aug


def add_temporal_shift(seq, rng, max_shift=3):
    T = seq.shape[0]
    shift = rng.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return seq.copy()
    aug = np.roll(seq, shift, axis=0)
    if shift > 0:
        aug[:shift] = seq[0]
    else:
        aug[shift:] = seq[-1]
    return aug


def augment_sequence(seq, label, rng, aug_factor, transforms=None):
    if transforms is None:
        transforms = (
            "translation",
            "scaling",
            "noise",
            "rotation",
            "temporal_shift",
        )

    results = [(seq, label)]
    for _ in range(aug_factor):
        aug = seq.copy()
        if "translation" in transforms and rng.rand() > 0.5:
            aug = add_translation(aug, rng)
        if "scaling" in transforms and rng.rand() > 0.5:
            aug = add_scaling(aug, rng)
        if "noise" in transforms and rng.rand() > 0.5:
            aug = add_gaussian_noise(aug, rng)
        if "rotation" in transforms and rng.rand() > 0.5:
            aug = add_rotation(aug, rng)
        if "temporal_shift" in transforms and rng.rand() > 0.5:
            aug = add_temporal_shift(aug, rng)
        results.append((aug, label))
    return results


def augment_dataset(X, y, rng, aug_factor, transforms=None):
    all_X, all_y = [], []
    for i in range(len(X)):
        augmented = augment_sequence(
            X[i],
            y[i],
            rng,
            aug_factor,
            transforms,
        )
        for kp, lbl in augmented:
            all_X.append(kp)
            all_y.append(lbl)
    return np.array(all_X, dtype=np.float32), np.array(all_y)


# ─── Verificacion de splits ─────────────────────────────────────────────────

def verify_no_group_leakage(groups_train, groups_val, groups_test):
    train_groups = set(np.asarray(groups_train).astype(str).tolist())
    val_groups = set(np.asarray(groups_val).astype(str).tolist())
    test_groups = set(np.asarray(groups_test).astype(str).tolist())

    assert not (train_groups & val_groups), (
        f"Fuga de eventos entre train y val: "
        f"{sorted(train_groups & val_groups)[:10]}"
    )
    assert not (train_groups & test_groups), (
        f"Fuga de eventos entre train y test: "
        f"{sorted(train_groups & test_groups)[:10]}"
    )
    assert not (val_groups & test_groups), (
        f"Fuga de eventos entre val y test: "
        f"{sorted(val_groups & test_groups)[:10]}"
    )
    return train_groups, val_groups, test_groups


def verify_no_duplicate_videos(ids_train, ids_val, ids_test):
    train_ids = np.asarray(ids_train).astype(str)
    val_ids = np.asarray(ids_val).astype(str)
    test_ids = np.asarray(ids_test).astype(str)

    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        assert len(ids) == len(np.unique(ids)), (
            f"Videos duplicados dentro de {name}"
        )

    assert not (set(train_ids) & set(val_ids)), "Video repetido entre train y val"
    assert not (set(train_ids) & set(test_ids)), "Video repetido entre train y test"
    assert not (set(val_ids) & set(test_ids)), "Video repetido entre val y test"


def verify_arrays_finite(*arrays, names=None):
    names = names or [f"array_{i}" for i in range(len(arrays))]

    for name, array in zip(names, arrays):
        array = np.asarray(array)
        assert not np.isnan(array).any(), f"NaN detectado en {name}"
        assert not np.isinf(array).any(), f"Inf detectado en {name}"


def verify_split_shapes(X_train, X_val, X_test, target_frames, expected_keypoints):
    for name, array in (
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ):
        array = np.asarray(array)
        assert array.ndim == 4, f"{name} debe ser 4D (N,T,K,C): {array.shape}"
        assert array.shape[1] == target_frames, (
            f"{name} frames={array.shape[1]}, esperado={target_frames}"
        )
        assert array.shape[2] == expected_keypoints, (
            f"{name} keypoints={array.shape[2]}, esperado={expected_keypoints}"
        )
        assert array.shape[3] == 3, f"{name} canales={array.shape[3]}, esperado=3"


def summarize_partition(name, labels, groups):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    print(
        f"{name:<6}: {len(labels)} muestras "
        f"(validos={int(sum(labels == 1))}, "
        f"invalidos={int(sum(labels == 0))}, "
        f"eventos={len(np.unique(groups))})"
    )


# ─── Scaling ────────────────────────────────────────────────────────────────

def create_scaler(X_train):
    N = X_train.shape[0]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(N, -1))
    return scaler


def scale_data(scaler, X):
    N = X.shape[0]
    return scaler.transform(X.reshape(N, -1)).reshape(X.shape)


# ─── Analysis ───────────────────────────────────────────────────────────────

def analyze_curves(history):
    analysis = {}
    train_loss = np.array(history["train_loss"])
    val_loss = np.array(history["val_loss"])
    best_epoch = int(np.argmin(val_loss)) + 1
    analysis["best_epoch"] = best_epoch
    analysis["best_val_loss"] = float(val_loss[best_epoch - 1])

    last_train = train_loss[-1]
    last_val = val_loss[-1]
    min_val = val_loss.min()

    gap = abs(last_train - last_val)

    if last_val > min_val * 1.08 and last_train < train_loss[best_epoch - 1] * 0.95:
        analysis["diagnosis"] = "OVERFITTING"
        divergence_epoch = None
        for i in range(1, len(val_loss)):
            if val_loss[i] > val_loss[i - 1] and val_loss[i] > min_val * 1.03:
                divergence_epoch = i
                break
        if divergence_epoch is None:
            for i in range(1, len(val_loss)):
                if val_loss[i] > val_loss[i - 1] and train_loss[i] < train_loss[i - 1]:
                    divergence_epoch = i
                    break
        analysis["divergence_epoch"] = divergence_epoch
    elif last_train > 0.4 and last_val > 0.4:
        analysis["diagnosis"] = "UNDERFITTING"
        analysis["divergence_epoch"] = None
    elif gap > 0.15:
        analysis["diagnosis"] = "POSIBLE OVERFITTING"
        analysis["divergence_epoch"] = best_epoch
    else:
        analysis["diagnosis"] = "NORMAL"
        analysis["divergence_epoch"] = None

    return analysis


def print_analysis(analysis):
    print(f"\n{'='*50}")
    print(f"ANÁLISIS DE CURVAS DE ENTRENAMIENTO")
    print(f"{'='*50}")
    print(f"Mejor época (menor val_loss): {analysis['best_epoch']}")
    print(f"Mejor val_loss: {analysis['best_val_loss']:.4f}")
    print(f"Diagnóstico: {analysis['diagnosis']}")
    if analysis["divergence_epoch"]:
        print(f"El sobreajuste comenzó aproximadamente en época {analysis['divergence_epoch']}")
    if analysis["diagnosis"] == "OVERFITTING":
        print("- El validation loss sube mientras train loss sigue bajando.")
        print("- El modelo memoriza el train pero no generaliza a val.")
    elif analysis["diagnosis"] == "UNDERFITTING":
        print("- Ambas perdidas se mantienen altas.")
        print("- El modelo no esta aprendiendo lo suficiente.")
    elif analysis["diagnosis"] == "POSIBLE OVERFITTING":
        print("- Hay cierta divergencia entre train y val loss.")
        print("- Monitorear en futuros entrenamientos.")
    else:
        print("- No hay evidencia de overfitting ni underfitting significativos.")
        print("- El modelo generaliza correctamente.")
    print(f"{'='*50}\n")


def plot_training_curves(history, best_epoch, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5722")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Mejor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss durante entrenamiento")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2196F3")
    ax.plot(epochs, history["val_acc"], label="Val Accuracy", color="#FF5722")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Mejor época ({best_epoch})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy durante entrenamiento")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "train_val_loss.png", dpi=150, bbox_inches="tight")
    print(f"Curvas guardadas en: {save_dir / 'train_val_loss.png'}")
    plt.close()
