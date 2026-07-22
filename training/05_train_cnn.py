import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "augmented" / "augmented_data.npz"
OUTPUT_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 15


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


def load_data():
    data = np.load(INPUT_PATH)
    X_train = torch.from_numpy(data["X_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).float()
    X_val = torch.from_numpy(data["X_val"]).float()
    y_val = torch.from_numpy(data["y_val"]).float()
    X_test = torch.from_numpy(data["X_test"]).float()
    y_test = torch.from_numpy(data["y_test"]).float()
    return X_train, y_train, X_val, y_val, X_test, y_test


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


def plot_training_curves(history, best_epoch):
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
    plt.savefig(RESULTS_DIR / "train_val_loss.png", dpi=150, bbox_inches="tight")
    print(f"Curvas guardadas en: {RESULTS_DIR / 'train_val_loss.png'}")
    plt.close()


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


def train():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    X_train = X_train.permute(0, 3, 1, 2)
    X_val = X_val.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = PoseCNN(T=X_train.shape[2], K=X_train.shape[3]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Iniciando entrenamiento por {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        val_acc = accuracy_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping en época {epoch}")
                break

    history = {k: np.array(v) for k, v in history.items()}
    analysis = analyze_curves(history)
    plot_training_curves(history, analysis["best_epoch"])
    print_analysis(analysis)

    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))

    _, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    print(f"\n{'='*50}")
    print(f"RESULTADOS EN TEST")
    print(f"{'='*50}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"F1-score:   {test_metrics['f1']:.4f}")
    if test_metrics["roc_auc"] is not None:
        print(f"ROC-AUC:    {test_metrics['roc_auc']:.4f}")
    print(f"Matriz de confusión:")
    cm = np.array(test_metrics["confusion_matrix"])
    print(cm)

    np.savez(OUTPUT_DIR / "history.npz", **history)
    print(f"\nHistorial guardado en: {OUTPUT_DIR / 'history.npz'}")
    print(f"Modelo guardado en: {OUTPUT_DIR / 'best_model.pth'}")

    return model, test_metrics, history, analysis


if __name__ == "__main__":
    train()
