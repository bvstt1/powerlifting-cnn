import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from train_utils import (
    PoseCNN, DEVICE, train_model, evaluate, compute_metrics,
    analyze_curves, print_analysis, plot_training_curves,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "augmented" / "augmented_data.npz"
OUTPUT_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 15


def load_data():
    data = np.load(INPUT_PATH)
    X_train = torch.from_numpy(data["X_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).float()
    X_val = torch.from_numpy(data["X_val"]).float()
    y_val = torch.from_numpy(data["y_val"]).float()
    X_test = torch.from_numpy(data["X_test"]).float()
    y_test = torch.from_numpy(data["y_test"]).float()
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
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

    print(f"Iniciando entrenamiento por {EPOCHS} epochs...")
    model, history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        output_dir=OUTPUT_DIR,
        fold_name="",
        verbose=True,
    )

    analysis = analyze_curves(history)
    plot_training_curves(history, analysis["best_epoch"], RESULTS_DIR)
    print_analysis(analysis)

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
    main()
