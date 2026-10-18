import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.train_utils import (
    DEVICE,
    PoseCNN,
    analyze_curves,
    compute_metrics,
    evaluate,
    plot_training_curves,
    print_analysis,
    train_model,
)
from training.configs.training_config import parse_config


BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 8
WEIGHT_DECAY = 1e-4


def load_data(config):
    data = np.load(config["augmented_path"])
    return (
        torch.from_numpy(data["X_train"]).float(),
        torch.from_numpy(data["y_train"]).float(),
        torch.from_numpy(data["X_val"]).float(),
        torch.from_numpy(data["y_val"]).float(),
        torch.from_numpy(data["X_test"]).float(),
        torch.from_numpy(data["y_test"]).float(),
    )


def save_run_config(config, data_shape):
    metadata = {
        "profile": config["profile"],
        "display_name": config["display_name"],
        "target_frames": config["target_frames"],
        "keypoints": int(data_shape[3]),
        "channels": int(data_shape[1]),
        "split_strategy": config["split_strategy"],
        "augmentation": list(config["augmentation"]),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "random_seed": config["random_seed"],
    }

    path = config["model_dir"] / "run_config.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)


def main(config):
    config["model_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["random_seed"])

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    X_train = X_train.permute(0, 3, 1, 2)
    X_val = X_val.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
    )

    model = PoseCNN(T=X_train.shape[2], K=X_train.shape[3]).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    print(f"Perfil: {config['profile']} ({config['display_name']})")
    print(f"Dispositivo: {DEVICE}")
    print(f"Iniciando entrenamiento por {EPOCHS} epochs...")
    model, history, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        output_dir=config["model_dir"],
        fold_name="",
        verbose=True,
    )

    analysis = analyze_curves(history)
    plot_training_curves(
        history,
        analysis["best_epoch"],
        config["results_dir"],
    )
    print_analysis(analysis)

    _, test_preds, test_labels, test_probs = evaluate(
        model,
        test_loader,
        criterion,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)

    print(f"\n{'=' * 50}")
    print("RESULTADOS EN TEST")
    print(f"{'=' * 50}")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Precision:  {test_metrics['precision']:.4f}")
    print(f"Recall:     {test_metrics['recall']:.4f}")
    print(f"F1-score:   {test_metrics['f1']:.4f}")

    if test_metrics["roc_auc"] is not None:
        print(f"ROC-AUC:    {test_metrics['roc_auc']:.4f}")

    print("Matriz de confusion:")
    print(np.asarray(test_metrics["confusion_matrix"]))

    np.savez(config["history_path"], **history)
    save_run_config(config, X_train.shape)
    print(f"\nHistorial guardado en: {config['history_path']}")
    print(f"Modelo guardado en: {config['model_path']}")

    return model, test_metrics, history, analysis


if __name__ == "__main__":
    main(parse_config("Entrenamiento de PoseCNN"))
