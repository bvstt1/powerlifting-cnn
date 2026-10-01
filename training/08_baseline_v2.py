"""Entrena y evalua PoseCNN Baseline V2 sin modificar artefactos historicos.

Uso:
    python training/08_baseline_v2.py --profile dl_left
    python training/08_baseline_v2.py --profile dl_right
    python training/08_baseline_v2.py --profile dl_side
    python training/08_baseline_v2.py --paired

El modo --paired se ejecuta despues de entrenar left y right.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from train_utils import (
    DEVICE,
    PoseCNN,
    analyze_curves,
    compute_metrics,
    evaluate,
    plot_training_curves,
    train_model,
    verify_arrays_finite,
    verify_no_duplicate_videos,
    verify_no_group_leakage,
    verify_split_shapes,
)
from training_config import get_config


EXPERIMENT = "baseline_v2"
MAPPING_VERSION = "physical_event_gap_310"
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 8
WEIGHT_DECAY = 1e-4
SEED = 42

HISTORICAL = {
    "dl_left": {"f1": 0.8438, "roc_auc": 0.9134, "note": "historico"},
    "dl_right": {"f1": 0.7576, "roc_auc": 0.7979, "note": "historico"},
    "dl_side": {
        "f1": 0.8219,
        "roc_auc": 0.8398,
        "note": "PRE-MAPPING-FIX",
    },
}


def experiment_paths(config):
    model_dir = config["model_dir"] / EXPERIMENT
    results_dir = config["results_dir"] / EXPERIMENT
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, results_dir


def load_pipeline_data(config):
    split = np.load(config["split_path"])
    data = np.load(config["augmented_path"])
    return split, data


def sanity_checks(config, split, data):
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    verify_arrays_finite(
        X_train, X_val, X_test,
        names=["X_train", "X_val", "X_test"],
    )
    verify_split_shapes(
        X_train, X_val, X_test,
        config["target_frames"], config["expected_keypoints"],
    )
    verify_no_group_leakage(
        split["groups_train"], split["groups_val"], split["groups_test"]
    )
    verify_no_duplicate_videos(
        split["ids_train"], split["ids_val"], split["ids_test"]
    )

    assert len(X_val) == len(split["ids_val"])
    assert len(X_test) == len(split["ids_test"])
    assert len(X_train) == 4 * len(split["ids_train"]), (
        "Train aumentado debe contener original + 3 aumentaciones"
    )

    if config["profile"] == "dl_side":
        normalized = np.load(config["normalized_path"])
        groups = normalized["group_ids"].astype(str)
        views = normalized["source_views"].astype(str)
        event_views = {}

        for group_id, view in zip(groups, views):
            event_views.setdefault(group_id, set()).add(view)

        paired = sum(views == {"left", "right"} for views in event_views.values())
        right_only = sum(views == {"right"} for views in event_views.values())
        left_only = sum(views == {"left"} for views in event_views.values())
        counts = (
            len(normalized["labels"]), len(event_views), paired,
            right_only, left_only,
        )
        expected_counts = (
            (814, 412, 402, 9, 1)
            if EXPERIMENT == "baseline_v3_manual_audit"
            else (816, 413, 403, 9, 1)
        )
        assert counts == expected_counts, (
            f"Estado inesperado de dl_side: {counts}"
        )
        split_counts = (
            len(split["X_train"]), len(np.unique(split["groups_train"])),
            len(split["X_val"]), len(np.unique(split["groups_val"])),
            len(split["X_test"]), len(np.unique(split["groups_test"])),
        )
        expected_split = (
            (570, 288, 122, 62, 122, 62)
            if EXPERIMENT == "baseline_v3_manual_audit"
            else (571, 289, 122, 62, 123, 62)
        )
        assert split_counts == expected_split, "Split dl_side inesperado"
        print(
            f"DL_SIDE confirmado: muestras/eventos/paired/right-only/left-only "
            f"= {expected_counts}"
        )

    print(
        f"SANITY {config['profile']}: NaN=0, Inf=0, leakage=0, "
        "duplicados=0, shapes=OK"
    )


def make_loader(X, y, shuffle=False):
    X_tensor = torch.from_numpy(np.asarray(X)).float().permute(0, 3, 1, 2)
    y_tensor = torch.from_numpy(np.asarray(y)).float()
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def metrics_with_specificity(labels, predictions, probabilities):
    metrics = compute_metrics(labels, predictions, probabilities)
    matrix = np.asarray(metrics["confusion_matrix"])
    assert matrix.shape == (2, 2), "Matriz de confusion debe ser 2x2"
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    metrics.update({
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })
    return metrics


def evaluate_loader(model, loader, criterion):
    _, predictions, labels, probabilities = evaluate(model, loader, criterion)
    metrics = metrics_with_specificity(labels, predictions, probabilities)
    return metrics, labels, predictions, probabilities


def save_predictions(path, ids, groups, views, labels, predictions, probabilities):
    rows = []

    for values in zip(ids, groups, views, labels, predictions, probabilities):
        video_id, group_id, view, true, predicted, probability = values
        rows.append({
            "video_id": str(video_id),
            "physical_event": str(group_id),
            "source_view": str(view),
            "true_label": int(true),
            "probability": float(probability),
            "prediction": int(predicted),
            "correct": int(true) == int(predicted),
        })

    fieldnames = list(rows[0].keys())

    with open(path / "predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(path / "misclassified.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row for row in rows if not row["correct"]])


def save_run_config(config, model_dir, data):
    metadata = {
        "experiment": EXPERIMENT,
        "mapping_version": MAPPING_VERSION,
        "profile": config["profile"],
        "display_name": config["display_name"],
        "seed": SEED,
        "target_frames": config["target_frames"],
        "keypoints": config["expected_keypoints"],
        "channels": 3,
        "split_strategy": config["split_strategy"],
        "augmentation": list(config["augmentation"]),
        "augmentation_factor": 3,
        "scaler_fit": "train_original_only",
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "dropout2d": 0.40,
        "train_augmented_samples": int(len(data["X_train"])),
        "validation_samples": int(len(data["X_val"])),
        "test_samples": int(len(data["X_test"])),
    }

    if EXPERIMENT == "baseline_v3_manual_audit":
        metadata.update({
            "manual_label_audit": True,
            "manual_audit_cases": 41,
            "manual_discarded_events": [302],
            "quality_flags_enabled": True,
        })

    with open(model_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)


def fmt_metrics(metrics):
    return (
        f"Acc={metrics['accuracy']:.4f} Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"AUC={metrics['roc_auc']:.4f} Specificity={metrics['specificity']:.4f} "
        f"TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )


def json_metrics(metrics):
    return {
        key: value if isinstance(value, list) else float(value)
        for key, value in metrics.items()
    }


def save_report(config, results_dir, val_metrics, test_metrics, analysis,
                best_epoch, final_epoch, sustained_divergence):
    historical = HISTORICAL[config["profile"]]
    report = f"""# {EXPERIMENT}: {config['display_name']}

- Experiment: `{EXPERIMENT}`
- Mapping: `{MAPPING_VERSION}`
- Seed: {SEED}
- Best epoch: {best_epoch}
- Final epoch: {final_epoch}
- Diagnostico curvas: {analysis['diagnosis']}
- Divergencia sostenida aproximada: {sustained_divergence}

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | {val_metrics['accuracy']:.4f} | {val_metrics['precision']:.4f} | {val_metrics['recall']:.4f} | {val_metrics['f1']:.4f} | {val_metrics['roc_auc']:.4f} | {val_metrics['specificity']:.4f} | {val_metrics['tn']} | {val_metrics['fp']} | {val_metrics['fn']} | {val_metrics['tp']} |
| Test | {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1']:.4f} | {test_metrics['roc_auc']:.4f} | {test_metrics['specificity']:.4f} | {test_metrics['tn']} | {test_metrics['fp']} | {test_metrics['fn']} | {test_metrics['tp']} |

## Comparacion Historica

| Perfil | F1 anterior | F1 Baseline V2 | AUC anterior | AUC Baseline V2 | Nota |
|---|---:|---:|---:|---:|---|
| {config['profile']} | {historical['f1']:.4f} | {test_metrics['f1']:.4f} | {historical['roc_auc']:.4f} | {test_metrics['roc_auc']:.4f} | {historical['note']} |

La comparacion no aisla cambios de arquitectura: cambiaron el mapping, el split y el orden metodologico del pipeline.
"""
    (results_dir / "report.md").write_text(report, encoding="utf-8")


def train_profile(profile):
    config = get_config(profile)
    model_dir, results_dir = experiment_paths(config)
    split, data = load_pipeline_data(config)
    sanity_checks(config, split, data)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader = make_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"])
    test_loader = make_loader(data["X_test"], data["y_test"])

    model = PoseCNN(
        T=config["target_frames"], K=config["expected_keypoints"]
    ).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"Entrenando {profile} desde cero en {DEVICE}...")
    model, history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        patience=PATIENCE,
        output_dir=model_dir,
        verbose=True,
    )

    final_epoch = len(history["train_loss"])
    analysis = analyze_curves(history)
    sustained_divergence = (
        best_epoch + 1
        if final_epoch > best_epoch
        and history["val_loss"][-1] > history["val_loss"][best_epoch - 1]
        and history["train_loss"][-1] < history["train_loss"][best_epoch - 1]
        else None
    )
    val_metrics, _, _, _ = evaluate_loader(model, val_loader, criterion)
    test_metrics, labels, predictions, probabilities = evaluate_loader(
        model, test_loader, criterion
    )

    np.savez(model_dir / "history.npz", **history)
    np.savez(results_dir / "metrics.npz", **{
        f"val_{key}": value for key, value in val_metrics.items()
        if key != "confusion_matrix"
    }, **{
        f"test_{key}": value for key, value in test_metrics.items()
        if key != "confusion_matrix"
    }, val_confusion_matrix=np.asarray(val_metrics["confusion_matrix"]),
       test_confusion_matrix=np.asarray(test_metrics["confusion_matrix"]),
       best_epoch=np.asarray([best_epoch]),
       final_epoch=np.asarray([final_epoch]))
    plot_training_curves(history, best_epoch, results_dir)
    save_run_config(config, model_dir, data)
    save_predictions(
        results_dir,
        split["ids_test"], split["groups_test"], split["views_test"],
        labels, predictions, probabilities,
    )
    save_report(
        config, results_dir, val_metrics, test_metrics, analysis,
        best_epoch, final_epoch, sustained_divergence,
    )

    print(f"VALIDATION {profile}: {fmt_metrics(val_metrics)}")
    print(f"TEST {profile}: {fmt_metrics(test_metrics)}")
    print(
        f"Epochs {profile}: best={best_epoch}, final={final_epoch}, "
        f"diagnostico={analysis['diagnosis']}, "
        f"divergencia_sostenida={sustained_divergence}"
    )


def load_experiment_model(profile):
    config = get_config(profile)
    model_dir, _ = experiment_paths(config)
    model = PoseCNN(
        T=config["target_frames"], K=config["expected_keypoints"]
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pth", map_location=DEVICE)
    )
    model.eval()
    return config, model


def paired_evaluation():
    event_data = np.load(get_config("dl_left")["data_dir"].parent / "event_split.npz")
    paired_events = set(event_data["paired_test_events"].astype(str))
    records = {}
    metrics = {}

    for profile in ("dl_left", "dl_right"):
        config, model = load_experiment_model(profile)
        split, data = load_pipeline_data(config)
        test_groups = split["groups_test"].astype(str)
        indices = np.asarray([
            index for index, group_id in enumerate(test_groups)
            if group_id in paired_events
        ])
        order = np.argsort(test_groups[indices])
        indices = indices[order]
        groups = test_groups[indices]

        assert set(groups) == paired_events
        assert len(groups) == len(np.unique(groups))

        loader = make_loader(data["X_test"][indices], data["y_test"][indices])
        result, labels, predictions, probabilities = evaluate_loader(
            model, loader, nn.BCELoss()
        )
        metrics[profile] = result
        records[profile] = {
            "events": groups,
            "videos": split["ids_test"][indices].astype(str),
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    left = records["dl_left"]
    right = records["dl_right"]
    assert np.array_equal(left["events"], right["events"]), (
        "left_event_ids != right_event_ids"
    )

    output_dir = Path(__file__).resolve().parent / "results" / "dl" / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "paired_predictions.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "physical_event", "left_video", "right_video", "left_true",
            "right_true", "left_probability", "right_probability",
            "left_prediction", "right_prediction",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, event in enumerate(left["events"]):
            writer.writerow({
                "physical_event": event,
                "left_video": left["videos"][index],
                "right_video": right["videos"][index],
                "left_true": int(left["labels"][index]),
                "right_true": int(right["labels"][index]),
                "left_probability": float(left["probabilities"][index]),
                "right_probability": float(right["probabilities"][index]),
                "left_prediction": int(left["predictions"][index]),
                "right_prediction": int(right["predictions"][index]),
            })

    summary = {
        "experiment": EXPERIMENT,
        "mapping_version": MAPPING_VERSION,
        "n": len(left["events"]),
        "left": json_metrics(metrics["dl_left"]),
        "right": json_metrics(metrics["dl_right"]),
    }
    (output_dir / "paired_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Paired event IDs left == right: OK (N={len(left['events'])})")
    print(f"LEFT PAIRED TEST: {fmt_metrics(metrics['dl_left'])}")
    print(f"RIGHT PAIRED TEST: {fmt_metrics(metrics['dl_right'])}")
    print(f"Predicciones pareadas: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PoseCNN Baseline V2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", choices=("dl_left", "dl_right", "dl_side"))
    group.add_argument("--paired", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.paired:
        paired_evaluation()
    else:
        train_profile(args.profile)
