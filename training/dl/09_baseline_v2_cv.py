"""5-Fold CV de Baseline V2 usando solo train+validation originales.

El test de Baseline V2 queda protegido y nunca se escala, aumenta ni evalua.
Todos los perfiles reutilizan un unico physical_event -> fold_id.

Uso:
    python -m training.dl.09_baseline_v2_cv --prepare
    python -m training.dl.09_baseline_v2_cv --profile dl_left
    python -m training.dl.09_baseline_v2_cv --profile dl_right
    python -m training.dl.09_baseline_v2_cv --profile dl_side
    python -m training.dl.09_baseline_v2_cv --paired
    python -m training.dl.09_baseline_v2_cv --summary
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common.train_utils import (
    DEVICE,
    PoseCNN,
    augment_dataset,
    compute_metrics,
    create_scaler,
    evaluate,
    plot_training_curves,
    scale_data,
    train_model,
    verify_arrays_finite,
    verify_no_duplicate_videos,
    verify_split_shapes,
)
from training.configs.training_config import get_config


EXPERIMENT = "baseline_v2_cv"
MAPPING_VERSION = "physical_event_gap_310"
N_FOLDS = 5
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 8
WEIGHT_DECAY = 1e-4
AUG_FACTOR = 3
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parents[1]
COMMON_FOLDS_PATH = SCRIPT_DIR / "data" / "dl" / "cv_event_folds.csv"
COMMON_RESULTS_DIR = SCRIPT_DIR / "results" / "dl" / EXPERIMENT

PROFILES = ("dl_left", "dl_right", "dl_side")
METRIC_NAMES = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "specificity"
)

HISTORICAL_CV = {
    "dl_left": {
        "accuracy": (0.782, 0.039),
        "f1": (0.793, 0.042),
        "roc_auc": (0.861, 0.025),
    },
    "dl_right": {
        "accuracy": (0.772, 0.034),
        "f1": (0.800, 0.032),
        "roc_auc": (0.842, 0.033),
    },
    "dl_side": {
        "accuracy": None,
        "f1": (0.796, 0.046),
        "roc_auc": (0.848, 0.031),
    },
}


def profile_paths(config):
    model_dir = config["model_dir"] / EXPERIMENT
    results_dir = config["results_dir"] / EXPERIMENT
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, results_dir


def baseline_partitions(profile):
    config = get_config(profile)
    split = np.load(config["split_path"])
    dev = set(split["groups_train"].astype(str)) | set(
        split["groups_val"].astype(str)
    )
    test = set(split["groups_test"].astype(str))
    assert not (dev & test), f"{profile}: leakage dev/test en Baseline V2"
    return dev, test


def normalized_label_map(profile):
    config = get_config(profile)
    data = np.load(config["normalized_path"])
    groups = data["group_ids"].astype(str)
    labels = data["labels"].astype(int)

    if profile in ("dl_left", "dl_right"):
        assert len(groups) == len(np.unique(groups)), f"{profile}: evento duplicado"
        return dict(zip(groups, labels))

    return None


def audited_development_universe():
    partitions = {profile: baseline_partitions(profile) for profile in PROFILES}
    side_dev, side_test = partitions["dl_side"]
    right_dev, right_test = partitions["dl_right"]
    left_dev, left_test = partitions["dl_left"]

    # El split fisico es comun. Las diferencias de cardinalidad provienen de
    # muestras descartadas por preprocessing, no de otra asignacion.
    assert right_test == side_test, "right/side no protegen el mismo test"
    assert left_test <= side_test, "left contiene test fuera del test comun"
    assert left_dev <= side_dev and right_dev <= side_dev
    expected_dev = 350 if EXPERIMENT == "baseline_v3_manual_audit_cv" else 351
    assert len(side_dev) == expected_dev and len(side_test) == 62
    assert not (side_dev & side_test)

    return side_dev, side_test, partitions


def prepare_common_folds():
    dev_events, protected_test, _ = audited_development_universe()
    left_labels = normalized_label_map("dl_left")
    right_labels = normalized_label_map("dl_right")
    events = np.asarray(sorted(dev_events))
    strata = []
    rows = []

    for event in events:
        left_label = left_labels.get(event)
        right_label = right_labels.get(event)

        if left_label is not None and right_label is not None:
            stratum = f"{left_label}{right_label}"
        elif left_label is not None:
            stratum = f"left_only_label_{left_label}"
        else:
            assert right_label is not None
            stratum = f"right_only_label_{right_label}"

        strata.append(stratum)
        rows.append({
            "physical_event": event,
            "stratum": stratum,
            "left_label": "" if left_label is None else int(left_label),
            "right_label": "" if right_label is None else int(right_label),
        })

    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    fold_ids = np.full(len(events), -1, dtype=int)

    for fold, (_, val_indices) in enumerate(splitter.split(events, strata)):
        fold_ids[val_indices] = fold

    assert (fold_ids >= 0).all()
    assert len(events) == len(np.unique(events))
    assert not (set(events) & protected_test)

    for index, fold in enumerate(fold_ids):
        rows[index]["fold_id"] = int(fold)

    output = pd.DataFrame(rows)[[
        "physical_event", "fold_id", "stratum", "left_label", "right_label"
    ]]
    COMMON_FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(COMMON_FOLDS_PATH, index=False)

    print(f"Folds comunes guardados en: {COMMON_FOLDS_PATH}")
    print(f"Development: {len(events)} eventos | Test protegido: {len(protected_test)}")
    print("Distribucion por fold/estrato:")
    print(pd.crosstab(output["fold_id"], output["stratum"]).to_string())
    print("Eventos por fold:", output["fold_id"].value_counts().sort_index().to_dict())
    return output


def load_fold_map():
    if not COMMON_FOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Falta {COMMON_FOLDS_PATH}. Ejecutar primero --prepare."
        )

    folds = pd.read_csv(COMMON_FOLDS_PATH, dtype={"physical_event": str})
    expected_folds = 350 if EXPERIMENT == "baseline_v3_manual_audit_cv" else 351
    assert len(folds) == expected_folds
    assert folds["physical_event"].is_unique
    assert set(folds["fold_id"]) == set(range(N_FOLDS))

    dev_events, protected_test, _ = audited_development_universe()
    assert set(folds["physical_event"]) == dev_events
    assert not (set(folds["physical_event"]) & protected_test)
    return folds


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
    matrix = np.asarray(metrics.pop("confusion_matrix"))
    assert matrix.shape == (2, 2)
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    metrics.update({
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })
    return metrics, matrix


def evaluate_validation(model, loader):
    _, predictions, labels, probabilities = evaluate(
        model, loader, nn.BCELoss()
    )
    metrics, matrix = metrics_with_specificity(
        labels, predictions, probabilities
    )
    return metrics, matrix, labels, predictions, probabilities


def fold_seed():
    # Misma semilla maestra e inicializacion para todos los folds/perfiles.
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def print_fold_counts(profile, fold, train_groups, val_groups, y_train, y_val):
    print(f"\n{profile} FOLD {fold}")
    print(f"  N train events:  {len(np.unique(train_groups))}")
    print(f"  N val events:    {len(np.unique(val_groups))}")
    print(f"  N train samples: {len(y_train)}")
    print(f"  N val samples:   {len(y_val)}")
    print(
        f"  Train V/I: {int(sum(y_train == 1))}/{int(sum(y_train == 0))}"
    )
    print(f"  Val V/I:   {int(sum(y_val == 1))}/{int(sum(y_val == 0))}")


def summarize_results(results):
    summary = {}

    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in results], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }

    summary["best_epoch"] = [int(row["best_epoch"]) for row in results]
    summary["final_epoch"] = [int(row["final_epoch"]) for row in results]
    return summary


def run_config(profile):
    config = get_config(profile)
    metadata = {
        "experiment": EXPERIMENT,
        "mapping_version": MAPPING_VERSION,
        "profile": profile,
        "development_only": True,
        "test_used": False,
        "common_fold_map": str(COMMON_FOLDS_PATH),
        "folds": N_FOLDS,
        "seed": SEED,
        "fold_seed_policy": "reset_to_42_each_fold",
        "target_frames": config["target_frames"],
        "keypoints": config["expected_keypoints"],
        "channels": 3,
        "augmentation": list(config["augmentation"]),
        "augmentation_factor": AUG_FACTOR,
        "scaler_fit": "fold_train_original_only",
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "dropout2d": 0.40,
    }

    if EXPERIMENT == "baseline_v3_manual_audit_cv":
        metadata.update({
            "manual_label_audit": True,
            "manual_audit_cases": 41,
            "manual_discarded_events": [302],
            "quality_flags_enabled": True,
        })

    return metadata


def train_profile(profile):
    folds = load_fold_map()
    fold_of = dict(zip(folds["physical_event"], folds["fold_id"]))
    config = get_config(profile)
    model_dir, results_dir = profile_paths(config)
    normalized = np.load(config["normalized_path"])
    split = np.load(config["split_path"])

    X = np.asarray(normalized["keypoints"], dtype=np.float32)
    y = np.asarray(normalized["labels"], dtype=int)
    ids = normalized["video_ids"].astype(str)
    groups = normalized["group_ids"].astype(str)
    views = normalized["source_views"].astype(str)
    dev_events = set(split["groups_train"].astype(str)) | set(
        split["groups_val"].astype(str)
    )
    protected_test = set(split["groups_test"].astype(str))

    assert not (dev_events & protected_test)
    assert dev_events <= set(fold_of)
    dev_mask = np.asarray([group in dev_events for group in groups])
    assert not any(group in protected_test for group in groups[dev_mask])
    X_dev = X[dev_mask]
    y_dev = y[dev_mask]
    ids_dev = ids[dev_mask]
    groups_dev = groups[dev_mask]
    views_dev = views[dev_mask]
    verify_arrays_finite(X_dev, names=[f"{profile}_development"])

    fold_assignments = np.asarray([fold_of[group] for group in groups_dev])
    results = []
    prediction_rows = []

    for fold in range(N_FOLDS):
        val_mask = fold_assignments == fold
        train_mask = ~val_mask
        X_train = X_dev[train_mask]
        X_val = X_dev[val_mask]
        y_train = y_dev[train_mask]
        y_val = y_dev[val_mask]
        ids_train = ids_dev[train_mask]
        ids_val = ids_dev[val_mask]
        groups_train = groups_dev[train_mask]
        groups_val = groups_dev[val_mask]
        views_val = views_dev[val_mask]

        train_events = set(groups_train)
        val_events = set(groups_val)
        assert not (train_events & val_events)
        assert not ((train_events | val_events) & protected_test)
        verify_no_duplicate_videos(ids_train, ids_val, [])
        verify_arrays_finite(
            X_train, X_val, names=["fold_train_original", "fold_validation"]
        )
        verify_split_shapes(
            X_train, X_val, X_val,
            config["target_frames"], config["expected_keypoints"],
        )
        assert X_train.shape[1:] == (170, 2, 3)
        assert X_val.shape[1:] == (170, 2, 3)
        assert set(np.unique(y_train)) == {0, 1}
        assert set(np.unique(y_val)) == {0, 1}
        print_fold_counts(
            profile, fold, groups_train, groups_val, y_train, y_val
        )

        # Orden obligatorio: fit(train original) -> augmentation(train) ->
        # transform(train aumentado / validation).
        scaler = create_scaler(X_train)
        assert int(scaler.n_samples_seen_) == len(X_train)
        rng = np.random.RandomState(SEED)
        X_train_aug, y_train_aug = augment_dataset(
            X_train,
            y_train,
            rng,
            AUG_FACTOR,
            transforms=config["augmentation"],
        )
        assert len(X_train_aug) == len(X_train) * (AUG_FACTOR + 1)
        X_train_scaled = scale_data(scaler, X_train_aug).astype(np.float32)
        X_val_scaled = scale_data(scaler, X_val).astype(np.float32)
        verify_arrays_finite(
            X_train_scaled, X_val_scaled,
            names=["fold_train_scaled", "fold_validation_scaled"],
        )
        verify_split_shapes(
            X_train_scaled, X_val_scaled, X_val_scaled,
            config["target_frames"], config["expected_keypoints"],
        )
        print(
            "  Assertions: leakage=0, duplicados=0, NaN=0, Inf=0, "
            "shape=(N,170,2,3), scaler=train_original"
        )

        fold_seed()
        train_loader = make_loader(X_train_scaled, y_train_aug, shuffle=True)
        val_loader = make_loader(X_val_scaled, y_val)
        model = PoseCNN(T=170, K=2, C=3).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        fold_model_dir = model_dir / f"fold_{fold}"
        fold_results_dir = results_dir / f"fold_{fold}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        fold_results_dir.mkdir(parents=True, exist_ok=True)
        model, history, best_epoch = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=EPOCHS,
            patience=PATIENCE,
            output_dir=fold_model_dir,
            fold_name="",
            verbose=True,
        )
        final_epoch = len(history["train_loss"])
        metrics, matrix, labels, predictions, probabilities = (
            evaluate_validation(model, val_loader)
        )
        result = {
            "profile": profile,
            "fold": fold,
            "n_train_events": len(train_events),
            "n_val_events": len(val_events),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "train_valid": int(sum(y_train == 1)),
            "train_invalid": int(sum(y_train == 0)),
            "val_valid": int(sum(y_val == 1)),
            "val_invalid": int(sum(y_val == 0)),
            **metrics,
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
        }
        results.append(result)

        for values in zip(
            ids_val, groups_val, views_val, labels, predictions, probabilities
        ):
            video_id, event, view, true, prediction, probability = values
            prediction_rows.append({
                "profile": profile,
                "fold": fold,
                "physical_event": event,
                "video_id": video_id,
                "source_view": view,
                "true_label": int(true),
                "probability": float(probability),
                "prediction": int(prediction),
            })

        np.savez(fold_model_dir / "history.npz", **history)
        np.save(fold_results_dir / "confusion_matrix.npy", matrix)
        with open(fold_model_dir / "scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)
        plot_training_curves(history, best_epoch, fold_results_dir)
        print(
            f"  METRICS: Acc={metrics['accuracy']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"AUC={metrics['roc_auc']:.4f} "
            f"Specificity={metrics['specificity']:.4f} "
            f"TN={metrics['tn']} FP={metrics['fp']} "
            f"FN={metrics['fn']} TP={metrics['tp']} "
            f"best={best_epoch} final={final_epoch}"
        )

    summary = summarize_results(results)
    pd.DataFrame(results).to_csv(results_dir / "cv_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        results_dir / "cv_predictions.csv", index=False
    )
    (results_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    config_data = run_config(profile)
    config_text = json.dumps(config_data, indent=2)
    (results_dir / "run_config.json").write_text(config_text, encoding="utf-8")
    (model_dir / "run_config.json").write_text(config_text, encoding="utf-8")

    print(f"\nRESUMEN {profile}")
    for metric in METRIC_NAMES:
        values = summary[metric]
        print(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print("  best epochs:", summary["best_epoch"])
    print("  final epochs:", summary["final_epoch"])


def paired_cv():
    folds = load_fold_map()
    expected_fold = dict(zip(folds["physical_event"], folds["fold_id"]))
    predictions = {}

    for profile in ("dl_left", "dl_right"):
        _, results_dir = profile_paths(get_config(profile))
        path = results_dir / "cv_predictions.csv"
        predictions[profile] = pd.read_csv(path, dtype={"physical_event": str})

    paired_results = []
    paired_rows = []

    for fold in range(N_FOLDS):
        left = predictions["dl_left"]
        right = predictions["dl_right"]
        left = left[left["fold"] == fold].sort_values("physical_event")
        right = right[right["fold"] == fold].sort_values("physical_event")
        paired_events = sorted(
            set(left["physical_event"]) & set(right["physical_event"])
        )
        left = left[left["physical_event"].isin(paired_events)].sort_values(
            "physical_event"
        )
        right = right[right["physical_event"].isin(paired_events)].sort_values(
            "physical_event"
        )
        left_events = left["physical_event"].to_numpy()
        right_events = right["physical_event"].to_numpy()

        assert np.array_equal(left_events, right_events)
        assert len(left_events) == len(np.unique(left_events))
        assert all(expected_fold[event] == fold for event in left_events)

        for profile, frame in (("dl_left", left), ("dl_right", right)):
            metrics, _ = metrics_with_specificity(
                frame["true_label"].to_numpy(),
                frame["prediction"].to_numpy(),
                frame["probability"].to_numpy(),
            )
            paired_results.append({
                "profile": profile,
                "fold": fold,
                "n_val": len(frame),
                **metrics,
            })

        for left_row, right_row in zip(
            left.to_dict("records"), right.to_dict("records")
        ):
            paired_rows.append({
                "fold": fold,
                "physical_event": left_row["physical_event"],
                "left_video": left_row["video_id"],
                "right_video": right_row["video_id"],
                "left_true": left_row["true_label"],
                "right_true": right_row["true_label"],
                "left_probability": left_row["probability"],
                "right_probability": right_row["probability"],
                "left_prediction": left_row["prediction"],
                "right_prediction": right_row["prediction"],
            })
        print(f"Fold {fold}: LEFT paired events == RIGHT paired events (N={len(left)})")

    COMMON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(paired_results)
    results_frame.to_csv(
        COMMON_RESULTS_DIR / "paired_cv_results.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        COMMON_RESULTS_DIR / "paired_cv_predictions.csv", index=False
    )
    summary = {}

    for profile in ("dl_left", "dl_right"):
        subset = results_frame[results_frame["profile"] == profile]
        summary[profile] = {}

        for metric in METRIC_NAMES:
            values = subset[metric].to_numpy(dtype=float)
            summary[profile][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }

    (COMMON_RESULTS_DIR / "paired_cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    for profile in ("dl_left", "dl_right"):
        values = summary[profile]
        print(
            f"{profile} PAIRED: Accuracy={values['accuracy']['mean']:.4f} +/- "
            f"{values['accuracy']['std']:.4f}, "
            f"F1={values['f1']['mean']:.4f} +/- {values['f1']['std']:.4f}, "
            f"AUC={values['roc_auc']['mean']:.4f} +/- "
            f"{values['roc_auc']['std']:.4f}"
        )


def fmt(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def aggregate_summary():
    rows = []
    report_lines = [
        "# Baseline V2 - 5-Fold Cross-Validation",
        "",
        "CV usa solamente development (train+validation). Test no fue evaluado.",
        "",
        "| Perfil | Accuracy CV | F1 CV | AUC CV |",
        "|---|---:|---:|---:|",
    ]

    for profile in PROFILES:
        config = get_config(profile)
        _, results_dir = profile_paths(config)
        summary = json.loads((results_dir / "cv_summary.json").read_text())
        test_metrics = np.load(
            config["results_dir"] / "baseline_v2" / "metrics.npz"
        )
        row = {
            "profile": profile,
            "cv": summary,
            "test_baseline_v2": {
                "accuracy": float(test_metrics["test_accuracy"]),
                "f1": float(test_metrics["test_f1"]),
                "roc_auc": float(test_metrics["test_roc_auc"]),
            },
            "historical_cv_pre_baseline_v2": HISTORICAL_CV[profile],
        }
        rows.append(row)
        report_lines.append(
            f"| {profile} | {fmt(summary['accuracy']['mean'], summary['accuracy']['std'])} | "
            f"{fmt(summary['f1']['mean'], summary['f1']['std'])} | "
            f"{fmt(summary['roc_auc']['mean'], summary['roc_auc']['std'])} |"
        )

    report_lines.extend([
        "",
        "## Test Baseline V2 guardado vs CV",
        "",
        "| Perfil | Test Accuracy/F1/AUC | CV Accuracy/F1/AUC |",
        "|---|---|---|",
    ])

    for row in rows:
        test = row["test_baseline_v2"]
        cv = row["cv"]
        report_lines.append(
            f"| {row['profile']} | {test['accuracy']:.4f} / {test['f1']:.4f} / "
            f"{test['roc_auc']:.4f} | "
            f"{fmt(cv['accuracy']['mean'], cv['accuracy']['std'])} / "
            f"{fmt(cv['f1']['mean'], cv['f1']['std'])} / "
            f"{fmt(cv['roc_auc']['mean'], cv['roc_auc']['std'])} |"
        )

    report_lines.extend([
        "",
        "Los valores historicos se marcan PRE-BASELINE-V2; cambiaron mapping, folds y orden scaler/augmentation.",
    ])
    COMMON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (COMMON_RESULTS_DIR / "cv_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (COMMON_RESULTS_DIR / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(report_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline V2 development-only CV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--profile", choices=PROFILES)
    group.add_argument("--paired", action="store_true")
    group.add_argument("--summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.prepare:
        prepare_common_folds()
    elif args.profile:
        train_profile(args.profile)
    elif args.paired:
        paired_cv()
    else:
        aggregate_summary()
