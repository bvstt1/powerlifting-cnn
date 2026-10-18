"""Auditoria reproducible de runs SQ; congela splits y completa left v2.

python -m training.sq.run_analysis --stage freeze
python -m training.sq.run_analysis --stage left
"""
import argparse
import csv
import hashlib
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from training.common.train_utils import PoseCNN
from training.sq import train_posecnn as sq


BASE = sq.ROOT / "training"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference(view):
    return "baseline_t170_k10" if view == "front" else "baseline_t170_v2"


def freeze():
    for view in ("front", "left"):
        p = sq.paths(view, reference(view))
        dest = p["models"] / "provenance"
        dest.mkdir(parents=True, exist_ok=True)
        source_split = p["data"] / "split.npz"
        source_csv = p["data"] / "split.csv"
        with np.load(source_split) as split:
            membership = {role: split[f"ids_{role}"].astype(str).tolist()
                          for role in ("train", "validation", "test")}
            assert len(set(sum(membership.values(), []))) == sum(map(len, membership.values()))
        manifest = {"view": view, "run": reference(view), "split_ids": membership,
                    "sha256": {}, "folds": {}}
        for name, path in (("split.npz", source_split), ("split.csv", source_csv)):
            target = dest / name
            if target.exists() and digest(target) != digest(path):
                raise RuntimeError(f"Snapshot diferente: {target}")
            if not target.exists():
                shutil.copy2(path, target)
            manifest["sha256"][name] = digest(target)
        development = set(membership["train"] + membership["validation"])
        for fold in range(1, 6):
            src = p["models"] / "cv" / f"fold_{fold}" / "membership.csv"
            target = dest / f"fold_{fold}_membership.csv"
            if target.exists() and digest(target) != digest(src):
                raise RuntimeError(f"Snapshot diferente: {target}")
            if not target.exists():
                shutil.copy2(src, target)
            with target.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert {r["video_id"] for r in rows} == development
            assert not ({r["video_id"] for r in rows} & set(membership["test"]))
            manifest["folds"][str(fold)] = {"sha256": digest(target),
                                            "roles": dict(Counter(r["role"] for r in rows))}
        sq.save_json(dest / "manifest.json", manifest)
        print(view, "snapshot", {k: len(v) for k, v in membership.items()}, dest)


def metrics(y, prob, threshold=0.5):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {"n": len(y), "n_invalid": int(sum(y == 0)), "n_valid": int(sum(y == 1)),
            "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_valid": precision_score(y, pred, pos_label=1, zero_division=0),
            "recall_valid": recall_score(y, pred, pos_label=1, zero_division=0),
            "f1_valid": f1_score(y, pred, pos_label=1, zero_division=0),
            "precision_invalid": precision_score(y, pred, pos_label=0, zero_division=0),
            "recall_invalid": recall_score(y, pred, pos_label=0, zero_division=0),
            "f1_invalid": f1_score(y, pred, pos_label=0, zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "specificity": tn / (tn + fp), "npv": tn / (tn + fn) if tn + fn else 0.,
            "roc_auc": roc_auc_score(y, prob) if len(set(y)) == 2 else None,
            "matrix": [[tn, fp], [fn, tp]], "threshold": threshold}


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def left():
    p = sq.paths("left", reference("left"))
    snapshot = json.loads((p["models"] / "provenance" / "manifest.json").read_text(encoding="utf-8"))
    data = np.load(p["models"] / "provenance" / "split.npz")
    conf = json.loads((p["models"] / "config.json").read_text(encoding="utf-8"))
    assert conf["input_shape"] == [170, 5, 3] and conf["subset"] == "all"
    with (p["models"] / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    assert scaler.n_samples_seen_ == len(data["X_train"])
    assert scaler.n_features_in_ == 170 * 5 * 3
    model = PoseCNN(T=170, K=5, C=3).to(sq.DEVICE)
    checkpoint = p["models"] / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=sq.DEVICE, weights_only=True))
    assert snapshot["split_ids"]["validation"] == data["ids_validation"].astype(str).tolist()
    X_val = sq.transform(scaler, data["X_validation"].astype(np.float32))
    m_val, pred_val, probs_val = sq.evaluate(model, X_val, data["y_validation"], sq.SEED)
    assert m_val["confusion_matrix"] == [[12, 7], [1, 36]], m_val
    validation = [{"video_id": str(ident), "true_label": int(y), "predicted_label": int(pred),
                   "probability_valid": float(prob)} for ident, y, pred, prob in
                  zip(data["ids_validation"], data["y_validation"], pred_val, probs_val)]
    target = p["results"] / "validation_predictions_recovered.csv"
    sq.save_csv(target, validation, list(validation[0]))
    test_rows = read_predictions(p["results"] / "test_predictions.csv")
    assert [r["video_id"] for r in test_rows] == data["ids_test"].astype(str).tolist()
    test = metrics([r["true_label"] for r in test_rows], [r["probability_valid"] for r in test_rows])
    val = metrics([r["true_label"] for r in validation], [r["probability_valid"] for r in validation])
    folds = []
    cv_rows = read_predictions(p["results"] / "cv_predictions.csv")
    assert len(cv_rows) == len(data["X_train"]) + len(data["X_validation"])
    for fold in range(1, 6):
        rows = [r for r in cv_rows if int(r["fold"]) == fold]
        ids = {r["video_id"] for r in rows}
        with (p["models"] / "provenance" / f"fold_{fold}_membership.csv").open(encoding="utf-8", newline="") as stream:
            assert ids == {r["video_id"] for r in csv.DictReader(stream) if r["role"] == "outer_validation"}
        folds.append({"fold": fold, **metrics([r["true_label"] for r in rows],
                                                 [r["probability_valid"] for r in rows])})
    keys = ("accuracy", "balanced_accuracy", "precision_valid", "recall_valid", "f1_valid",
            "precision_invalid", "recall_invalid", "f1_invalid", "f1_macro", "specificity", "npv", "roc_auc")
    aggregate = {key: {"mean": float(np.mean([r[key] for r in folds])),
                       "std": float(np.std([r[key] for r in folds]))} for key in keys}
    labels = {r["video_id"]: r for r in sq.load_labels(p["labels"])}
    faults = {}
    for code in sorted({c for r in test_rows for c in labels[r["video_id"]]["codigos"].split("|") if c}):
        rows = [r for r in test_rows if code in labels[r["video_id"]]["codigos"].split("|")]
        faults[code] = {"n": len(rows), "correct": sum(r["true_label"] == r["predicted_label"] for r in rows),
                        "errors": [r["video_id"] for r in rows if r["true_label"] != r["predicted_label"]]}
    history = read_predictions(p["models"] / "history.csv")
    report = {"run": reference("left"), "checkpoint_sha256": digest(checkpoint),
              "scaler_sha256": digest(p["models"] / "scaler.pkl"),
              "split_sha256": snapshot["sha256"]["split.npz"],
              "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
              "validation": val, "test": test, "cv_folds": folds, "cv_summary": aggregate,
              "fault_codes_nonexclusive": faults, "test_errors": [r for r in test_rows if r["true_label"] != r["predicted_label"]],
              "history": {"n_epochs": len(history), "best_epoch": int(min(history, key=lambda r: float(r["val_loss"]))["epoch"]),
                          "first_train_loss": float(history[0]["train_loss"]), "last_train_loss": float(history[-1]["train_loss"]),
                          "first_val_loss": float(history[0]["val_loss"]), "last_val_loss": float(history[-1]["val_loss"])}}
    sq.save_json(p["results"] / "complete_metrics_recovered.json", report)
    print("left v2 verificado, sin reentrenar:", report["test"], "folds", len(folds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("freeze", "left"))
    args = parser.parse_args()
    {"freeze": freeze, "left": left}[args.stage]()
