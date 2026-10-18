"""Tanda controlada SQ front K10: CV sin test, evaluación final explícita.

python -m training.sq.controlled_front --variant A --stage prepare
python -m training.sq.controlled_front --variant A --stage cv
... --stage fit-final (no consulta test)
... --stage evaluate-test (solo configuración seleccionada)
"""
import argparse
import csv
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from training.common.train_utils import PoseCNN, augment_dataset
from training.sq import train_posecnn as sq
from training.sq.run_analysis import digest, metrics, read_predictions


SETUPS = {
    "A": {"frames": 170, "scale": "shoulders", "scaler": "temporal", "patience": 20},
    "B": {"frames": 96, "scale": "shoulders", "scaler": "temporal", "patience": 8},
    "C": {"frames": 170, "scale": "shanks", "scaler": "temporal", "patience": 8},
    "D": {"frames": 170, "scale": "shoulders", "scaler": "point_xy", "patience": 8},
    # Solo ejecutar si CV A–D justifica una combinación.
    "BC": {"frames": 96, "scale": "shanks", "scaler": "temporal", "patience": 8},
    "CD": {"frames": 170, "scale": "shanks", "scaler": "point_xy", "patience": 8},
}
REFERENCE = sq.paths("front", "baseline_t170_k10")
ROOT_DATA = sq.ROOT / "training" / "data" / "sq" / "front" / "posecnn" / "runs"
SELECTED = sq.ROOT / "training" / "results" / "sq" / "front" / "posecnn" / "controlled_selection.json"
IDX = sq.VIEW_CONFIGS["front"]["subsets"]["legs"]


def paths(variant):
    run_id = "controlled_" + variant.lower()
    out = sq.paths("front", run_id)
    out["data"] = ROOT_DATA / run_id
    return out


def reference():
    prov = REFERENCE["models"] / "provenance"
    manifest = json.loads((prov / "manifest.json").read_text(encoding="utf-8"))
    if digest(prov / "split.npz") != manifest["sha256"]["split.npz"]:
        raise ValueError("El snapshot de referencia cambió")
    return manifest, prov


def make_sequences(ids, setup):
    result = []
    for ident in ids:
        raw = np.load(REFERENCE["source"] / f"{ident}.npy", allow_pickle=False)
        data, _, _ = sq.preprocess_front(raw, target_frames=setup["frames"], scale_mode=setup["scale"])
        result.append(data[:, list(IDX), :])
    return np.asarray(result, dtype=np.float32)


def prepare(variant):
    setup = SETUPS[variant]
    p = paths(variant)
    manifest, prov = reference()
    p["data"].mkdir(parents=True, exist_ok=True)
    target = p["data"] / "development.npz"
    if target.exists():
        raise FileExistsError(target)
    with np.load(prov / "split.npz") as split:
        ids_train = split["ids_train"].astype(str)
        ids_val = split["ids_validation"].astype(str)
        y_train = split["y_train"]
        y_val = split["y_validation"]
        # Nunca leer split['X_test']/split['y_test'] en preparación/CV.
        reference_x = np.concatenate([split["X_train"], split["X_validation"]])
    ids = np.concatenate([ids_train, ids_val])
    assert set(ids) == set(manifest["split_ids"]["train"] + manifest["split_ids"]["validation"])
    if setup["frames"] == 170 and setup["scale"] == "shoulders":
        X = sq.apply_subset(reference_x, IDX)
    else:
        X = make_sequences(ids, setup)
    assert X.shape == (377, setup["frames"], 10, 3) and np.isfinite(X).all()
    y = np.concatenate([y_train, y_val]).astype(int)
    np.savez_compressed(target, X=X, ids=ids, y=y, n_train=len(ids_train))
    model = PoseCNN(T=setup["frames"], K=10, C=3)
    effective = {
        "variant": variant, "changes_from_K10": setup, "baseline_run": "baseline_t170_k10",
        "split_snapshot_sha256": manifest["sha256"]["split.npz"],
        "ids_train": manifest["split_ids"]["train"], "ids_validation": manifest["split_ids"]["validation"],
        "test_ids_only": manifest["split_ids"]["test"], "positive_class": "1=valido",
        "architecture": "training.common.train_utils.PoseCNN", "trainable_parameters": sum(q.numel() for q in model.parameters() if q.requires_grad),
        "seed": 42, "fold_seeds": list(range(43, 48)), "threshold": 0.5,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "scaler": "train original only", "augmentation": "train x4 before transform; xy only, no flip/rotation",
        "model": {"optimizer": "Adam", "lr": 0.001, "weight_decay": 1e-4, "batch": 16,
                  "epochs_max": 200, "scheduler_patience": 4, "early_patience": setup["patience"]}}
    sq.save_json(p["data"] / "config.json", effective)
    p["models"].mkdir(parents=True, exist_ok=True)
    sq.save_json(p["models"] / "config.json", effective)
    print(variant, "desarrollo aislado", X.shape, "test no leído", target)


class PointXYScaler:
    """Media/std de x/y por articulación, compartida entre todos los frames.

    Confianza no se estandariza. Estadísticas solo del train original.
    """
    def fit(self, X):
        self.n_samples_seen_ = len(X)
        xy = np.asarray(X, dtype=float)[..., :2]
        self.mean_ = xy.mean(axis=(0, 1))
        self.scale_ = xy.std(axis=(0, 1))
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        out = np.asarray(X, dtype=np.float32).copy()
        out[..., :2] = (out[..., :2] - self.mean_) / self.scale_
        return out


def scale(variant, X_fit, y_fit, X_val, seed, path):
    setup = SETUPS[variant]
    if setup["scaler"] == "point_xy":
        scaler = PointXYScaler().fit(X_fit)
    else:
        scaler = StandardScaler().fit(X_fit.reshape(len(X_fit), -1))
    assert scaler.n_samples_seen_ == len(X_fit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(scaler, stream)
    aug, labels = augment_dataset(X_fit, y_fit, np.random.RandomState(seed), 3,
                                  transforms=("translation", "scaling", "noise", "temporal_shift"))
    assert np.array_equal(aug[::4, :, :, 2], X_fit[..., 2])
    return transform(scaler, aug), labels, transform(scaler, X_val), scaler


def transform(scaler, X):
    if isinstance(scaler, PointXYScaler):
        return scaler.transform(X)
    return scaler.transform(X.reshape(len(X), -1)).reshape(X.shape).astype(np.float32)


def fold_indices(ids, y, prov):
    for fold, (outer_fit, outer_val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y), 1):
        fit, early = train_test_split(outer_fit, test_size=0.15, stratify=y[outer_fit], random_state=42+fold)
        with (prov / f"fold_{fold}_membership.csv").open(encoding="utf-8", newline="") as f:
            frozen = {(r["video_id"], r["role"]) for r in csv.DictReader(f)}
        actual = {(str(ids[i]), role) for role, indices in (("fit",fit),("early",early),("outer_validation",outer_val)) for i in indices}
        if actual != frozen:
            raise ValueError(f"Fold {fold} cambió respecto a referencia")
        yield fold, fit, early, outer_val


def cv(variant):
    setup, p = SETUPS[variant], paths(variant)
    if (p["results"] / "cv_metrics.csv").exists():
        raise FileExistsError("CV ya existe, no sobrescribir")
    dev = np.load(p["data"] / "development.npz")
    X, y, ids = dev["X"], dev["y"], dev["ids"].astype(str)
    _, prov = reference()
    p["results"].mkdir(parents=True, exist_ok=True)
    scores, predictions = [], []
    for fold, fit, early, outer in fold_indices(ids, y, prov):
        folder = p["models"] / "cv" / f"fold_{fold}"
        X_fit, y_fit, X_early, scaler = scale(variant, X[fit], y[fit], X[early], 42+fold, folder / "scaler.pkl")
        model, epoch, _ = sq.fit(X_fit, y_fit, X_early, y[early], 42+fold, folder, patience=setup["patience"])
        _, pred, probs = sq.evaluate(model, transform(scaler, X[outer]), y[outer], 42+fold)
        score = {"fold": fold, "best_epoch": epoch, **metrics(y[outer], probs)}
        scores.append(score)
        predictions.extend({"fold": fold, "video_id": str(ids[i]), "true_label": int(y[i]),
                            "probability_valid": float(probs[j]), "predicted_label": int(pred[j])}
                           for j, i in enumerate(outer))
        print(variant, "fold", fold, "balanced", round(score["balanced_accuracy"],4),
              "macro", round(score["f1_macro"],4), "AUC", round(score["roc_auc"],4),
              "invalid recall", round(score["recall_invalid"],4))
    assert {r["video_id"] for r in predictions} == set(ids)
    sq.save_csv(p["results"] / "cv_metrics.csv", scores, list(scores[0]))
    sq.save_csv(p["results"] / "cv_predictions.csv", predictions, list(predictions[0]))
    keys = ("balanced_accuracy", "f1_macro", "roc_auc", "recall_invalid", "precision_invalid", "f1_valid")
    summary = {k: {"mean": float(np.mean([r[k] for r in scores])),
                   "std": float(np.std([r[k] for r in scores]))} for k in keys}
    sq.save_json(p["results"] / "cv_summary.json", summary)
    print(variant, "CV", summary)


def fit_final(variant):
    setup, p = SETUPS[variant], paths(variant)
    if (p["models"] / "best_model.pth").exists():
        raise FileExistsError("Checkpoint final ya existe")
    dev = np.load(p["data"] / "development.npz")
    n = int(dev["n_train"])
    X, y = dev["X"], dev["y"]
    X_fit, y_fit, X_val, scaler = scale(variant, X[:n], y[:n], X[n:], 42, p["models"] / "scaler.pkl")
    model, epoch, _ = sq.fit(X_fit, y_fit, X_val, y[n:], 42, p["models"], patience=setup["patience"])
    _, pred, probs = sq.evaluate(model, X_val, y[n:], 42)
    rows = [{"video_id": str(i), "true_label": int(label), "probability_valid": float(prob), "predicted_label": int(pr)}
            for i, label, pr, prob in zip(dev["ids"][n:], y[n:], pred, probs)]
    p["results"].mkdir(parents=True, exist_ok=True)
    sq.save_csv(p["results"] / "validation_predictions.csv", rows, list(rows[0]))
    sq.save_json(p["results"] / "validation_metrics.json", {"best_epoch": epoch, **metrics(y[n:], probs)})
    print(variant, "final fit solo desarrollo", epoch, "validación", metrics(y[n:], probs))


def evaluate_test(variant):
    if not SELECTED.exists():
        raise RuntimeError("Falta selección congelada de desarrollo")
    choice = json.loads(SELECTED.read_text(encoding="utf-8"))
    if choice["selected_variant"] != variant:
        raise RuntimeError("Solo el candidato seleccionado puede evaluar test")
    p, setup = paths(variant), SETUPS[variant]
    if (p["results"] / "test_metrics.json").exists():
        raise FileExistsError("Test ya evaluado")
    manifest, prov = reference()
    assert choice["reference_split_sha256"] == manifest["sha256"]["split.npz"]
    with np.load(prov / "split.npz") as split:
        test_ids = split["ids_test"].astype(str)
        y = split["y_test"].astype(int)
    X = make_sequences(test_ids, setup)
    with (p["models"] / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    model = PoseCNN(T=setup["frames"], K=10, C=3).to(sq.DEVICE)
    model.load_state_dict(torch.load(p["models"] / "best_model.pth", map_location=sq.DEVICE, weights_only=True))
    _, preds, probs = sq.evaluate(model, transform(scaler, X), y, 42)
    sq.save_csv(p["results"] / "test_predictions.csv",
                [{"video_id": str(i), "true_label": int(label), "probability_valid": float(prob), "predicted_label": int(pred)}
                 for i, label, pred, prob in zip(test_ids, y, preds, probs)],
                ["video_id", "true_label", "probability_valid", "predicted_label"])
    sq.save_json(p["results"] / "test_metrics.json", metrics(y, probs))
    print("Test exploratorio por consulta histórica previa:", variant, metrics(y, probs))


def verify(variant):
    setup, p = SETUPS[variant], paths(variant)
    manifest, prov = reference()
    with np.load(p["data"] / "development.npz") as dev:
        X, y, ids = dev["X"], dev["y"], dev["ids"].astype(str)
        assert X.shape == (377, setup["frames"], 10, 3)
        assert np.isfinite(X).all()
        assert ids.tolist() == manifest["split_ids"]["train"] + manifest["split_ids"]["validation"]
        assert not set(ids) & set(manifest["split_ids"]["test"])
        folds = list(fold_indices(ids, y, prov))
        assert len(folds) == 5
    data_conf = p["data"] / "config.json"
    with data_conf.open(encoding="utf-8") as f:
        conf = json.load(f)
    assert conf["trainable_parameters"] == 102017
    assert conf["split_snapshot_sha256"] == manifest["sha256"]["split.npz"]
    model_conf = p["models"] / "config.json"
    if not model_conf.exists():
        sq.save_json(model_conf, conf)
    assert digest(model_conf) == digest(data_conf)
    with (p["models"] / "cv" / "fold_1" / "scaler.pkl").open("rb") as stream:
        scaler = pickle.load(stream)
    assert scaler.n_samples_seen_ == len(folds[0][1])
    if variant == "D":
        assert isinstance(scaler, PointXYScaler)
    else:
        assert isinstance(scaler, StandardScaler)
    print(variant, "verificado: shape, clases, split/folds, scaler train original, config, parámetros")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(SETUPS))
    parser.add_argument("--stage", required=True, choices=("prepare", "cv", "fit-final", "evaluate-test", "verify"))
    args = parser.parse_args()
    {"prepare":prepare, "cv":cv, "fit-final":fit_final, "evaluate-test":evaluate_test,
     "verify":verify}[args.stage](args.variant)
