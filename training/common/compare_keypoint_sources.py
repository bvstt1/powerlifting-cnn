"""Compara keypoints YOLO vs MediaPipe con la receta PoseCNN final de cada vista.

Solo cambia la carpeta de entrada keypoints/<ejercicio>/<fuente>/<vista>. Los
extractores MediaPipe guardan los mismos puntos, en el mismo orden y forma
(T, K, 3) que YOLO; visibility MediaPipe ocupa el canal de confianza YOLO (no
son equivalentes, pero el canal recibe el mismo tratamiento).

Recetas reutilizadas sin cambiar hiperparametros:
- bp_front: bp_front_02_posecnn_common_v1 (+ CV de bp_front_04): BPFrontPoseCNN,
  mascara de ausencia, aumento con rotacion, StandardScaler, patience 15,
  scheduler 5, semillas finales 42/1337 (minima validation loss).
- bp_left / bp_right: training.bp.train_posecnn: PoseCNN K8, StandardScaler,
  aumento sin rotacion, patience 8, scheduler 4.
- sq_front: training.sq.controlled_front variante D: K10 piernas, PointXYScaler.
- sq_left: training.sq.train_posecnn baseline_t170_v2: K5, StandardScaler.
- sq_right: training.sq.controlled_right point_xy_v1: K5, PointXYScaler.

Particion: un unico split por vista, por video_id, compartido por ambas fuentes.
bp_front usa el split comun congelado (training/data/bp/front/posecnn/
bp_front_pose_split.csv); el resto reproduce prepare() de su receta sobre los
videos YOLO utilizables (70/15/15 estratificado, semillas 42/43). La cohorte
comparada es ese split restringido a videos utilizables en AMBAS fuentes, con
los mismos filtros de calidad. Scaler ajustado solo con train original del fold;
aumento solo en train. Test se evalua una vez por fuente tras la CV.

Uso (desde la raiz del repo):
    python -m training.common.compare_keypoint_sources --target bp_left --stage prepare
    python -m training.common.compare_keypoint_sources --target bp_left --stage cv --source yolo
    python -m training.common.compare_keypoint_sources --target bp_left --stage final --source mediapipe
    python -m training.common.compare_keypoint_sources --target bp_left --stage compare
    python -m training.common.compare_keypoint_sources --target all --stage all
    python -m training.common.compare_keypoint_sources --stage report
"""

import argparse
import csv
import hashlib
import json
import pickle
import platform
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy.stats import binomtest, rankdata
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from training.bp import train_posecnn as bp
from training.bp.experiments.bp_front_02_posecnn_common_v1 import data as front_data
from training.bp.experiments.bp_front_02_posecnn_common_v1 import train as front_train
from training.bp.experiments.bp_front_02_posecnn_common_v1.model import BPFrontPoseCNN
from training.common.train_utils import augment_dataset
from training.sq import train_posecnn as sq
from training.sq.controlled_front import PointXYScaler, transform as scale
from training.sq.run_analysis import metrics as base_metrics


ROOT = Path(__file__).resolve().parents[2]
RUN = "source_cmp_v1"
SOURCES = ("yolo", "mediapipe")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUG_FACTOR = 3
LATERAL_TRANSFORMS = ("translation", "scaling", "noise", "temporal_shift")
FRONT_SPLIT = ROOT / "training" / "data" / "bp" / "front" / "posecnn" / "bp_front_pose_split.csv"
FRONT_SEEDS = (42, 1337)
FRONT_PATIENCE = 15
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261020
GLOBAL_REPORT = ROOT / "training" / "results" / "keypoint_source_comparison_v1"
DELTA_KEYS = ("balanced_accuracy", "f1_macro", "roc_auc", "recall_invalid", "recall_valid", "accuracy")
SUMMARY_KEYS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_valid", "f1_invalid", "precision_valid",
                "recall_valid", "precision_invalid", "recall_invalid", "roc_auc", "average_precision")

TARGETS = {
    "bp_front": {"exercise": "bp", "view": "front", "recipe": "bp_front", "keypoints": 6,
                 "reference": "bp_front_02_posecnn_common_v1 + CV bp_front_04_cross_validation_5fold_v1"},
    "bp_left": {"exercise": "bp", "view": "left", "recipe": "posecnn", "keypoints": 8,
                "scaler": "temporal", "subset": None, "patience": bp.PATIENCE,
                "reference": "training.bp.train_posecnn (bp_left_02_posecnn_compatibility_v1)"},
    "bp_right": {"exercise": "bp", "view": "right", "recipe": "posecnn", "keypoints": 8,
                 "scaler": "temporal", "subset": None, "patience": bp.PATIENCE,
                 "reference": "training.bp.train_posecnn (bp_right_02_posecnn_common_v1)"},
    "sq_front": {"exercise": "sq", "view": "front", "recipe": "posecnn", "keypoints": 16,
                 "scaler": "point_xy", "subset": sq.VIEW_CONFIGS["front"]["subsets"]["legs"],
                 "patience": sq.PATIENCE, "reference": "training.sq.controlled_front variante D"},
    "sq_left": {"exercise": "sq", "view": "left", "recipe": "posecnn", "keypoints": 5,
                "scaler": "temporal", "subset": None, "patience": sq.PATIENCE,
                "reference": "training.sq.train_posecnn baseline_t170_v2"},
    "sq_right": {"exercise": "sq", "view": "right", "recipe": "posecnn", "keypoints": 5,
                 "scaler": "point_xy", "subset": None, "patience": sq.PATIENCE,
                 "reference": "training.sq.controlled_right point_xy_v1"},
}


# ─── Rutas y E/S ────────────────────────────────────────────────────────────

def dirs(target, source=None):
    cfg = TARGETS[target]
    base = ROOT / "training"
    rel = Path(cfg["exercise"]) / cfg["view"] / "posecnn"
    out = {"data": base / "data" / rel / RUN, "compare": base / "results" / rel / RUN,
           "labels": ROOT / "etiquetado" / cfg["exercise"] / f"{target}.csv"}
    if source is not None:
        out["keypoints"] = ROOT / "keypoints" / cfg["exercise"] / source / cfg["view"]
        out["models"] = base / "models" / rel / "runs" / f"{RUN}_{source}"
        out["results"] = base / "results" / rel / "runs" / f"{RUN}_{source}"
    return out


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=float),
                    encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard(path):
    if path.exists():
        raise FileExistsError(f"Ejecucion existente, no se sobrescribe: {path}")


def metrics(y, probs):
    result = base_metrics(y, probs)
    result["average_precision"] = float(average_precision_score(y, probs))
    return result


# ─── Auditoria y preprocesamiento por receta ────────────────────────────────

def read_labels(target):
    labels, blocking = {}, []
    for row in sq.load_labels(dirs(target)["labels"]):
        ident, value = row["video_id"].strip(), row["label"].strip()
        if not value:
            continue
        if value not in ("0", "1"):
            blocking.append(ident)
        elif row["resultado"].strip().lower() != {"0": "invalido", "1": "valido"}[value]:
            blocking.append(ident)
        labels[ident] = int(value) if value in ("0", "1") else None
    if blocking:
        raise ValueError(f"Etiquetas invalidas o contradictorias: {blocking[:10]}")
    return labels


def inspect_front(path):
    row = {"video_id": path.stem, "status": "excluded", "reasons": "", "frames": 0,
           "missing_xy_ratio": None, "max_gap": 0, "median_confidence": None, "sha256": ""}
    data = np.load(path, allow_pickle=False)
    row["frames"] = len(data)
    if data.ndim != 3 or data.shape[1:] != (6, 3):
        row["reasons"] = "shape_or_dtype"
        return row
    missing = ~np.isfinite(data[..., :2]).all(axis=2)
    row["missing_xy_ratio"] = float(missing.mean()) if len(data) else 1.0
    row["max_gap"] = max((sq.longest_run(missing[:, i]) for i in range(6)), default=0)
    conf = data[..., 2][np.isfinite(data[..., 2])]
    row["median_confidence"] = float(np.median(conf)) if conf.size else None
    row["status"] = "usable"
    return row


def inspect(cfg, path):
    if cfg["recipe"] == "bp_front":
        return inspect_front(path)
    if cfg["exercise"] == "bp":
        return bp.inspect_file(path)
    return sq.inspect_file(path, cfg["keypoints"], cfg["view"])


def prepare_sequence(cfg, raw):
    """Devuelve (secuencia (170,K,3), mascara|None) con la receta de la vista."""
    if cfg["recipe"] == "bp_front":
        sequence, mask, _, _ = front_data.secuencia_a_170(raw)
        return sequence, mask
    if cfg["exercise"] == "bp":
        return bp.preprocess(raw)[0], None
    if cfg["view"] == "front":
        sequence = sq.preprocess_front(raw)[0]
        return sequence[:, list(cfg["subset"]), :], None
    return sq.preprocess_left(raw)[0], None


def audit_source(target, source, labels):
    cfg, d = TARGETS[target], dirs(target, source)
    files = sorted(d["keypoints"].glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"Sin keypoints: {d['keypoints']}")
    manifest, sequences, hashes = [], {}, {}
    for path in files:
        row = inspect(cfg, path)
        reasons = [r for r in row["reasons"].split("|") if r]
        digest = row.get("sha256", "")
        # Misma regla que audit(): duplicado exacto de pose excluye la copia.
        if cfg["recipe"] != "bp_front" and digest:
            if digest in hashes:
                row["status"] = "excluded"
                reasons.append(f"duplicate_of_{hashes[digest]}")
            else:
                hashes[digest] = row["video_id"]
        if row["status"] == "usable" and labels.get(row["video_id"]) is None:
            row["status"] = "excluded"
            reasons.append("no_label")
        if row["status"] == "usable":
            try:
                sequences[row["video_id"]] = prepare_sequence(cfg, np.load(path, allow_pickle=False))
            except ValueError as error:
                row["status"] = "excluded"
                reasons.append(str(error))
        row["reasons"] = "|".join(reasons)
        manifest.append(row)
    return manifest, sequences


def base_split(target, yolo_manifest, labels):
    """Split por video_id definido por la receta original (YOLO)."""
    if target == "bp_front":
        rows = [r for r in read_csv(FRONT_SPLIT) if r["common_member"] == "True"]
        roles = {r["video_id"]: r["common_split"] for r in rows}
        counts = Counter(roles.values())
        if [counts[k] for k in ("train", "validation", "test")] != [462, 99, 100]:
            raise ValueError(f"Split comun BP front alterado: {counts}")
        order = sorted(roles)
        return {role: [v for v in order if roles[v] == role] for role in ("train", "validation", "test")}
    # Replica exacta de prepare() en training.bp/sq.train_posecnn.
    ids = [r["video_id"] for r in yolo_manifest if r["status"] == "usable"]
    y = np.asarray([labels[i] for i in ids])
    idx = np.arange(len(y))
    development, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=sq.SEED)
    train, validation = train_test_split(development, test_size=0.15 / 0.85,
                                         stratify=y[development], random_state=sq.SEED + 1)
    return {role: [ids[i] for i in indices]
            for role, indices in (("train", train), ("validation", validation), ("test", test))}


def prepare(target):
    cfg, d = TARGETS[target], dirs(target)
    guard(d["data"] / "summary.json")
    labels = read_labels(target)
    audits = {source: audit_source(target, source, labels) for source in SOURCES}
    split = base_split(target, audits["yolo"][0], labels)
    status = {source: {r["video_id"]: r for r in audits[source][0]} for source in SOURCES}
    cohort_rows, cohort = [], {}
    for role in ("train", "validation", "test"):
        cohort[role] = []
        for ident in split[role]:
            usable = {s: ident in audits[s][1] for s in SOURCES}
            if all(usable.values()):
                cohort[role].append(ident)
            row = {"video_id": ident, "label": labels[ident], "split": role, "in_cohort": all(usable.values())}
            for source in SOURCES:
                info = status[source].get(ident, {"status": "missing_file", "reasons": "missing_file"})
                row[f"{source}_status"] = "usable" if usable[source] else "excluded"
                row[f"{source}_reasons"] = info["reasons"]
            cohort_rows.append(row)
        if {labels[i] for i in cohort[role]} != {0, 1}:
            raise ValueError(f"Particion {role} sin ambas clases")
    d["data"].mkdir(parents=True, exist_ok=True)
    save_csv(d["data"] / "cohort.csv", cohort_rows)
    hashes = {}
    for source in SOURCES:
        fields = ["video_id", "status", "reasons", "frames", "missing_xy_ratio", "max_gap", "median_confidence"]
        save_csv(d["data"] / f"audit_{source}.csv", audits[source][0], fields)
        arrays = {}
        for role in ("train", "validation", "test"):
            items = [audits[source][1][i] for i in cohort[role]]
            arrays[f"X_{role}"] = np.stack([x for x, _ in items]).astype(np.float32)
            if cfg["recipe"] == "bp_front":
                arrays[f"M_{role}"] = np.stack([m for _, m in items])
            arrays[f"y_{role}"] = np.asarray([labels[i] for i in cohort[role]], dtype=np.int64)
            arrays[f"ids_{role}"] = np.asarray(cohort[role])
            if not np.isfinite(arrays[f"X_{role}"]).all():
                raise ValueError(f"No finitos en {source}/{role}")
        np.savez_compressed(d["data"] / f"{source}.npz", **arrays)
        hashes[f"{source}.npz"] = sha256(d["data"] / f"{source}.npz")
    coverage = {}
    for source in SOURCES:
        rows = audits[source][0]
        in_split = {i for role in split.values() for i in role}
        excluded = [r for r in rows if r["video_id"] in in_split and r["video_id"] not in audits[source][1]]
        coverage[source] = {
            "pose_files": len(rows), "usable": len(audits[source][1]),
            "base_split_excluded": len(excluded),
            "base_split_exclusion_reasons": dict(Counter(x for r in excluded for x in r["reasons"].split("|") if x)),
            "median_missing_xy_ratio": float(np.median([r["missing_xy_ratio"] for r in rows
                                                         if r["missing_xy_ratio"] is not None])),
            "median_of_median_confidence": float(np.median([r["median_confidence"] for r in rows
                                                             if r["median_confidence"] is not None])),
        }
    summary = {
        "target": target, "recipe": cfg["recipe"], "reference": cfg["reference"],
        "input_shape": list(audits["yolo"][1][cohort["train"][0]][0].shape),
        "base_split": {k: len(v) for k, v in split.items()},
        "cohort": {k: {"n": len(v), "valid": int(sum(labels[i] for i in v)),
                       "invalid": int(sum(1 - labels[i] for i in v))} for k, v in cohort.items()},
        "coverage": coverage, "labels_sha256": sha256(d["labels"]), "arrays_sha256": hashes,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "training/common/train_utils.py", ROOT / "training/bp/train_posecnn.py",
            ROOT / "training/sq/train_posecnn.py", ROOT / "training/sq/controlled_front.py",
            Path(front_data.__file__), Path(front_train.__file__))},
        "split_policy": ("split comun congelado BP front" if target == "bp_front" else
                         "prepare() original sobre YOLO utilizables: 70/15/15 estratificado, semillas 42/43"),
        "cohort_policy": "split base restringido a videos utilizables en yolo y mediapipe",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "sklearn": sklearn.__version__},
    }
    save_json(d["data"] / "summary.json", summary)
    print(target, "cohorte", summary["cohort"], "cobertura",
          {s: (c["usable"], c["base_split_excluded"]) for s, c in coverage.items()}, flush=True)


# ─── Entrenamiento ──────────────────────────────────────────────────────────

def load_split(target, source):
    with np.load(dirs(target)["data"] / f"{source}.npz") as data:
        return {key: data[key] for key in data.files}


def development(target, source):
    data = load_split(target, source)
    X = np.concatenate([data["X_train"], data["X_validation"]])
    M = (np.concatenate([data["M_train"], data["M_validation"]]) if "M_train" in data else None)
    y = np.concatenate([data["y_train"], data["y_validation"]])
    ids = np.concatenate([data["ids_train"], data["ids_validation"]]).astype(str)
    if target == "bp_front":
        # bp_front_04 ordena development por video_id antes de los folds.
        order = np.argsort(ids, kind="stable")
        X, M, y, ids = X[order], M[order], y[order], ids[order]
    return X, M, y, ids


def folds(target, ids, y):
    outer_split = StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)
    for fold, (outer, evaluation) in enumerate(outer_split, 1):
        if target == "bp_front":
            fit, early = next(StratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=42)
                              .split(outer, y[outer]))
            yield fold, outer[fit], outer[early], evaluation, 42
        else:
            fit, early = train_test_split(outer, test_size=0.15, stratify=y[outer], random_state=42 + fold)
            yield fold, fit, early, evaluation, 42 + fold


def train_posecnn(cfg, X, y, X_early, y_early, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if cfg["scaler"] == "point_xy":
        scaler = PointXYScaler().fit(X)
    else:
        scaler = StandardScaler().fit(X.reshape(len(X), -1))
    assert scaler.n_samples_seen_ == len(X)
    with (folder / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    aug, labels = augment_dataset(X, y, np.random.RandomState(seed), AUG_FACTOR, transforms=LATERAL_TRANSFORMS)
    assert np.array_equal(aug[::AUG_FACTOR + 1, :, :, 2], X[..., 2])
    model, epoch, history = sq.fit(scale(scaler, aug), labels, scale(scaler, X_early), y_early,
                                   seed, folder, patience=cfg["patience"])
    return {"model": model, "scaler": scaler, "seed": seed, "best_epoch": epoch,
            "best_loss": min(r["val_loss"] for r in history), "epochs_run": len(history)}


def train_bp_front(X, M, y, X_early, y_early, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
    front_train.reproducible(seed)
    scaler = StandardScaler().fit(X.reshape(len(X), -1))
    assert scaler.n_samples_seen_ == len(X)
    with (folder / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    aug, aug_y = front_data.augment_train(X, M, y, seed)
    train_loader = front_train.loader(front_data.transformar(aug, scaler), aug_y, shuffle=True, seed=seed)
    early_loader = front_train.loader(front_data.transformar(X_early, scaler), y_early)
    model = BPFrontPoseCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.BCELoss()
    best, best_epoch, waiting, history = float("inf"), 0, 0, []
    for epoch in range(1, 201):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
        val_loss, _ = front_train.evaluar(model, early_loader, DEVICE, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best:
            best, best_epoch, waiting = val_loss, epoch, 0
            torch.save(model.state_dict(), folder / "best_model.pth")
        else:
            waiting += 1
            if waiting >= FRONT_PATIENCE:
                break
    model.load_state_dict(torch.load(folder / "best_model.pth", map_location=DEVICE, weights_only=True))
    save_csv(folder / "history.csv", history)
    return {"model": model, "scaler": scaler, "seed": seed, "best_epoch": best_epoch,
            "best_loss": best, "epochs_run": len(history)}


def train(target, X, M, y, X_early, M_early, y_early, seed, folder):
    cfg = TARGETS[target]
    if cfg["recipe"] == "bp_front":
        return train_bp_front(X, M, y, X_early, y_early, seed, folder)
    return train_posecnn(cfg, X, y, X_early, y_early, seed, folder)


def predict(target, trained, X, y):
    if TARGETS[target]["recipe"] == "bp_front":
        loader = front_train.loader(front_data.transformar(X, trained["scaler"]), y)
        return front_train.evaluar(trained["model"], loader, DEVICE, torch.nn.BCELoss())[1]
    return np.asarray(sq.evaluate(trained["model"], scale(trained["scaler"], X), y, trained["seed"])[2])


def summarize(scores):
    return {key: {"mean": float(np.mean([s[key] for s in scores])), "std": float(np.std([s[key] for s in scores]))}
            for key in SUMMARY_KEYS}


def run_cv(target, source):
    d = dirs(target, source)
    guard(d["results"] / "cv_predictions.csv")
    X, M, y, ids = development(target, source)
    scores, rows = [], []
    start = time.perf_counter()
    for fold, fit, early, outer, seed in folds(target, ids, y):
        folder = d["models"] / "cv" / f"fold_{fold}"
        members = [{"video_id": ids[i], "label": int(y[i]), "role": role}
                   for role, indices in (("fit", fit), ("early", early), ("outer_validation", outer)) for i in indices]
        save_csv(folder / "membership.csv", members)
        trained = train(target, X[fit], None if M is None else M[fit], y[fit],
                        X[early], None if M is None else M[early], y[early], seed, folder)
        probs = predict(target, trained, X[outer], y[outer])
        score = {"fold": fold, "seed": seed, "best_epoch": trained["best_epoch"],
                 "epochs_run": trained["epochs_run"], "n_fit": len(fit), "n_early": len(early),
                 **metrics(y[outer], probs)}
        scores.append(score)
        rows.extend({"fold": fold, "video_id": ids[i], "true_label": int(y[i]),
                     "probability_valid": float(probs[j]), "predicted_label": int(probs[j] >= 0.5)}
                    for j, i in enumerate(outer))
        print(f"{target} {source} fold {fold}: bal_acc={score['balanced_accuracy']:.4f} "
              f"f1_macro={score['f1_macro']:.4f} auc={score['roc_auc']:.4f} epoch={trained['best_epoch']} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    assert len(rows) == len(ids) and {r["video_id"] for r in rows} == set(ids)
    save_csv(d["results"] / "cv_metrics.csv", scores)
    save_csv(d["results"] / "cv_predictions.csv", rows)
    save_json(d["results"] / "cv_summary.json", summarize(scores))


def run_final(target, source):
    d = dirs(target, source)
    guard(d["results"] / "test_metrics.json")
    data = load_split(target, source)
    M = {role: data.get(f"M_{role}") for role in ("train", "validation", "test")}
    seeds = FRONT_SEEDS if target == "bp_front" else (sq.SEED,)
    candidates = []
    for seed in seeds:
        folder = d["models"] / f"seed_{seed}" if len(seeds) > 1 else d["models"]
        candidates.append(train(target, data["X_train"], M["train"], data["y_train"],
                                data["X_validation"], M["validation"], data["y_validation"], seed, folder))
    # bp_front_02: minima validation loss entre semillas (empate: menor semilla).
    chosen = min(candidates, key=lambda c: (c["best_loss"], c["seed"]))
    results = {}
    for role in ("validation", "test"):
        probs = predict(target, chosen, data[f"X_{role}"], data[f"y_{role}"])
        rows = [{"video_id": str(i), "true_label": int(t), "probability_valid": float(p), "predicted_label": int(p >= 0.5)}
                for i, t, p in zip(data[f"ids_{role}"], data[f"y_{role}"], probs)]
        save_csv(d["results"] / f"{role}_predictions.csv", rows)
        results[role] = metrics(data[f"y_{role}"], probs)
    save_json(d["results"] / "validation_metrics.json", {"seed": chosen["seed"], "best_epoch": chosen["best_epoch"],
                                                         **results["validation"]})
    save_json(d["results"] / "test_metrics.json", {"seed": chosen["seed"], "best_epoch": chosen["best_epoch"],
                                                   "candidates": [{k: c[k] for k in ("seed", "best_epoch", "best_loss", "epochs_run")}
                                                                  for c in candidates], **results["test"]})
    save_json(d["models"] / "config.json", {
        "target": target, "source": source, "run_id": f"{RUN}_{source}", "reference": TARGETS[target]["reference"],
        "keypoints_dir": str(d["keypoints"].relative_to(ROOT)), "selected_seed": chosen["seed"],
        "split_arrays_sha256": sha256(dirs(target)["data"] / f"{source}.npz"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__}})
    print(f"{target} {source} final: validation bal_acc={results['validation']['balanced_accuracy']:.4f}; "
          f"test bal_acc={results['test']['balanced_accuracy']:.4f} auc={results['test']['roc_auc']:.4f}", flush=True)


# ─── Comparacion pareada ────────────────────────────────────────────────────

def paired(target, name):
    """Predicciones de ambas fuentes alineadas por video_id."""
    tables = {s: {r["video_id"]: r for r in read_csv(dirs(target, s)["results"] / name)} for s in SOURCES}
    ids = sorted(tables["yolo"])
    if set(ids) != set(tables["mediapipe"]):
        raise ValueError(f"IDs distintos entre fuentes en {name}")
    y = np.asarray([int(tables["yolo"][i]["true_label"]) for i in ids])
    if any(int(tables["mediapipe"][i]["true_label"]) != t for i, t in zip(ids, y)):
        raise ValueError("Etiquetas distintas entre fuentes")
    probs = {s: np.asarray([float(tables[s][i]["probability_valid"]) for i in ids]) for s in SOURCES}
    folds_ = None
    if "fold" in tables["yolo"][ids[0]]:
        folds_ = np.asarray([int(tables["yolo"][i]["fold"]) for i in ids])
        if any(int(tables["mediapipe"][i]["fold"]) != f for i, f in zip(ids, folds_)):
            raise ValueError("Folds distintos entre fuentes")
    return ids, y, probs, folds_


def fast_scores(y, prob):
    """Mismas definiciones que metrics() (zero_division=0), en NumPy para el bootstrap."""
    pred = prob >= 0.5
    tp, tn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
    fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
    recall_valid, recall_invalid = tp / (tp + fn), tn / (tn + fp)
    precision_valid = tp / (tp + fp) if tp + fp else 0.0
    precision_invalid = tn / (tn + fn) if tn + fn else 0.0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    ranks = rankdata(prob)
    positives, negatives = int(np.sum(y == 1)), int(np.sum(y == 0))
    auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (recall_valid + recall_invalid) / 2,
            "f1_macro": (f1(precision_valid, recall_valid) + f1(precision_invalid, recall_invalid)) / 2,
            "roc_auc": auc, "recall_invalid": recall_invalid, "recall_valid": recall_valid}


def bootstrap_delta(y, probs):
    """IC95% pareado por video de mediapipe - yolo (percentiles)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = {key: [] for key in DELTA_KEYS}
    rejected = 0
    while len(deltas[DELTA_KEYS[0]]) < BOOTSTRAP:
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            rejected += 1
            continue
        a, b = fast_scores(y[idx], probs["yolo"][idx]), fast_scores(y[idx], probs["mediapipe"][idx])
        for key in DELTA_KEYS:
            deltas[key].append(b[key] - a[key])
    point = {s: metrics(y, probs[s]) for s in SOURCES}
    return {key: {"delta": point["mediapipe"][key] - point["yolo"][key],
                  "lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))}
            for key, values in deltas.items()} | {"_rejected_single_class": rejected, "_replicates": BOOTSTRAP}


def mcnemar(y, probs):
    correct = {s: (probs[s] >= 0.5).astype(int) == y for s in SOURCES}
    only_yolo = int(np.sum(correct["yolo"] & ~correct["mediapipe"]))
    only_mp = int(np.sum(~correct["yolo"] & correct["mediapipe"]))
    total = only_yolo + only_mp
    p = float(binomtest(min(only_yolo, only_mp), total, 0.5).pvalue) if total else 1.0
    return {"only_yolo_correct": only_yolo, "only_mediapipe_correct": only_mp,
            "both_correct": int(np.sum(correct["yolo"] & correct["mediapipe"])),
            "both_wrong": int(np.sum(~correct["yolo"] & ~correct["mediapipe"])), "exact_p_value": p}


def verdict(cv, oof_ci):
    """Regla fijada antes de ver resultados (misma de SQ): media CV de
    balanced_accuracy + f1_macro; concluyente solo si el IC95% pareado OOF de
    delta balanced_accuracy excluye 0."""
    score = {s: cv[s]["balanced_accuracy"]["mean"] + cv[s]["f1_macro"]["mean"] for s in SOURCES}
    leader = "mediapipe" if score["mediapipe"] > score["yolo"] else "yolo"
    ci = oof_ci["balanced_accuracy"]
    conclusive = ci["lower"] > 0 or ci["upper"] < 0
    return {"development_score": score, "leader": leader, "conclusive": bool(conclusive),
            "label": (f"{leader} superior (IC excluye 0)" if conclusive else
                      f"sin diferencia concluyente (ventaja descriptiva {leader})")}


def compare(target):
    d = dirs(target)
    summary = load_json(d["data"] / "summary.json")
    cv = {s: load_json(dirs(target, s)["results"] / "cv_summary.json") for s in SOURCES}
    fold_scores = {s: read_csv(dirs(target, s)["results"] / "cv_metrics.csv") for s in SOURCES}
    test = {s: load_json(dirs(target, s)["results"] / "test_metrics.json") for s in SOURCES}
    validation = {s: load_json(dirs(target, s)["results"] / "validation_metrics.json") for s in SOURCES}
    _, y_oof, p_oof, _ = paired(target, "cv_predictions.csv")
    _, y_test, p_test, _ = paired(target, "test_predictions.csv")
    per_fold = []
    for a, b in zip(fold_scores["yolo"], fold_scores["mediapipe"]):
        per_fold.append({"fold": int(a["fold"]), **{f"delta_{k}": float(b[k]) - float(a[k])
                                                     for k in ("balanced_accuracy", "f1_macro", "roc_auc")}})
    oof_ci = bootstrap_delta(y_oof, p_oof)
    result = {
        "target": target, "reference": summary["reference"], "cohort": summary["cohort"],
        "coverage": summary["coverage"], "cv": cv, "cv_per_fold_delta": per_fold,
        "mediapipe_better_folds_balanced_accuracy": sum(r["delta_balanced_accuracy"] > 0 for r in per_fold),
        "oof": {s: metrics(y_oof, p_oof[s]) for s in SOURCES}, "oof_delta_ci95": oof_ci,
        "oof_mcnemar": mcnemar(y_oof, p_oof),
        "validation": validation, "test": {s: metrics(y_test, p_test[s]) for s in SOURCES},
        "test_selected_seed": {s: test[s]["seed"] for s in SOURCES},
        "test_delta_ci95": bootstrap_delta(y_test, p_test), "test_mcnemar": mcnemar(y_test, p_test),
    }
    result["verdict"] = verdict(cv, oof_ci)
    save_json(d["compare"] / "comparison.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for source, color in zip(SOURCES, ("#2b6cb0", "#c05621")):
        values = [float(r["balanced_accuracy"]) for r in fold_scores[source]]
        axes[0].plot(range(1, 6), values, marker="o", color=color, label=source)
        fpr, tpr, _ = roc_curve(y_test, p_test[source])
        axes[1].plot(fpr, tpr, color=color, label=f"{source} AUC={result['test'][source]['roc_auc']:.3f}")
    axes[0].set(xlabel="Fold", ylabel="Balanced accuracy", title=f"{target}: CV desarrollo", xticks=range(1, 6))
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set(xlabel="FPR (inválido aprobado)", ylabel="TPR (válido aprobado)", title=f"{target}: test")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d["compare"] / "comparison.png", dpi=130)
    plt.close(fig)
    write_target_report(result, d["compare"] / "report.md")
    print(target, result["verdict"]["label"], flush=True)
    return result


def write_target_report(r, path):
    t = r["target"]
    lines = [f"# {t}: PoseCNN con YOLO vs MediaPipe", "",
             f"Receta final reutilizada: {r['reference']}. Solo cambia la fuente de keypoints.",
             "Clase positiva 1 = válido; umbral 0.5; matriz [[TN,FP],[FN,TP]]; FP = inválido aprobado.", "",
             "## Cohorte y cobertura", "",
             "| Partición | N | Válidos | Inválidos |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['n']} | {v['valid']} | {v['invalid']} |" for k, v in r["cohort"].items()]
    lines += ["", "| Fuente | Archivos | Utilizables | Excluidos del split base | Motivos |", "|---|---:|---:|---:|---|"]
    lines += [f"| {s} | {c['pose_files']} | {c['usable']} | {c['base_split_excluded']} | {c['base_split_exclusion_reasons'] or '—'} |"
              for s, c in r["coverage"].items()]
    lines += ["", "## CV 5-fold en desarrollo (media ± desviación poblacional)", "",
              "| Métrica | YOLO | MediaPipe |", "|---|---:|---:|"]
    lines += [f"| {k} | {r['cv']['yolo'][k]['mean']:.4f} ± {r['cv']['yolo'][k]['std']:.4f} | "
              f"{r['cv']['mediapipe'][k]['mean']:.4f} ± {r['cv']['mediapipe'][k]['std']:.4f} |" for k in SUMMARY_KEYS]
    lines += ["", f"Folds con mejor balanced accuracy MediaPipe: {r['mediapipe_better_folds_balanced_accuracy']}/5.", "",
              "## OOF pooled y diferencia pareada (MediaPipe − YOLO)", "",
              "| Métrica | YOLO | MediaPipe | Δ | IC95% bootstrap |", "|---|---:|---:|---:|---|"]
    for key in DELTA_KEYS:
        ci = r["oof_delta_ci95"][key]
        lines.append(f"| {key} | {r['oof']['yolo'][key]:.4f} | {r['oof']['mediapipe'][key]:.4f} | "
                     f"{ci['delta']:+.4f} | [{ci['lower']:+.4f}, {ci['upper']:+.4f}] |")
    m = r["oof_mcnemar"]
    lines += ["", f"McNemar OOF: solo YOLO acierta {m['only_yolo_correct']}, solo MediaPipe acierta "
                  f"{m['only_mediapipe_correct']}; p exacto = {m['exact_p_value']:.4f}.",
              f"Matrices OOF: YOLO {r['oof']['yolo']['matrix']}, MediaPipe {r['oof']['mediapipe']['matrix']}.", "",
              "## Test (evaluado una vez por fuente, después de la CV)", "",
              "| Métrica | YOLO | MediaPipe | Δ | IC95% bootstrap |", "|---|---:|---:|---:|---|"]
    for key in DELTA_KEYS:
        ci = r["test_delta_ci95"][key]
        lines.append(f"| {key} | {r['test']['yolo'][key]:.4f} | {r['test']['mediapipe'][key]:.4f} | "
                     f"{ci['delta']:+.4f} | [{ci['lower']:+.4f}, {ci['upper']:+.4f}] |")
    m = r["test_mcnemar"]
    lines += ["", f"McNemar test: solo YOLO acierta {m['only_yolo_correct']}, solo MediaPipe acierta "
                  f"{m['only_mediapipe_correct']}; p exacto = {m['exact_p_value']:.4f}.",
              f"Matrices test: YOLO {r['test']['yolo']['matrix']}, MediaPipe {r['test']['mediapipe']['matrix']}.",
              f"Semilla final seleccionada por validation: {r['test_selected_seed']}.", "",
              "## Veredicto", "",
              f"**{r['verdict']['label']}.** Regla fijada antes de entrenar: mayor media CV de balanced accuracy + "
              "F1 macro; se declara diferencia solo si el IC95% pareado OOF de Δ balanced accuracy excluye 0. "
              "Test es confirmación descriptiva (históricamente consultado con YOLO).", "",
              "![Comparación](comparison.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def global_report():
    results = {}
    for target in TARGETS:
        path = dirs(target)["compare"] / "comparison.json"
        if path.exists():
            results[target] = load_json(path)
    if not results:
        raise FileNotFoundError("No hay comparaciones completas")
    lines = ["# PoseCNN: keypoints YOLO vs MediaPipe", "",
             "Misma receta final por vista; mismo split y mismos folds para ambas fuentes; cohorte = videos "
             "utilizables en ambas. Δ = MediaPipe − YOLO. Clase positiva 1 = válido.", "",
             "## Desarrollo (CV 5-fold) — base de la decisión", "",
             "| Vista | N dev | Bal. acc YOLO | Bal. acc MP | F1 macro YOLO | F1 macro MP | AUC YOLO | AUC MP | "
             "Δ bal. acc OOF [IC95%] | Folds MP mejor | Veredicto |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for target, r in results.items():
        cv, ci = r["cv"], r["oof_delta_ci95"]["balanced_accuracy"]
        dev = r["cohort"]["train"]["n"] + r["cohort"]["validation"]["n"]
        lines.append(f"| {target} | {dev} | {cv['yolo']['balanced_accuracy']['mean']:.4f} | "
                     f"{cv['mediapipe']['balanced_accuracy']['mean']:.4f} | {cv['yolo']['f1_macro']['mean']:.4f} | "
                     f"{cv['mediapipe']['f1_macro']['mean']:.4f} | {cv['yolo']['roc_auc']['mean']:.4f} | "
                     f"{cv['mediapipe']['roc_auc']['mean']:.4f} | {ci['delta']:+.4f} [{ci['lower']:+.4f}, {ci['upper']:+.4f}] | "
                     f"{r['mediapipe_better_folds_balanced_accuracy']}/5 | {r['verdict']['label']} |")
    lines += ["", "## Test (confirmación descriptiva)", "",
              "| Vista | N test | Bal. acc YOLO | Bal. acc MP | AUC YOLO | AUC MP | Inválidos aprobados (FP) YOLO / MP | "
              "Δ bal. acc [IC95%] | McNemar p |", "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for target, r in results.items():
        t, ci = r["test"], r["test_delta_ci95"]["balanced_accuracy"]
        lines.append(f"| {target} | {t['yolo']['n']} | {t['yolo']['balanced_accuracy']:.4f} | "
                     f"{t['mediapipe']['balanced_accuracy']:.4f} | {t['yolo']['roc_auc']:.4f} | {t['mediapipe']['roc_auc']:.4f} | "
                     f"{t['yolo']['matrix'][0][1]} / {t['mediapipe']['matrix'][0][1]} | "
                     f"{ci['delta']:+.4f} [{ci['lower']:+.4f}, {ci['upper']:+.4f}] | {r['test_mcnemar']['exact_p_value']:.4f} |")
    lines += ["", "## Cobertura de keypoints", "",
              "| Vista | Utilizables YOLO | Utilizables MP | Excluidos del split base YOLO / MP |", "|---|---:|---:|---|"]
    for target, r in results.items():
        c = r["coverage"]
        lines.append(f"| {target} | {c['yolo']['usable']}/{c['yolo']['pose_files']} | "
                     f"{c['mediapipe']['usable']}/{c['mediapipe']['pose_files']} | "
                     f"{c['yolo']['base_split_excluded']} / {c['mediapipe']['base_split_excluded']} |")
    lines += ["", "Detalle por vista: `training/results/<ejercicio>/<vista>/posecnn/source_cmp_v1/report.md`.", ""]
    GLOBAL_REPORT.mkdir(parents=True, exist_ok=True)
    (GLOBAL_REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(GLOBAL_REPORT / "summary.json", {t: {"verdict": r["verdict"], "cv": r["cv"], "test": r["test"],
                                                   "oof_delta_ci95": r["oof_delta_ci95"]} for t, r in results.items()})
    print("Reporte global:", GLOBAL_REPORT / "report.md", flush=True)


def run_all(target, sources):
    if not (dirs(target)["data"] / "summary.json").exists():
        prepare(target)
    for source in sources:
        if not (dirs(target, source)["results"] / "cv_predictions.csv").exists():
            run_cv(target, source)
    for source in sources:
        if not (dirs(target, source)["results"] / "test_metrics.json").exists():
            run_final(target, source)
    if all((dirs(target, s)["results"] / "test_metrics.json").exists() for s in SOURCES):
        compare(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="all", choices=("all", *TARGETS))
    parser.add_argument("--stage", required=True, choices=("prepare", "cv", "final", "compare", "all", "report"))
    parser.add_argument("--source", default="both", choices=("both", *SOURCES))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "report":
        global_report()
        return
    targets = tuple(TARGETS) if args.target == "all" else (args.target,)
    sources = SOURCES if args.source == "both" else (args.source,)
    for target in targets:
        if args.stage == "prepare":
            prepare(target)
        elif args.stage == "cv":
            for source in sources:
                run_cv(target, source)
        elif args.stage == "final":
            for source in sources:
                run_final(target, source)
        elif args.stage == "compare":
            compare(target)
        else:
            run_all(target, sources)
    if args.stage == "all" and args.target == "all":
        global_report()


if __name__ == "__main__":
    main()
