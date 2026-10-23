"""Compara YOLO y MediaPipe en PoseCNN DL lateral con cohorte y split comunes.

Desde la raiz: python -m training.dl.compare_keypoint_sources --stage prepare
Luego --stage cv, --stage final y --stage compare (o --stage all).
Requiere los arrays YOLO V3 locales y los keypoints MediaPipe; nunca reescribe V3.
"""

import argparse
import csv
import hashlib
import importlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import sklearn
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from training.common.train_utils import (DEVICE, PoseCNN, augment_dataset, create_scaler,
                                         evaluate, scale_data, train_model)
from training.configs.training_config import get_config, physical_event


ROOT = Path(__file__).resolve().parents[2]
RUN = "source_cmp_v1"
VIEWS = ("left", "right")
SOURCES = ("yolo", "mediapipe")
PRE = importlib.import_module("training.dl.01_preprocessing")
TEMPORAL = importlib.import_module("training.dl.02_temporal_normalization")
BASE = importlib.import_module("training.dl.08_baseline_v2")
CV = importlib.import_module("training.dl.09_baseline_v2_cv")


def paths(view, source=None):
    base = ROOT / "training"
    result = {"data": base / "data" / "dl" / view / "posecnn" / RUN,
              "report": base / "results" / "dl" / view / "posecnn" / RUN}
    if source:
        result["model"] = base / "models" / "dl" / view / "posecnn" / "runs" / f"{RUN}_{source}"
        result["results"] = base / "results" / "dl" / view / "posecnn" / "runs" / f"{RUN}_{source}"
    return result


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def guard(path):
    if path.exists():
        raise FileExistsError(f"Ejecucion existente: {path}")


def prepare(view):
    config = get_config(f"dl_{view}")
    dest = paths(view)["data"]
    guard(dest / "manifest.json")
    # La fuente YOLO completa sobrevive en los arrays V3 aunque falten .npy locales.
    with np.load(config["normalized_path"]) as data:
        yolo = {str(i): (x, int(y)) for i, x, y in zip(
            data["video_ids"], data["keypoints"], data["labels"])}
    with np.load(config["split_path"]) as data:
        roles = {str(i): role for role in ("train", "val", "test")
                 for i in data[f"ids_{role}"].astype(str)}
        groups = {str(i): str(g) for role in ("train", "val", "test")
                  for i, g in zip(data[f"ids_{role}"], data[f"groups_{role}"])}
    if set(roles) != set(yolo):
        raise ValueError("El split YOLO no corresponde al array normalizado V3")
    labels = PRE.load_labels(config)
    mp_dir = ROOT / "keypoints" / "dl" / "mediapipe" / view
    mp = {}
    audit = []
    for ident in sorted(roles):
        path = mp_dir / f"{ident}.npy"
        reason = ""
        if not path.exists():
            reason = "missing_file"
        else:
            try:
                raw = np.load(path, allow_pickle=False)
                clean = PRE.validate_and_clean(raw, 2)
                if clean is None:
                    reason = "invalid_shape_or_values"
                else:
                    clean = PRE.interpolate_small_gaps(clean)
                    if np.isnan(clean[:, :, :2]).any(axis=(1, 2)).sum() > .3 * len(clean):
                        reason = "nan_ratio_over_30pct"
                    else:
                        # Misma convencion de indices usada por V3; el extractor
                        # guarda [cadera, hombro] en ambas fuentes.
                        sequence = PRE.normalize_spatial(clean, config, view)
                        sequence = TEMPORAL.normalize_temporal(sequence, 170)
                        if not np.isfinite(sequence).all():
                            reason = "nonfinite_after_temporal"
                        else:
                            mp[ident] = sequence
            except (ValueError, OSError) as error:
                reason = f"read_error:{error}"
        audit.append({"video_id": ident, "physical_event": groups[ident], "split": roles[ident],
                      "mediapipe_usable": ident in mp, "reason": reason})
    for ident, (sequence, label) in yolo.items():
        if sequence.shape != (170, 2, 3) or not np.isfinite(sequence).all():
            raise ValueError(f"YOLO V3 invalido: {ident}")
        if label != labels[ident] or groups[ident] != physical_event(ident, view)[0]:
            raise ValueError(f"Label o mapping fisico inconsistente: {ident}")
    ids = {role: sorted(i for i in mp if roles[i] == role) for role in ("train", "val", "test")}
    for role, values in ids.items():
        if {labels[i] for i in values} != {0, 1}:
            raise ValueError(f"Particion {role} sin ambas clases")
    if set(groups[i] for i in ids["train"]) & set(groups[i] for i in ids["test"]):
        raise ValueError("Leakage de evento fisico")
    dest.mkdir(parents=True, exist_ok=True)
    save_csv(dest / "cohort.csv", audit)
    for source, sequences in (("yolo", {i: x for i, (x, _) in yolo.items()}), ("mediapipe", mp)):
        arrays = {}
        for role, values in ids.items():
            arrays[f"X_{role}"] = np.stack([sequences[i] for i in values]).astype(np.float32)
            arrays[f"y_{role}"] = np.asarray([labels[i] for i in values], dtype=np.int64)
            arrays[f"ids_{role}"] = np.asarray(values)
            arrays[f"groups_{role}"] = np.asarray([groups[i] for i in values])
        np.savez_compressed(dest / f"{source}.npz", **arrays)
    manifest = {"view": view, "run_id": RUN, "reference": "DL lateral PoseCNN baseline V3 manual audit",
                "keypoint_order": "0=cadera, 1=hombro (convencion V3)",
                "confidence_note": "visibility MediaPipe y confidence YOLO no son equivalentes",
                "cohort": {role: {"n": len(v), "valid": sum(labels[i] for i in v)} for role, v in ids.items()},
                "excluded_mediapipe": len(roles) - len(mp),
                "input_sha256": {"labels": hash_file(config["csv_path"]),
                                 "yolo_normalized": hash_file(config["normalized_path"]),
                                 "v3_split": hash_file(config["split_path"]),
                                 "v3_folds": hash_file(ROOT / "training/data/dl/shared/cv_event_folds_v3.csv"),
                                 **{f"{s}_cohort": hash_file(dest / f"{s}.npz") for s in SOURCES},
                                 "code": hash_file(Path(__file__))},
                "mediapipe_files_sha256": {i: hash_file(mp_dir / f"{i}.npy") for i in sorted(mp)},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "sklearn": sklearn.__version__}}
    save_json(dest / "manifest.json", manifest)
    print(view, "cohorte", manifest["cohort"], "MP excluidos", manifest["excluded_mediapipe"], flush=True)


def load(view, source):
    with np.load(paths(view)["data"] / f"{source}.npz") as data:
        return {key: data[key] for key in data.files}


def fit(view, X, y, X_val, y_val, folder):
    folder.mkdir(parents=True, exist_ok=False)
    scaler = create_scaler(X)
    assert int(scaler.n_samples_seen_) == len(X)
    with (folder / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    aug, aug_y = augment_dataset(X, y, np.random.RandomState(42), 3,
                                 transforms=get_config(f"dl_{view}")["augmentation"])
    BASE.np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = PoseCNN(T=170, K=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=5)
    model, history, epoch = train_model(model, BASE.make_loader(scale_data(scaler, aug).astype(np.float32), aug_y, True),
                                       BASE.make_loader(scale_data(scaler, X_val).astype(np.float32), y_val),
                                       torch.nn.BCELoss(), optimizer, scheduler, 200, 8, folder, verbose=False)
    np.savez(folder / "history.npz", **history)
    return model, scaler, epoch


def predictions(model, scaler, X, y, ids, groups, fold=None):
    loader = BASE.make_loader(scale_data(scaler, X).astype(np.float32), y)
    _, pred, truth, prob = evaluate(model, loader, torch.nn.BCELoss())
    return [{"video_id": str(i), "physical_event": str(g), "fold": fold if fold is not None else "",
             "true_label": int(t), "prediction": int(p), "probability": float(v)}
            for i, g, t, p, v in zip(ids, groups, truth, pred, prob)]


def scores(rows):
    y = np.asarray([int(r["true_label"]) for r in rows])
    p = np.asarray([float(r["probability"]) for r in rows])
    return BASE.metrics_with_specificity(y, (p >= .5).astype(int), p)


def cv(view, source):
    dest = paths(view, source)
    guard(dest["results"] / "cv_predictions.csv")
    data = load(view, source)
    fold_rows = read_csv(ROOT / "training/data/dl/shared/cv_event_folds_v3.csv")
    folds = {r["physical_event"]: int(r["fold_id"]) for r in fold_rows}
    X = np.concatenate([data["X_train"], data["X_val"]])
    y = np.concatenate([data["y_train"], data["y_val"]])
    ids = np.concatenate([data["ids_train"], data["ids_val"]])
    groups = np.concatenate([data["groups_train"], data["groups_val"]])
    assigned = np.asarray([folds[g] for g in groups])
    output, summary = [], []
    for fold in range(5):
        fit_mask, val_mask = assigned != fold, assigned == fold
        if {int(v) for v in y[val_mask]} != {0, 1}:
            raise ValueError(f"Fold {fold} sin ambas clases")
        model, scaler, epoch = fit(view, X[fit_mask], y[fit_mask], X[val_mask], y[val_mask],
                                   dest["model"] / "cv" / f"fold_{fold}")
        rows = predictions(model, scaler, X[val_mask], y[val_mask], ids[val_mask], groups[val_mask], fold)
        output.extend(rows)
        summary.append({"fold": fold, "best_epoch": epoch, "n": len(rows), **scores(rows)})
        print(view, source, "fold", fold, summary[-1]["f1"], flush=True)
    assert len(output) == len(X) and len({r["video_id"] for r in output}) == len(X)
    save_csv(dest["results"] / "cv_predictions.csv", output)
    save_json(dest["results"] / "cv_metrics.json", summary)


def final(view, source):
    dest = paths(view, source)
    guard(dest["results"] / "test_predictions.csv")
    data = load(view, source)
    model, scaler, epoch = fit(view, data["X_train"], data["y_train"],
                               data["X_val"], data["y_val"], dest["model"] / "final")
    result = {"best_epoch": epoch, "source": source, "view": view}
    for role in ("val", "test"):
        rows = predictions(model, scaler, data[f"X_{role}"], data[f"y_{role}"],
                           data[f"ids_{role}"], data[f"groups_{role}"])
        save_csv(dest["results"] / f"{role}_predictions.csv", rows)
        result[role] = scores(rows)
    save_json(dest["results"] / "metrics.json", result)
    save_json(dest["model"] / "config.json", {"view": view, "source": source, "run_id": RUN,
                                              "seed": 42, "baseline": "V3", "best_epoch": epoch,
                                              "cohort_sha256": hash_file(paths(view)["data"] / f"{source}.npz"),
                                              "code_sha256": hash_file(Path(__file__))})
    print(view, source, "test", result["test"], flush=True)


def paired(view, stage):
    name = "cv_predictions.csv" if stage == "cv" else "test_predictions.csv"
    tables = {s: {r["video_id"]: r for r in read_csv(paths(view, s)["results"] / name)} for s in SOURCES}
    ids = sorted(tables["yolo"])
    if set(ids) != set(tables["mediapipe"]):
        raise ValueError("La cohorte no coincide entre fuentes")
    for ident in ids:
        a, b = (tables[s][ident] for s in SOURCES)
        if (a["physical_event"], a["true_label"], a["fold"]) != (b["physical_event"], b["true_label"], b["fold"]):
            raise ValueError(f"Etiqueta, evento o fold distinto: {ident}")
    rows = {s: [tables[s][i] for i in ids] for s in SOURCES}
    y = np.asarray([int(r["true_label"]) for r in rows["yolo"]])
    correct = {s: np.asarray([int(r["prediction"]) for r in rows[s]]) == y for s in SOURCES}
    only_yolo = int(np.sum(correct["yolo"] & ~correct["mediapipe"]))
    only_mp = int(np.sum(~correct["yolo"] & correct["mediapipe"]))
    discordant = only_yolo + only_mp
    rng = np.random.default_rng(20261023)
    deltas = {key: [] for key in ("accuracy", "f1", "roc_auc", "specificity")}
    probabilities = {s: np.asarray([float(r["probability"]) for r in rows[s]]) for s in SOURCES}
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        a = BASE.metrics_with_specificity(y[idx], probabilities["yolo"][idx] >= .5, probabilities["yolo"][idx])
        b = BASE.metrics_with_specificity(y[idx], probabilities["mediapipe"][idx] >= .5, probabilities["mediapipe"][idx])
        for key in deltas:
            deltas[key].append(b[key] - a[key])
    return {"n": len(ids), "yolo": scores(rows["yolo"]), "mediapipe": scores(rows["mediapipe"]),
            "delta_mediapipe_minus_yolo_ci95": {k: [float(v) for v in np.percentile(values, [2.5, 97.5])]
                                                 for k, values in deltas.items()},
            "mcnemar": {"only_yolo_correct": only_yolo, "only_mediapipe_correct": only_mp,
                        "exact_p": float(binomtest(min(only_yolo, only_mp), discordant).pvalue) if discordant else 1.0}}


def compare(view):
    out = paths(view)["report"]
    guard(out / "comparison.json")
    manifest = json.loads((paths(view)["data"] / "manifest.json").read_text(encoding="utf-8"))
    result = {"view": view, "cohort": manifest["cohort"], "cv_oof": paired(view, "cv"),
              "test": paired(view, "test")}
    save_json(out / "comparison.json", result)
    lines = [f"# DL {view}: YOLO frente a MediaPipe (PoseCNN V3)", "",
             "Ambas fuentes se reentrenaron con la misma cohorte, split por evento físico y folds V3.",
             "El test se evaluó tras CV. Los resultados YOLO finales reproducen V3; "
             "los folds CV se reentrenaron con orden de muestras por split, por lo que su "
             "trayectoria no es idéntica a la ejecución CV V3 congelada.",
             "El canal visibility de MediaPipe no equivale al confidence de YOLO.",
             "Se conserva la convención de normalización V3 (índice 0 como hombro aunque ambos extractores guardan cadera primero).", "",
             f"Cohorte: {manifest['cohort']}; MediaPipe excluidos: {manifest['excluded_mediapipe']}.", "",
             "| Partición | Fuente | N | Accuracy | F1 válido | AUC | Especificidad | FP | FN |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for stage in ("cv_oof", "test"):
        for source in SOURCES:
            m = result[stage][source]
            lines.append(f"| {stage} | {source} | {result[stage]['n']} | {m['accuracy']:.4f} | {m['f1']:.4f} | "
                         f"{m['roc_auc']:.4f} | {m['specificity']:.4f} | {m['fp']} | {m['fn']} |")
        lines.append(f"\n{stage}: IC95% bootstrap pareado de Δ (MediaPipe−YOLO): "
                     f"{result[stage]['delta_mediapipe_minus_yolo_ci95']}; "
                     f"McNemar exacto: {result[stage]['mcnemar']}.\n")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(view, "reporte", out / "report.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("all", *VIEWS), default="all")
    parser.add_argument("--source", choices=("both", *SOURCES), default="both")
    parser.add_argument("--stage", choices=("prepare", "cv", "final", "compare", "all"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    views = VIEWS if args.view == "all" else (args.view,)
    sources = SOURCES if args.source == "both" else (args.source,)
    for view in views:
        for stage in (("prepare", "cv", "final", "compare") if args.stage == "all" else (args.stage,)):
            if stage == "prepare":
                prepare(view)
            elif stage == "compare":
                compare(view)
            else:
                for source in sources:
                    (cv if stage == "cv" else final)(view, source)


if __name__ == "__main__":
    main()
