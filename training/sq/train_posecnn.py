"""PoseCNN independiente para squat (YOLO, una red por vista).

Uso (desde la raiz del repo):
    python -m training.sq.train_posecnn --view front --stage audit
    python -m training.sq.train_posecnn --view front --stage all --run-id baseline_t170_v1
Repetir con --view left. `all` audita, prepara, ejecuta CV y finalmente test.
Solo existen front y left (sin CSV de right).
La preparacion estable vive en training/data/sq/<view>/posecnn; cada ejecucion
guarda modelos en training/models/sq/<view>/posecnn/runs/<run-id> y resultados
(incluidos logs) en training/results/sq/<view>/posecnn/runs/<run-id>.
"""

import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.common.train_utils import PoseCNN, augment_dataset


ROOT = Path(__file__).resolve().parents[2]
EXERCISE = "sq"
VIEWS = ("front", "left")
DEFAULT_RUN_ID = "baseline_t170_v1"
SEED = 42
FRAMES = 170
MIN_FRAMES = 10
MAX_GAP = 5
MAX_MISSING = 0.10
AUG_FACTOR = 3
BATCH = 16
EPOCHS = 200
PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Esquema YOLO SQ (extract_keypoints/sq/extract_keypoints_yolo_sq.py).
# front K=16 (sq_front_skeleton_v5.pt); left K=5 (sq_side_skeleton_v3.pt:
# rodilla, cadera, tobillo, punta, talon). Los lados frontales se dedujeron
# del esqueleto de visualizacion; no se comparan lados entre vistas.
VIEW_CONFIGS = {
    "front": {
        "keypoints": 16,
        "points": ("hombro_izq", "hombro_der", "cadera_der", "cadera_izq",
                   "rodilla_izq", "rodilla_der", "tobillo_der", "punta_der",
                   "talon_der", "tobillo_izq", "punta_izq", "talon_izq",
                   "codo_izq", "codo_der", "muneca_der", "muneca_izq"),
        # Centro fijo: mediana temporal del punto medio de hombros (0,1);
        # escala fija: mediana de la distancia interhombros. El descenso de
        # cadera respecto a rodillas queda intacto al ser centro/escala fijos.
        "center_pair": (0, 1),
        "scale_pair": (0, 1),
        # Subset "legs": caderas, rodillas, tobillos, puntas y talones
        # (10 puntos, 5100 features). La normalizacion se calcula igual con
        # hombros; el subset se aplica despues, solo cambia la entrada.
        "subsets": {"legs": (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)},
    },
    "left": {
        "keypoints": 5,
        "points": ("rodilla", "cadera", "tobillo", "punta", "talon"),
        # Centro fijo: mediana temporal de la cadera (1); escala fija:
        # mediana del segmento rodilla-tobillo (0,2), rigido (tibia), de
        # modo que el descenso de cadera (profundidad) se conserva.
        "center_index": 1,
        "scale_pair": (0, 2),
    },
}


def view_config(view):
    if view not in VIEW_CONFIGS:
        raise ValueError(f"Vista desconocida: {view}. Disponibles: {sorted(VIEW_CONFIGS)}")
    return VIEW_CONFIGS[view]


def paths(view, run_id=DEFAULT_RUN_ID):
    base = ROOT / "training"
    return {
        "source": ROOT / "keypoints" / EXERCISE / "yolo" / view,
        "labels": ROOT / "etiquetado" / EXERCISE / f"{EXERCISE}_{view}.csv",
        "data": base / "data" / EXERCISE / view / "posecnn",
        "models": base / "models" / EXERCISE / view / "posecnn" / "runs" / run_id,
        "results": base / "results" / EXERCISE / view / "posecnn" / "runs" / run_id,
        "run_id": run_id,
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_labels(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "resultado", "label", "archivo"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV vacio o columnas faltantes: {path}")
    return rows


def longest_run(mask):
    maximum = current = 0
    for value in mask:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def inspect_file(path, keypoints, view):
    row = {"video_id": path.stem, "status": "excluded", "reasons": "", "warnings": "", "shape": "",
           "frames": 0, "missing_xy_ratio": None, "missing_frames": 0, "max_gap": 0,
           "median_confidence": None, "norm_scale": None, "sha256": ""}
    try:
        data = np.load(path, allow_pickle=False)
        row["shape"] = str(data.shape)
        if data.ndim != 3 or data.shape[1:] != (keypoints, 3) or data.dtype.kind not in "fiu":
            row["reasons"] = "shape_or_dtype"
            return row
        row["frames"] = len(data)
        digest = hashlib.sha256()
        digest.update(str(data.shape).encode())
        digest.update(np.ascontiguousarray(data).tobytes())
        row["sha256"] = digest.hexdigest()
        reasons = []
        if len(data) < MIN_FRAMES:
            reasons.append("too_few_frames")
        if np.isinf(data).any():
            reasons.append("contains_inf")
        xy = data[..., :2]
        # YOLO devuelve (0,0) al no detectar un punto; conf=0 lo confirma.
        missing = ~np.isfinite(xy).all(axis=2) | ((xy == 0).all(axis=2) & (data[..., 2] == 0))
        row["missing_xy_ratio"] = float(missing.mean()) if len(data) else 1.0
        row["missing_frames"] = int(missing.all(axis=1).sum())
        row["max_gap"] = max((longest_run(missing[:, i]) for i in range(keypoints)), default=0)
        if row["missing_xy_ratio"] > MAX_MISSING:
            reasons.append("missing_xy_above_10pct")
        if row["max_gap"] > MAX_GAP:
            reasons.append("missing_gap_above_5")
        if len(data) and (missing[0].any() or missing[-1].any()):
            reasons.append("boundary_missing_not_interpolable")
        conf = data[..., 2]
        observed = conf[np.isfinite(conf)]
        if observed.size:
            row["median_confidence"] = float(np.median(observed))
        if np.any((observed < 0) | (observed > 1)):
            reasons.append("confidence_out_of_range")
        # Los frames sin persona tienen los tres canales NaN; se rellenan
        # exclusivamente los huecos interiores y se conserva confianza=0.
        if np.any(~np.isfinite(conf) & ~missing):
            reasons.append("confidence_missing_with_xy_present")
        pair = view_scale_pair(view)
        valid = ~missing[:, pair[0]] & ~missing[:, pair[1]]
        if valid.any():
            dist = np.linalg.norm(xy[valid, pair[0]] - xy[valid, pair[1]], axis=1)
            row["norm_scale"] = float(np.median(dist))
        if row["norm_scale"] is None or row["norm_scale"] <= 1e-6:
            reasons.append("invalid_norm_scale")
        warnings = []
        if row["missing_xy_ratio"] > 0:
            warnings.append("interpolated_missing_xy")
        if row["median_confidence"] is not None and row["median_confidence"] < 0.5:
            warnings.append("low_median_confidence")
        row["warnings"] = "|".join(warnings)
        row["status"] = "excluded" if reasons else "usable"
        row["reasons"] = "|".join(reasons)
    except (OSError, ValueError, TypeError) as error:
        row["reasons"] = f"load_error:{type(error).__name__}"
    return row


def view_scale_pair(view):
    cfg = view_config(view)
    return cfg["scale_pair"]


def resolve_subset(view, subset):
    """Devuelve (nombre, indices|None). Solo front admite subsets."""
    if subset in (None, "all"):
        return "all", None
    cfg = view_config(view)
    subsets = cfg.get("subsets") or {}
    if subset not in subsets:
        raise ValueError(f"Subset '{subset}' no disponible para {view}. Disponibles: {sorted(subsets)}")
    return subset, subsets[subset]


def apply_subset(X, indices):
    if indices is None:
        return np.asarray(X, dtype=np.float32)
    return np.asarray(X, dtype=np.float32)[:, :, list(indices), :]


def audit(view):
    cfg = view_config(view)
    p = paths(view)
    for name in ("source", "labels"):
        if not p[name].exists():
            raise FileNotFoundError(p[name])
    p["data"].mkdir(parents=True, exist_ok=True)
    rows = load_labels(p["labels"])
    label_ids = [row["video_id"].strip() for row in rows]
    files = sorted(p["source"].glob("*.npy"))
    pose_ids = {path.stem for path in files}
    issues = []
    duplicates = sorted(key for key, count in Counter(label_ids).items() if count > 1)
    if duplicates:
        issues.append("duplicate_label_ids")
    contradictory = []
    bad_labels = []
    blank_label_ids = []
    archivo_mismatch = []
    for row in rows:
        ident = row["video_id"].strip()
        # Fila completamente vacia (sin etiqueta posible): se reporta y su
        # pose se excluye con motivo explicito; no bloquea ni se "corrige".
        if not row["label"].strip():
            blank_label_ids.append(ident)
            continue
        if row["label"].strip() not in ("0", "1"):
            bad_labels.append(ident)
        elif row["resultado"].strip().lower() != {"0": "invalido", "1": "valido"}[row["label"].strip()]:
            contradictory.append(ident)
        # La columna archivo es metadata cam1 heredada: no inferir ID por ella.
        if Path(row["archivo"].strip()).stem not in (ident, ident + "_cam1", ident + "_cam2", ident + "_cam3"):
            archivo_mismatch.append(ident)
    if bad_labels or contradictory:
        issues.append("invalid_or_contradictory_labels")
    manifest = [inspect_file(file, cfg["keypoints"], view) for file in files]
    hashes = {}
    exact_duplicates = []
    for row in manifest:
        digest = row["sha256"]
        if digest and digest in hashes:
            row["status"] = "excluded"
            row["reasons"] = (row["reasons"] + "|" if row["reasons"] else "") + f"duplicate_of_{hashes[digest]}"
            exact_duplicates.append((row["video_id"], hashes[digest]))
        elif digest:
            hashes[digest] = row["video_id"]
    labels_by_id = {row["video_id"].strip(): row["label"].strip() for row in rows}
    save_csv(p["data"] / "audit.csv", manifest, list(manifest[0]) if manifest else
             ["video_id", "status", "reasons", "warnings", "shape", "frames", "missing_xy_ratio", "missing_frames", "max_gap", "median_confidence", "norm_scale", "sha256"])
    sizes = [r["frames"] for r in manifest if r["frames"]]
    summary = {
        "exercise": EXERCISE, "view": view, "source": str(p["source"]), "labels": str(p["labels"]),
        "extractor": "extract_keypoints/sq/extract_keypoints_yolo_sq.py",
        "keypoints": list(enumerate(cfg["points"])), "channels": ["xyn_x", "xyn_y", "yolo_keypoint_confidence"],
        "pose_files": len(files), "csv_rows": len(rows),
        "label_counts": dict(Counter(row["label"].strip() for row in rows)),
        "pose_without_label": sorted(pose_ids - set(label_ids)),
        "label_without_pose": sorted(set(label_ids) - pose_ids),
        "duplicate_label_ids": duplicates, "bad_labels": bad_labels,
        "blank_label_ids": blank_label_ids,
        "resultado_mismatch": contradictory, "archivo_stem_mismatch": archivo_mismatch,
        "exact_pose_duplicates": exact_duplicates,
        "duplicate_label_conflicts": [pair for pair in exact_duplicates
                                      if labels_by_id.get(pair[0]) != labels_by_id.get(pair[1])],
        "quality_counts": dict(Counter(row["status"] for row in manifest)),
        "shape_counts": dict(Counter("(T,%d,3)" % cfg["keypoints"] if row["shape"].endswith(f", {cfg['keypoints']}, 3)") else row["shape"] for row in manifest)),
        "exclusion_reasons": dict(Counter(reason for row in manifest for reason in row["reasons"].split("|") if reason)),
        "quality_warnings": dict(Counter(reason for row in manifest for reason in row["warnings"].split("|") if reason)),
        "original_frames": {"min": min(sizes) if sizes else None,
                            "median": float(np.median(sizes)) if sizes else None,
                            "p95": float(np.percentile(sizes, 95)) if sizes else None,
                            "max": max(sizes) if sizes else None,
                            "fraction_below_170": float(np.mean(np.asarray(sizes) < FRAMES)) if sizes else None},
        "blocking_issues": issues,
    }
    save_json(p["data"] / "audit_summary.json", summary)
    print(f"{view}: {len(files)} pose, {len(rows)} labels; clases {summary['label_counts']}; "
          f"calidad {summary['quality_counts']}; duplicados {exact_duplicates}; "
          f"labels inconsistentes {summary['blocking_issues']}; "
          f"archivo metadata discordante {len(archivo_mismatch)}")
    return summary, manifest, rows


def fill_gaps(data):
    result = np.asarray(data, dtype=np.float32).copy()
    missing = ~np.isfinite(result[..., :2]).all(axis=2) | ((result[..., :2] == 0).all(axis=2) & (result[..., 2] == 0))
    result[..., :2][missing] = np.nan
    source = np.arange(len(result))
    for point in range(result.shape[1]):
        for channel in range(2):
            values = result[:, point, channel]
            valid = np.isfinite(values)
            if valid.sum() < 2 or not valid[0] or not valid[-1] or longest_run(~valid) > MAX_GAP:
                raise ValueError("non_interpolable_coordinates")
            result[:, point, channel] = np.interp(source, source[valid], values[valid])
    # Confidence no es geometria: no inventar confianza en frames sin deteccion.
    result[..., 2][~np.isfinite(result[..., 2])] = 0
    return result


def preprocess_front(sequence, target_frames=FRAMES, scale_mode="shoulders"):
    data = fill_gaps(sequence)
    cfg = VIEW_CONFIGS["front"]
    a, b = cfg["center_pair"]
    shoulders = data[:, [a, b], :2]
    center = np.median((shoulders[:, 0] + shoulders[:, 1]) / 2, axis=0)
    if scale_mode == "shoulders":
        scale = float(np.median(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)))
    elif scale_mode == "shanks":
        # Índices YOLO originales: rodilla izq 4 ↔ tobillo izq 9;
        # rodilla der 5 ↔ tobillo der 6. Mediana conjunta de las
        # distancias finitas, positivas y observadas antes de interpolar.
        source = np.asarray(sequence, dtype=float)
        lengths = []
        for knee, ankle in ((4, 9), (5, 6)):
            pair = source[:, [knee, ankle], :2]
            confidence = source[:, [knee, ankle], 2]
            valid = np.isfinite(pair).all(axis=(1, 2)) & np.isfinite(confidence).all(axis=1)
            valid &= (confidence > 0).all(axis=1)
            distances = np.linalg.norm(pair[valid, 0] - pair[valid, 1], axis=1)
            lengths.extend(distances[np.isfinite(distances) & (distances > 1e-6)])
        scale = float(np.median(lengths)) if lengths else np.nan
    else:
        raise ValueError(f"Modo de escala desconocido: {scale_mode}")
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("invalid_norm_scale")
    return resample(data, center, scale, target_frames)


def preprocess_left(sequence):
    data = fill_gaps(sequence)
    cfg = VIEW_CONFIGS["left"]
    hip = data[:, cfg["center_index"], :2]
    a, b = cfg["scale_pair"]
    center = np.median(hip, axis=0)
    scale = float(np.median(np.linalg.norm(data[:, a, :2] - data[:, b, :2], axis=1)))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("invalid_norm_scale")
    return resample(data, center, scale)


def resample(data, center, scale, target_frames=FRAMES):
    # Referencia fija por video: conserva el movimiento relativo (descenso de
    # cadera, flexion de rodilla) y la confianza. Sin espejo ni normalizacion
    # frame a frame.
    data[..., :2] = (data[..., :2] - center) / scale
    target = np.linspace(0, len(data) - 1, target_frames)
    source = np.arange(len(data))
    result = np.empty((target_frames, data.shape[1], 3), dtype=np.float32)
    for point in range(data.shape[1]):
        for channel in range(3):
            result[:, point, channel] = np.interp(target, source, data[:, point, channel])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite_after_preprocessing")
    return result, center.astype(np.float32), float(scale)


PREPROCESSORS = {"front": preprocess_front, "left": preprocess_left}


def prepare(view):
    cfg = view_config(view)
    p = paths(view)
    summary, manifest, label_rows = audit(view)
    if summary["blocking_issues"]:
        raise ValueError(f"Inconsistencias de etiquetas: {summary['blocking_issues']}; ver audit_summary.json")
    labels = {row["video_id"].strip(): int(row["label"].strip()) for row in label_rows if row["label"].strip()}
    blank = {row["video_id"].strip() for row in label_rows if not row["label"].strip()}
    preprocess = PREPROCESSORS[view]
    ids, arrays, targets = [], [], []
    for row in manifest:
        ident = row["video_id"]
        if ident in blank:
            row["status"] = "excluded"
            row["reasons"] = (row["reasons"] + "|" if row["reasons"] else "") + "blank_label_in_csv"
        if row["status"] != "usable" or ident not in labels:
            continue
        try:
            sequence, center, scale = preprocess(np.load(p["source"] / f"{ident}.npy", allow_pickle=False))
        except (ValueError, OSError) as error:
            row["status"] = "excluded"
            row["reasons"] = str(error)
            continue
        row["center_x"], row["center_y"], row["scale"] = float(center[0]), float(center[1]), scale
        ids.append(ident)
        arrays.append(sequence)
        targets.append(labels[ident])
    save_csv(p["data"] / "preprocessing.csv", manifest,
             ["video_id", "status", "reasons", "warnings", "shape", "frames", "missing_xy_ratio", "missing_frames", "max_gap", "median_confidence", "norm_scale", "sha256", "center_x", "center_y", "scale"])
    if not arrays or min(Counter(targets).values()) < 5:
        raise ValueError("No hay suficientes secuencias de ambas clases para CV")
    X = np.stack(arrays)
    y = np.asarray(targets, dtype=np.int64)
    ids = np.asarray(ids)
    np.savez_compressed(p["data"] / "processed.npz", X=X, y=y, ids=ids)
    idx = np.arange(len(y))
    development, test = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
    train, validation = train_test_split(development, test_size=0.15 / 0.85,
                                         stratify=y[development], random_state=SEED + 1)
    roles = np.full(len(y), "", dtype="<U10")
    for role, indices in (("train", train), ("validation", validation), ("test", test)):
        roles[indices] = role
        if set(y[indices]) != {0, 1}:
            raise ValueError(f"Particion {role} sin ambas clases")
    save_csv(p["data"] / "split.csv", [{"video_id": ident, "label": int(label), "split": role, "seed": SEED}
                                          for ident, label, role in zip(ids, y, roles)],
             ["video_id", "label", "split", "seed"])
    np.savez_compressed(p["data"] / "split.npz", X_train=X[train], y_train=y[train], ids_train=ids[train],
                        X_validation=X[validation], y_validation=y[validation], ids_validation=ids[validation],
                        X_test=X[test], y_test=y[test], ids_test=ids[test])
    print(f"{view}: preparados {len(ids)} videos; train={len(train)}, validation={len(validation)}, test={len(test)}")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform(scaler, X):
    return scaler.transform(X.reshape(len(X), -1)).reshape(X.shape).astype(np.float32)


def scaled_arrays(X_fit, y_fit, X_val, seed, scaler_path):
    scaler = StandardScaler().fit(X_fit.reshape(len(X_fit), -1))
    assert scaler.n_samples_seen_ == len(X_fit)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    # Traslacion, escala, ruido y shift temporal del pipeline DL/BP; sin
    # rotacion (la orientacion tiene significado) ni flips. El shift desplaza
    # toda la secuencia, incluida la confianza, sin alterar sus valores.
    aug, labels = augment_dataset(X_fit, y_fit, np.random.RandomState(seed), AUG_FACTOR,
                                  transforms=("translation", "scaling", "noise", "temporal_shift"))
    assert np.allclose(aug[::AUG_FACTOR + 1, :, :, 2], X_fit[:, :, :, 2])
    return transform(scaler, aug), labels, transform(scaler, X_val), scaler


def loader(X, y, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(X).permute(0, 3, 1, 2),
                                    torch.from_numpy(np.asarray(y, dtype=np.float32))),
                      batch_size=BATCH, shuffle=shuffle, generator=generator)


def evaluate(model, X, y, seed):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch, _ in loader(X, y, False, seed):
            probabilities.extend(model(batch.to(DEVICE)).cpu().numpy().reshape(-1).tolist())
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    matrix = confusion_matrix(y, predictions, labels=[0, 1]).tolist()
    metrics = {"accuracy": float(accuracy_score(y, predictions)),
               "precision": float(precision_score(y, predictions, zero_division=0)),
               "recall": float(recall_score(y, predictions, zero_division=0)),
               "f1": float(f1_score(y, predictions, zero_division=0)),
               "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
               "confusion_matrix": matrix, "positive_class": "1=valido"}
    return metrics, predictions, probabilities


def fit(X_fit, y_fit, X_early, y_early, seed, model_dir, patience=PATIENCE):
    frames, keypoints = X_fit.shape[1], X_fit.shape[2]
    seed_all(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = PoseCNN(T=frames, K=keypoints, C=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    criterion = nn.BCELoss()
    train_loader = loader(X_fit, y_fit, True, seed)
    early_loader = loader(X_early, y_early, False, seed)
    best, stale, best_epoch = float("inf"), 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch.to(DEVICE)), labels.to(DEVICE).unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses.append((float(loss.item()), len(labels)))
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch, labels in early_loader:
                loss = criterion(model(batch.to(DEVICE)), labels.to(DEVICE).unsqueeze(1))
                val_loss += float(loss.item()) * len(labels)
                count += len(labels)
        val_loss /= count
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": sum(loss * n for loss, n in losses) / len(y_fit),
                        "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best:
            best, stale, best_epoch = val_loss, 0, epoch
            torch.save(model.state_dict(), model_dir / "best_model.pth")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=DEVICE, weights_only=True))
    save_csv(model_dir / "history.csv", history, ["epoch", "train_loss", "val_loss", "lr"])
    return model, best_epoch, history


def predictions_file(path, ids, y, predictions, probabilities, fold=None):
    rows = [{"video_id": ident, "true_label": int(label), "probability_valid": float(prob),
             "predicted_label": int(pred), "fold": fold if fold is not None else "test"}
            for ident, label, pred, prob in zip(ids, y, predictions, probabilities)]
    save_csv(path, rows, ["video_id", "true_label", "probability_valid", "predicted_label", "fold"])
    return rows


def config(view, run_id=DEFAULT_RUN_ID, subset="all"):
    cfg = view_config(view)
    subset_name, indices = resolve_subset(view, subset)
    eff_points = [cfg["points"][i] for i in indices] if indices is not None else list(cfg["points"])
    p = paths(view, run_id)
    p["models"].mkdir(parents=True, exist_ok=True)
    save_json(p["models"] / "config.json", {
        "exercise": EXERCISE, "view": view, "experiment": f"{EXERCISE}_{view}", "run_id": run_id,
        "source": str(p["source"]), "labels": str(p["labels"]),
        "seed": SEED, "fold_seeds": [SEED + i for i in range(1, 6)],
        "schema": list(enumerate(cfg["points"])), "channels": ["xyn_x", "xyn_y", "yolo_confidence"],
        "subset": subset_name, "subset_indices": list(indices) if indices is not None else None,
        "subset_points": eff_points,
        "input_shape": [FRAMES, len(eff_points), 3], "positive_class": "1=valido", "threshold": 0.5,
        "quality": {"min_frames": MIN_FRAMES, "max_gap": MAX_GAP, "max_missing_xy": MAX_MISSING,
                    "boundary_gaps": "exclude", "missing_confidence": "zero_for_missing_frames"},
        "normalization": ("median shoulder-midpoint center + median intershoulder scale (front)"
                          if view == "front" else
                          "median hip center + median knee-ankle (shank) scale (left)"),
        "temporal": "linear 170 frames over complete original sequence",
        "split": "70/15/15 stratified per view; video_id exclusive",
        "scaler": "StandardScaler on flattened (170*K*3) features, fit original train only",
        "augmentation": "train x4; translation 0.05, scale 0.9-1.1, noise 0.01, temporal shift 3; no rotation/flip; geometry only x,y",
        "architecture": f"training.common.train_utils.PoseCNN(T=170,K={len(eff_points)},C=3)",
        "training": {"batch": BATCH, "epochs": EPOCHS, "early_patience": PATIENCE,
                     "adam_lr": 0.001, "weight_decay": 1e-4, "lr_factor": 0.5, "lr_patience": 4},
        "cv": "5 StratifiedKFold development only; inner stratified validation 15% of outer train",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    })


def run_cv(view, run_id=DEFAULT_RUN_ID, subset="all"):
    subset_name, indices = resolve_subset(view, subset)
    p = paths(view, run_id)
    p["models"].mkdir(parents=True, exist_ok=True)
    p["results"].mkdir(parents=True, exist_ok=True)
    data = np.load(p["data"] / "split.npz")
    X = apply_subset(np.concatenate([data["X_train"], data["X_validation"]]), indices)
    y = np.concatenate([data["y_train"], data["y_validation"]])
    ids = np.concatenate([data["ids_train"], data["ids_validation"]])
    if min(Counter(y).values()) < 5:
        raise ValueError("CV requiere >=5 ejemplos por clase en desarrollo")
    results, predictions = [], []
    for fold, (outer_train, outer_val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), 1):
        seed = SEED + fold
        fit_idx, early_idx = train_test_split(outer_train, test_size=0.15, stratify=y[outer_train], random_state=seed)
        folder = p["models"] / "cv" / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        save_csv(folder / "membership.csv", [{"video_id": ids[i], "label": int(y[i]), "role": role}
                                                   for role, indices in (("fit", fit_idx), ("early", early_idx), ("outer_validation", outer_val))
                                                   for i in indices], ["video_id", "label", "role"])
        X_fit, y_fit, X_early, scaler = scaled_arrays(X[fit_idx], y[fit_idx], X[early_idx], seed, folder / "scaler.pkl")
        model, best_epoch, _ = fit(X_fit, y_fit, X_early, y[early_idx], seed, folder)
        metrics, pred, prob = evaluate(model, transform(scaler, X[outer_val]), y[outer_val], seed)
        results.append({"fold": fold, "seed": seed, "best_epoch": best_epoch,
                        "n_fit": len(fit_idx), "n_early": len(early_idx), "n_outer_validation": len(outer_val), **metrics})
        predictions.extend({"fold": fold, "video_id": ids[i], "true_label": int(y[i]),
                            "predicted_label": int(pred[j]), "probability_valid": float(prob[j])}
                           for j, i in enumerate(outer_val))
        print(f"{view} fold {fold}: {metrics}")
    assert set(row["video_id"] for row in predictions) == set(ids) and len(predictions) == len(ids)
    save_csv(p["results"] / "cv_metrics.csv", results, list(results[0]))
    save_csv(p["results"] / "cv_predictions.csv", predictions, list(predictions[0]))


def run_train(view, run_id=DEFAULT_RUN_ID, subset="all"):
    subset_name, indices = resolve_subset(view, subset)
    p = paths(view, run_id)
    p["models"].mkdir(parents=True, exist_ok=True)
    p["results"].mkdir(parents=True, exist_ok=True)
    if (p["results"] / "test_metrics.json").exists():
        raise RuntimeError("Test ya evaluado para esta vista y ejecucion. Conservar holdout; usar otro --run-id para repetir.")
    data = np.load(p["data"] / "split.npz")
    X_train = apply_subset(data["X_train"], indices)
    X_validation = apply_subset(data["X_validation"], indices)
    X_test = apply_subset(data["X_test"], indices)
    X_fit, y_fit, X_val, scaler = scaled_arrays(X_train, data["y_train"], X_validation,
                                                  SEED, p["models"] / "scaler.pkl")
    model, best_epoch, history = fit(X_fit, y_fit, X_val, data["y_validation"], SEED, p["models"])
    val_metrics, _, _ = evaluate(model, X_val, data["y_validation"], SEED)
    # Test una sola vez, tras seleccionar checkpoint por validation.
    metrics, predicted, probs = evaluate(model, transform(scaler, X_test), data["y_test"], SEED)
    predictions_file(p["results"] / "test_predictions.csv", data["ids_test"], data["y_test"], predicted, probs)
    save_json(p["results"] / "test_metrics.json", {"best_epoch": best_epoch, "validation": val_metrics, "test": metrics})
    np.save(p["results"] / "confusion_matrix.npy", np.asarray(metrics["confusion_matrix"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("train_loss", "val_loss"):
        axes[0].plot([r["epoch"] for r in history], [r[name] for r in history], label=name)
    axes[0].legend()
    axes[0].set(xlabel="Epoca", ylabel="Loss")
    axes[1].imshow(metrics["confusion_matrix"], cmap="Blues")
    axes[1].set(xlabel="Predicho (0 invalido, 1 valido)", ylabel="Real (0 invalido, 1 valido)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(metrics["confusion_matrix"][i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(p["results"] / "curves_and_confusion.png")
    plt.close(fig)
    print(f"{view} test: {metrics}")


def report(view, run_id=DEFAULT_RUN_ID):
    p = paths(view, run_id)
    p["results"].mkdir(parents=True, exist_ok=True)
    summary = json.loads((p["data"] / "audit_summary.json").read_text(encoding="utf-8"))
    results = json.loads((p["results"] / "test_metrics.json").read_text(encoding="utf-8"))
    with (p["results"] / "cv_metrics.csv").open(encoding="utf-8") as handle:
        folds = list(csv.DictReader(handle))
    names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    lines = [f"# SQ {view}: PoseCNN", "", "Clase positiva: 1 = válido; umbral 0.5. Matriz: filas real, columnas predicho [0, 1].", "",
             f"Fuente: `{p['source']}` ({summary['pose_files']} archivos); etiquetas: `{p['labels']}` ({summary['csv_rows']} filas).",
             f"Calidad: {summary['quality_counts']}; exclusiones: {summary['exclusion_reasons']}.",
             f"Frames originales: {summary['original_frames']}. T=170 es baseline por comparabilidad; comparar posteriormente solo en desarrollo.", "",
             "| Métrica | Test | CV media ± desviación |", "|---|---:|---:|"]
    for name in names:
        numbers = np.asarray([float(row[name]) for row in folds])
        lines.append(f"| {name} | {results['test'][name]:.4f} | {numbers.mean():.4f} ± {numbers.std():.4f} |")
    lines += ["", f"Matriz test: `{results['test']['confusion_matrix']}` (TN, FP / FN, TP).",
              "", "Consultar audit.csv, preprocessing.csv, split.csv, cv_metrics.csv y predicciones por video para trazabilidad."]
    (p["results"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", required=True, choices=VIEWS)
    parser.add_argument("--stage", default="audit", choices=("audit", "prepare", "cv", "train", "report", "all"))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--subset", default="all", help="Subset de keypoints (solo front: legs)")
    args = parser.parse_args()
    p = paths(args.view, args.run_id)
    if args.stage == "audit":
        audit(args.view)
        return
    for name in ("data", "models", "results"):
        p[name].mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(args.view)
        config(args.view, args.run_id, args.subset)
    if args.stage in ("cv", "all"):
        run_cv(args.view, args.run_id, args.subset)
    if args.stage in ("train", "all"):
        run_train(args.view, args.run_id, args.subset)
    if args.stage in ("report", "all"):
        report(args.view, args.run_id)


if __name__ == "__main__":
    main()
