"""Keypoints YOLO BP_FRONT en el split común. Test: solo IDs del CSV.

La única desviación de la receta histórica corresponde a bp_406:
9 frames seguidos sin pose en los .npy YOLO originales. No interpolar
ese hueco; normalizar con frames observados y codificar ausencia con
coordenadas neutras y confianza cero, más máscara de procedencia separada.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[4]
COHORTE = RAIZ / "segmentation/bp/front/temporal_features_v1/bp_front_features_common_cohort.csv"
POSE_DIR = RAIZ / "keypoints/bp/yolo/front"
T_OBJETIVO = 170


def leer_cohorte():
    with open(COHORTE, newline="", encoding="utf-8-sig") as arch:
        filas = list(csv.DictReader(arch))
    if len(filas) != 663 or len({r["video_id"] for r in filas}) != 663:
        raise ValueError("Cohorte incompleta/IDs duplicados")
    if {r["video_id"] for r in filas if r["excluido"] == "True"} != {"bp_349", "bp_352"}:
        raise ValueError("Exclusiones alteradas")
    grupos = {s: [r for r in filas if r["common_split"] == s and r["excluido"] == "False"]
              for s in ("train", "validation", "test")}
    if [len(grupos[s]) for s in ("train", "validation", "test")] != [462, 99, 100]:
        raise ValueError("Split común alterado")
    if [sum(r["label"] == "1" for r in grupos[s]) for s in ("train", "validation")] != [279, 60]:
        raise ValueError("Labels desarrollo alteradas")
    return grupos


def secuencia_a_170(pose):
    """Centro/escala de hombros por video, mismos instantes que SegCNN.

    Solo se interpola entre frames originales adyacentes ambos observados;
    así no se cruzan huecos de persona ni se inventan landmarks.
    """
    if pose.ndim != 3 or pose.shape[1:] != (6, 3) or len(pose) < 10:
        raise ValueError(f"Shape pose inválido: {pose.shape}")
    if np.isinf(pose).any():
        raise ValueError("Pose contiene Inf")
    valid = np.isfinite(pose).all(axis=2)
    # En datos normales no hay valores perdidos. En bp_406 la ausencia es
    # de frame completo; si apareciera una ausencia parcial, no interpolarla.
    hombros = valid[:, 1] & valid[:, 2]
    if not hombros.any():
        raise ValueError("Hombros ausentes en todo el video")
    xy = pose[..., :2]
    punto_medio = xy[hombros][:, [1, 2]].mean(axis=1)
    centro = np.median(punto_medio, axis=0)
    distancias = np.linalg.norm(xy[hombros][:, 1] - xy[hombros][:, 2], axis=1)
    escala = float(np.median(distancias))
    if not np.isfinite(escala) or escala <= 1e-6:
        raise ValueError("Distancia interhombros inválida")
    norm = pose.copy().astype(np.float32)
    norm[..., :2] = (xy - centro) / escala
    grid = np.linspace(0, len(pose) - 1, T_OBJETIVO, dtype=np.float64)
    salida = np.zeros((T_OBJETIVO, 6, 3), dtype=np.float32)
    disponibilidad = np.zeros((T_OBJETIVO, 6), dtype=np.uint8)
    for j, t in enumerate(grid):
        izq, der = int(np.floor(t)), int(np.ceil(t))
        if izq == der:
            ok = valid[izq]
            salida[j, ok] = norm[izq, ok]
        else:
            ok = valid[izq] & valid[der]
            salida[j, ok] = ((1 - (t - izq)) * norm[izq, ok]
                             + (t - izq) * norm[der, ok])
        disponibilidad[j] = ok
    # (x,y,conf)=(0,0,0) solo significa keypoint no disponible. El sidecar
    # mask desambigua los valores y evita tratarlos como observación física.
    return salida, disponibilidad, centro.astype(np.float32), escala


def cargar_desarrollo(grupos):
    """Cargar solo .npy Train/Validation, no abrir videos/poses de Test."""
    datos = {}
    auditoria = []
    for split in ("train", "validation"):
        xs, masks, ys, ids = [], [], [], []
        for fila in grupos[split]:
            video_id = fila["video_id"]
            archivo = POSE_DIR / f"{video_id}.npy"
            if not archivo.is_file() or archivo.resolve() != Path(fila["pose_path"]).resolve():
                raise ValueError(f"Pose YOLO front inexistente/ruta incorrecta: {video_id}")
            pose = np.load(archivo, allow_pickle=False)
            if len(pose) != int(fila["T"]):
                raise ValueError(f"T incorrecto en {video_id}")
            procesada, mask, centro, escala = secuencia_a_170(pose)
            xs.append(procesada)
            masks.append(mask)
            ys.append(int(fila["label"]))
            ids.append(video_id)
            ausentes = ~np.isfinite(pose).all(axis=2)
            auditoria.append({
                "video_id": video_id, "split": split,
                "T_original": len(pose), "nan_values": int(np.isnan(pose).sum()),
                "missing_original_frames": int(ausentes.any(axis=1).sum()),
                "missing_resampled_keypoints": int((~mask.astype(bool)).sum()),
                "spatial_center_x": float(centro[0]),
                "spatial_center_y": float(centro[1]),
                "spatial_scale_intershoulder": escala,
                "label": int(fila["label"]),
            })
        datos[split] = (np.stack(xs).astype(np.float32), np.stack(masks),
                        np.asarray(ys, dtype=np.int64), ids)
    return datos, auditoria


def augment_sequence(sequence, mask, rng):
    """Mismas cinco transformaciones históricas; ausencia permanece explícita."""
    a = sequence.copy()
    valido = mask.astype(bool).copy()
    if rng.random() > 0.5:
        a[..., 0] += rng.uniform(-0.05, 0.05)
        a[..., 1] += rng.uniform(-0.05, 0.05)
    if rng.random() > 0.5:
        a[..., :2] *= rng.uniform(0.9, 1.1)
    if rng.random() > 0.5:
        noise = rng.normal(0, 0.01, a[..., :2].shape)
        a[..., :2] += noise.astype(np.float32)
    if rng.random() > 0.5:
        angle = np.deg2rad(rng.uniform(-10.0, 10.0))
        cos, sin = np.cos(angle), np.sin(angle)
        x, y = a[..., 0].copy(), a[..., 1].copy()
        a[..., 0] = x * cos - y * sin
        a[..., 1] = x * sin + y * cos
    if rng.random() > 0.5:
        shift = int(rng.integers(-3, 4))
        if shift:
            previo, anterior_mask = a.copy(), valido.copy()
            a = np.roll(a, shift, axis=0)
            valido = np.roll(valido, shift, axis=0)
            if shift > 0:
                a[:shift], valido[:shift] = previo[0], anterior_mask[0]
            else:
                a[shift:], valido[shift:] = previo[-1], anterior_mask[-1]
    a[~valido] = 0.0
    return a.astype(np.float32)


def augment_train(x, mask, y, seed):
    rng = np.random.default_rng(seed)
    xx, yy = [], []
    for serie, disponible, label in zip(x, mask, y):
        xx.append(serie)
        yy.append(int(label))
        for _ in range(3):
            xx.append(augment_sequence(serie, disponible, rng))
            yy.append(int(label))
    return np.stack(xx), np.asarray(yy, dtype=np.int64)


def ajustar_scaler_train(x_train):
    if len(x_train) != 462:
        raise ValueError("El scaler exige 462 levantamientos originales Train")
    return StandardScaler().fit(x_train.reshape(len(x_train), -1))


def transformar(x, scaler):
    return scaler.transform(x.reshape(len(x), -1)).reshape(x.shape).astype(np.float32)


def preprocesar_video_para_inferencia(archivo_pose, media, escala):
    """Reutilizar scaler ajustado y grilla frozen sin acceder a Test por defecto."""
    pose = np.load(archivo_pose, allow_pickle=False)
    procesada, disponibilidad, centro, distancia = secuencia_a_170(pose)
    plano = (procesada.reshape(1, -1) - media) / escala
    return plano.reshape(1, T_OBJETIVO, 6, 3).astype(np.float32), disponibilidad
