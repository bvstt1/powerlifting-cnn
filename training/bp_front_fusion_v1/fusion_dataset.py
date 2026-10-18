"""Dataset de fusion BP_FRONT: solo Train/Validation del split comun.

Reproduce exactamente los preprocesamientos congelados de cada rama:
PoseCNN (scaler Train) y SegCNN (canales/mediana/media/std Train guardados).
Verifica que los IDs y el orden coincidan entre ramas. Nunca abre Test.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from fusion_models import RAIZ, POSE_DIR, SEG_DIR


def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo

COHORTE = (RAIZ / "segmentation/bp/front/temporal_features_v1"
           / "bp_front_features_common_cohort.csv")


def leer_grupos():
    with open(COHORTE, encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if len(filas) != 663 or len({f["video_id"] for f in filas}) != 663:
        raise ValueError("Cohorte alterada")
    grupos = {s: [f for f in filas
                  if f["common_split"] == s and f["excluido"] == "False"]
              for s in ("train", "validation", "test")}
    if [len(grupos[s]) for s in ("train", "validation", "test")] != [462, 99, 100]:
        raise ValueError("Split comun alterado")
    return grupos


def cargar_pose(grupos):
    posedata = _modulo("fusion_posedata", POSE_DIR / "data.py")
    datos, _ = posedata.cargar_desarrollo(grupos)
    with np.load(POSE_DIR / "train_scaler.npz", allow_pickle=False) as sc:
        media = sc["mean"].copy()
        escala = sc["scale"].copy()
    if media.shape != (3060,) or escala.shape != (3060,):
        raise ValueError("Scaler PoseCNN incompatible")
    salida = {}
    for split in ("train", "validation"):
        x, _, y, ids = datos[split]
        if len(x) != len(grupos[split]):
            raise ValueError(f"Conteo Pose {split} alterado")
        plano = posedata.transformar(x, _scaler(media, escala))
        salida[split] = (np.ascontiguousarray(plano.transpose(0, 3, 1, 2)), y, ids)
    return salida


class _scaler:
    def __init__(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, x):
        return (x - self.mean_) / self.scale_


def _canales_guardados(variante):
    sub = "full" if variante == "full" else "sin_relacion"
    with open(SEG_DIR / sub / "channels_and_train_preprocessing.json",
              encoding="utf-8") as archivo:
        return json.load(archivo)


def cargar_seg(grupos, variante):
    segdata = _modulo("fusion_segdata", SEG_DIR / "data.py")
    grupos_seg, nombres = segdata.leer_cohorte()
    if [r["video_id"] for r in grupos_seg["train"]] != [
            r["video_id"] for r in grupos["train"]] or [
            r["video_id"] for r in grupos_seg["validation"]] != [
            r["video_id"] for r in grupos["validation"]]:
        raise ValueError("Orden de IDs SegCNN difiere del split comun")
    datos = segdata.cargar_datos(grupos_seg, nombres)
    guardados = _canales_guardados(variante)
    preparado, canales = segdata.preparar_variante(
        datos, nombres, variante, ["profile_ambiguous"])
    for clave in ("mediana_train", "media_train", "std_train"):
        if not np.allclose(np.asarray(canales[clave], dtype=np.float64),
                           np.asarray(guardados[clave], dtype=np.float64),
                           atol=1e-6):
            raise ValueError(f"Preprocesamiento SegCNN {variante} no reproducible: {clave}")
    if canales["nombres"] != guardados["nombres"] or canales["indices"] != guardados["indices"]:
        raise ValueError(f"Canales SegCNN {variante} alterados")
    return preparado


def cargar_fusion(variante):
    grupos = leer_grupos()
    pose = cargar_pose(grupos)
    seg = cargar_seg(grupos, variante)
    datos = {}
    for split in ("train", "validation"):
        xp, yp, idp = pose[split]
        xs, ys, ids = seg[split]
        if idp != ids or list(yp) != list(ys):
            raise ValueError(f"IDs/labels Pose/Seg no emparejan en {split}")
        datos[split] = {"pose": xp, "seg": xs, "y": ys, "ids": idp}
    return datos
