"""Preprocesamiento de una secuencia de keypoints, idéntico al del entrenamiento de cada red.

No se reimplementa nada: se llama al código de `training/` con el que se prepararon los datos de
cada receta. Cada función recibe la secuencia cruda (T, K, 3), tal como la entrega el extractor, y
devuelve (170, K', 3) float32 listo para el scaler, o lanza InsufficientData si la secuencia no
cumple los requisitos de calidad con que se entrenó la red.

    bp_front       training/bp/experiments/bp_front_02_posecnn_common_v1/data.py
    bp_lateral     training/bp/train_posecnn.py
    sq_front_legs  training/sq/train_posecnn.py (preprocess_front) + subconjunto de piernas
    sq_lateral     training/sq/train_posecnn.py (preprocess_left)
    dl_front/side  training/dl/01_preprocessing.py + 02_temporal_normalization.py
"""

import importlib.util
from functools import lru_cache

import numpy as np

from judge.config import ROOT
from judge.evaluation.base import InsufficientData
from training.bp import train_posecnn as bp
from training.bp.experiments.bp_front_02_posecnn_common_v1 import data as bp_front_data
from training.sq import train_posecnn as sq

TARGET_FRAMES = 170
MIN_FRAMES = 10


@lru_cache(maxsize=None)
def _dl_script(filename):
    """Importa training/dl/<filename> (empiezan con dígitos, no son importables como módulo)."""
    path = ROOT / "training" / "dl" / filename
    spec = importlib.util.spec_from_file_location(f"judge_dl_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_shape(raw, num_points):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (num_points, 3):
        raise InsufficientData(f"forma {raw.shape}, se esperaba (T, {num_points}, 3)")
    if len(raw) < MIN_FRAMES:
        raise InsufficientData(f"{len(raw)} frames, se requieren al menos {MIN_FRAMES}")
    if np.isinf(raw).any():
        raise InsufficientData("la secuencia contiene valores infinitos")
    return raw


def _missing_ratio_gate(raw, max_missing):
    """Mismo criterio de calidad de las auditorías de BP lateral y SQ (`inspect_file`)."""
    xy = raw[..., :2]
    missing = ~np.isfinite(xy).all(axis=2) | ((xy == 0).all(axis=2) & (raw[..., 2] == 0))
    ratio = float(missing.mean())
    if ratio > max_missing:
        raise InsufficientData(f"{ratio:.0%} de los puntos sin detectar (máximo {max_missing:.0%})")


def _guarded(function, *args):
    try:
        return function(*args)
    except ValueError as error:
        if isinstance(error, InsufficientData):
            raise
        raise InsufficientData(str(error)) from error


def bp_front(raw, num_points):
    raw = _check_shape(raw, num_points)
    sequence = _guarded(bp_front_data.secuencia_a_170, raw)[0]
    return sequence.astype(np.float32)


def bp_lateral(raw, num_points):
    raw = _check_shape(raw, num_points)
    _missing_ratio_gate(raw, bp.MAX_MISSING)
    return _guarded(bp.preprocess, raw)[0].astype(np.float32)


def sq_front_legs(raw, num_points, subset_indices):
    raw = _check_shape(raw, num_points)
    _missing_ratio_gate(raw, sq.MAX_MISSING)
    sequence = _guarded(sq.preprocess_front, raw)[0]
    return sequence[:, list(subset_indices), :].astype(np.float32)


def sq_lateral(raw, num_points):
    raw = _check_shape(raw, num_points)
    _missing_ratio_gate(raw, sq.MAX_MISSING)
    return _guarded(sq.preprocess_left, raw)[0].astype(np.float32)


def _dl(raw, num_points, normalize):
    prep, temporal = _dl_script("01_preprocessing.py"), _dl_script("02_temporal_normalization.py")
    raw = _check_shape(raw, num_points)
    keypoints = prep.validate_and_clean(raw, num_points)
    if keypoints is None:
        raise InsufficientData("secuencia sin detecciones válidas")
    keypoints = prep.interpolate_small_gaps(keypoints)
    remaining = np.isnan(keypoints[:, :, :2]).any(axis=(1, 2))
    if remaining.sum() > keypoints.shape[0] * 0.3:
        raise InsufficientData(f"demasiados frames sin detección ({remaining.sum()}/{keypoints.shape[0]})")
    normalized = temporal.normalize_temporal(normalize(prep, keypoints), TARGET_FRAMES)
    if np.isnan(normalized).any():
        raise InsufficientData("quedaron valores sin detectar tras la normalización temporal")
    return normalized.astype(np.float32)


def dl_front(raw, num_points):
    return _dl(raw, num_points, lambda prep, kp: prep.normalize_spatial_front(kp))


def dl_side(raw, num_points, source_view):
    """`source_view` ('left' o 'right'): la vista izquierda se espeja en x, como en el entrenamiento."""
    return _dl(raw, num_points, lambda prep, kp: prep.normalize_spatial_side(kp, source_view))


def preprocess(evaluator, view, skeleton, raw):
    """Secuencia (170, K', 3) para la red de `evaluator` en la vista `view`."""
    recipe, num_points = evaluator.recipe, skeleton.num_points
    if recipe == "bp_front":
        return bp_front(raw, num_points)
    if recipe == "bp_lateral":
        return bp_lateral(raw, num_points)
    if recipe == "sq_front_legs":
        return sq_front_legs(raw, num_points, evaluator.subset_indices)
    if recipe == "sq_lateral":
        return sq_lateral(raw, num_points)
    if recipe == "dl_front":
        return dl_front(raw, num_points)
    if recipe == "dl_side":
        return dl_side(raw, num_points, view)
    raise ValueError(f"Receta desconocida: {recipe}")
