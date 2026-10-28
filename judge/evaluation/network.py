"""Carga y ejecución de la red PoseCNN de una vista, con su scaler."""

import pickle

import numpy as np
import torch

from judge.evaluation import recipes
from judge.evaluation.base import (INSUFFICIENT, INVALID, VALID, EvaluationError, InsufficientData,
                                   ViewVerdict)
from training.bp.experiments.bp_front_02_posecnn_common_v1.model import BPFrontPoseCNN
from training.common.train_utils import PoseCNN
from training.sq.controlled_front import PointXYScaler, transform as scale_sequences  # noqa: F401  (PointXYScaler: necesario para deserializar)

ARCHITECTURES = ("PoseCNN", "BPFrontPoseCNN")


def build_model(arch, frames, num_points):
    if arch == "PoseCNN":
        return PoseCNN(T=frames, K=num_points, C=3)
    if arch == "BPFrontPoseCNN":
        if (frames, num_points) != (170, 6):
            raise EvaluationError(f"BPFrontPoseCNN espera (170, 6), recibido ({frames}, {num_points})")
        return BPFrontPoseCNN()
    raise EvaluationError(f"Arquitectura desconocida: {arch}")


def load_scaler(path, scaler_type, frames, num_points):
    with path.open("rb") as handle:
        scaler = pickle.load(handle)
    if scaler_type == "standard":
        expected = frames * num_points * 3
        if getattr(scaler, "n_features_in_", None) != expected:
            raise EvaluationError(f"{path.name}: el scaler tiene {getattr(scaler, 'n_features_in_', None)} "
                                  f"características y la red necesita {expected} ({frames}x{num_points}x3)")
    elif scaler_type == "point_xy":
        if not isinstance(scaler, PointXYScaler) or scaler.mean_.shape != (num_points, 2):
            raise EvaluationError(f"{path.name}: scaler point_xy incompatible con {num_points} puntos")
    else:
        raise EvaluationError(f"Tipo de scaler desconocido: {scaler_type}")
    return scaler


class ViewEvaluator:
    """Red de una vista: preprocesa la secuencia cruda, la escala y devuelve el veredicto."""

    def __init__(self, evaluator, view, skeleton, threshold=0.5, frames=recipes.TARGET_FRAMES, device="cpu",
                 allow_unverified=False):
        self.evaluator, self.view, self.skeleton = evaluator, view, skeleton
        self.threshold, self.frames, self.device = threshold, frames, device
        self.allow_unverified = allow_unverified
        self.num_points = (len(evaluator.subset_indices) if evaluator.subset_indices is not None
                           else skeleton.num_points)
        self.scaler = load_scaler(evaluator.scaler, evaluator.scaler_type, frames, self.num_points)
        self.model = build_model(evaluator.arch, frames, self.num_points)
        state = torch.load(evaluator.checkpoint, map_location=device, weights_only=True)
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise EvaluationError(f"{evaluator.checkpoint.name}: el checkpoint no corresponde a "
                                  f"{evaluator.arch}(K={self.num_points}): {error}") from error
        self.model.to(device).eval()

    def prepare(self, raw):
        """Secuencia cruda (T, K, 3) -> (170, K', 3) escalada. Lanza InsufficientData."""
        sequence = recipes.preprocess(self.evaluator, self.view, self.skeleton, raw)
        return scale_sequences(self.scaler, sequence[None])[0]

    def probability(self, raw):
        x = torch.from_numpy(np.ascontiguousarray(self.prepare(raw)[None])).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            return float(self.model(x).reshape(-1)[0].cpu())

    @property
    def usable(self):
        """False si la red no reproduce sus resultados de entrenamiento y no se permitió usarla."""
        return self.evaluator.verified or self.allow_unverified

    def evaluate(self, raw):
        frames = 0 if raw is None else len(raw)
        if not self.usable:
            reason = f"red no verificada: {self.evaluator.verification_note}"
            return ViewVerdict(self.view, INSUFFICIENT, None, reason, frames)
        try:
            probability = self.probability(raw)
        except InsufficientData as error:
            return ViewVerdict(self.view, INSUFFICIENT, None, str(error), frames)
        label = VALID if probability >= self.threshold else INVALID
        return ViewVerdict(self.view, label, probability, None, frames)
