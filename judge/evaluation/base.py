"""Tipos del módulo de evaluación."""

from dataclasses import dataclass, field

import numpy as np

VALID = "valid"
INVALID = "invalid"
INSUFFICIENT = "insufficient"       # no hay datos suficientes para que la red decida


class EvaluationError(RuntimeError):
    """Red, scaler o checkpoint incompatibles con la configuración."""


class InsufficientData(ValueError):
    """La secuencia no cumple los requisitos de calidad con que se entrenó la red."""


@dataclass(frozen=True)
class ViewVerdict:
    view: str
    label: str                      # valid | invalid | insufficient
    probability: float = None       # probabilidad de VÁLIDO; None si no hubo datos suficientes
    reason: str = None              # motivo cuando label == insufficient
    frames: int = 0                 # frames crudos que se evaluaron

    @property
    def decided(self):
        return self.label in (VALID, INVALID)


@dataclass(frozen=True)
class Recording:
    """Keypoints de un levantamiento, una secuencia por vista: (T, K, 3), T puede diferir entre vistas."""
    sequences: dict
    duration_s: float
    truncated: bool = False
    frames: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "frames", {v: len(s) for v, s in self.sequences.items()})
        for view, sequence in self.sequences.items():
            if sequence.ndim != 3 or sequence.shape[2] != 3 or sequence.dtype != np.float32:
                raise ValueError(f"{view}: se esperaba (T, K, 3) float32, recibido {sequence.shape} {sequence.dtype}")
