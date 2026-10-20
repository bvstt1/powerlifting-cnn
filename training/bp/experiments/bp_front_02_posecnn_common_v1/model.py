"""Replica exacta de la arquitectura BPFrontPoseCNN histórica + acceso al GAP.

No se importan ni cargan pesos del checkpoint histórico.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BPFrontPoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), nn.Dropout2d(0.40),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(0.50), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.50), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def extract_embeddings(self, x):
        """Embedding antes del MLP histórico: (N,128); x=(N,3,170,6)."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        # Históricamente classifier recibe tensor (N,128,1,1); Flatten
        # preserva exactamente los mismos pesos que (N,128).
        return self.classifier(self.pool(self.features(x)))


def cargar_checkpoint(ruta, dispositivo="cpu"):
    """Para inferencia/fusión futura; solo checkpoint del experimento nuevo."""
    ruta = Path(ruta).resolve()
    esperado = Path(__file__).resolve().parent
    if esperado not in ruta.parents:
        raise ValueError("Se admite solo un checkpoint de PoseCNN Common V1")
    contenido = torch.load(ruta, map_location=dispositivo, weights_only=True)
    modelo = BPFrontPoseCNN().to(dispositivo)
    modelo.load_state_dict(contenido["state_dict"], strict=True)
    modelo.eval()
    return modelo, contenido


def cargar_inferencia(ruta_checkpoint, ruta_scaler=None, dispositivo="cpu"):
    """Checkpoint + scaler de Train; no preprocesa ni abre Test por sí mismo.

    Para un .npy del video deseado, llamar primero a
    data.preprocesar_video_para_inferencia(ruta, media, escala),
    después model.extract_embeddings(tensor BCHW) o model(tensor BCHW).
    """
    modelo, meta = cargar_checkpoint(ruta_checkpoint, dispositivo)
    if ruta_scaler is None:
        ruta_scaler = Path(__file__).resolve().parent / "train_scaler.npz"
    with np.load(ruta_scaler, allow_pickle=False) as archivo:
        media = archivo["mean"].copy()
        escala = archivo["scale"].copy()
    if media.shape != escala.shape or media.size != 170 * 6 * 3:
        raise ValueError("Scaler Train incompatible con entrada PoseCNN")
    return modelo, meta, media, escala
