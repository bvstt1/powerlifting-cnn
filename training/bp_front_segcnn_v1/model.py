"""CNN temporal 1D compacta para clasificación binaria por video BP_FRONT."""
import torch.nn as nn


class SegCNNTemporal(nn.Module):
    def __init__(self, canales):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(canales, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.20),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.40), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.clasificador(self.temporal(x)).squeeze(-1)
