"""Reproducir SOLO Validation desde checkpoint seleccionado y scaler Train."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .data import leer_cohorte, cargar_desarrollo, transformar
from .model import cargar_inferencia
from .train import OUT, metricas, json_file


def main():
    grupos = leer_cohorte()
    # Esta función abre únicamente Train/Validation, no Test.
    datos, _ = cargar_desarrollo(grupos)
    x, _, y, ids = datos["validation"]
    elegido = json.loads((OUT / "selected_checkpoint.json").read_text(encoding="utf-8"))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, meta, media, scale = cargar_inferencia(OUT / elegido["path"],
                                                   dispositivo=dispositivo)
    with np.load(OUT / "train_scaler.npz", allow_pickle=False) as estado:
        scaler = StandardScaler()
        scaler.mean_ = estado["mean"]
        scaler.scale_ = estado["scale"]
        scaler.var_ = estado["var"]
        scaler.n_samples_seen_ = int(estado["n_samples"][0])
        scaler.n_features_in_ = media.size
    inp = torch.from_numpy(transformar(x, scaler).transpose(0, 3, 1, 2).copy())
    with torch.no_grad():
        p = np.concatenate([modelo(z.to(dispositivo)).squeeze(-1).cpu().numpy()
                            for z in inp.split(16)]).astype(np.float64)
    saved = list(csv.DictReader(open(OUT / f"seed_{elegido['seed']}" /
                               "validation_predictions.csv", encoding="utf-8")))
    if len(ids) != 99 or [r["video_id"] for r in saved] != ids:
        raise RuntimeError("IDs Validation no coinciden")
    referencia = np.array([float(r["probability_valid"]) for r in saved])
    diferencia = float(np.max(np.abs(p - referencia)))
    if diferencia > 1e-5:
        raise RuntimeError(f"No reproducible: diferencia máxima {diferencia}")
    m = metricas(y, p)
    json_file(OUT / "validation_recheck.json", {
        "selected_seed": elegido["seed"], "n_validation": len(ids),
        "max_probability_difference": diferencia, "metrics": m,
        "test_npys_opened": 0,
    })
    print("Validation reproducida:", len(ids), "max_delta", diferencia,
          "F1", m["f1"], "Test no abierto")


if __name__ == "__main__":
    main()
