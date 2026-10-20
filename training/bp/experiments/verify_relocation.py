"""Verifica solo los bytes de los checkpoints BP front congelados tras moverlos.

El protocolo histórico puede no ser ejecutable en la nueva ruta: verificar
bytes no equivale a repetir la evaluación ni a certificar el código nuevo.
"""

import hashlib
import json
from pathlib import Path

from training.legacy.number_bp_experiments import RENAME


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = HERE / "bp_front_05_final_test_v1" / "bp_front_final_test_protocol_frozen.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check():
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    for name in ("pose_checkpoint", "seg_checkpoint", "pose_scaler", "seg_channels_scaler"):
        info = protocol["sha256_artifacts"][name]
        antiguo = Path(info["path"])
        if antiguo.parts[:1] != ("training",) or len(antiguo.parts) < 3:
            raise ValueError(f"Ruta histórica inesperada: {antiguo}")
        nuevo = HERE / RENAME.get(antiguo.parts[1], antiguo.parts[1]) / Path(*antiguo.parts[2:])
        if not nuevo.is_file() or sha256(nuevo) != info["sha256"]:
            raise RuntimeError(f"Bytes distintos o ausentes: {name}: {nuevo}")
        print(f"{name}: SHA-256 idéntico al protocolo histórico")


if __name__ == "__main__":
    check()
