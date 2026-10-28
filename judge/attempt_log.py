"""Registro de cada intento evaluado, para poder auditarlo después.

Cada intento se guarda en su propia carpeta `<log_directory>/<fecha>_<ejercicio>_<n>/` con:
    decision.json        decisión final, veredicto y probabilidad de cada vista, motivos
    keypoints_<vista>.npy  secuencia cruda de keypoints de cada vista (permite reevaluar)
    config.yaml          copia de la configuración usada
    last_frames.jpg      último mosaico anotado, si se entrega
Nunca se sobrescribe un intento anterior.
"""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from judge.config import ROOT, VIEWS


def relative_to_root(path):
    """Ruta relativa a la raíz del repositorio si está dentro de ella; si no, absoluta."""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Attempt:
    exercise: str
    decision: object            # FinalDecision
    recording: object           # Recording
    path: Path = None           # carpeta del registro, None si no se registra
    errors: int = 0             # frames cuya extracción falló durante la grabación


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AttemptLog:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.config = config
        self._hashes = {}

    def _hash(self, path):
        key = str(path)
        if key not in self._hashes:
            self._hashes[key] = sha256(path) if Path(path).is_file() else None
        return self._hashes[key]

    def _new_folder(self, exercise):
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for number in range(1, 1000):
            folder = self.directory / f"{stamp}_{exercise}_{number:03d}"
            try:
                folder.mkdir()
                return folder
            except FileExistsError:
                continue
        raise RuntimeError("No se pudo crear la carpeta del intento")

    def _networks(self):
        networks = {}
        for view, item in self.config.views().items():
            evaluator = item.evaluator
            networks[view] = {
                "name": evaluator.name, "recipe": evaluator.recipe, "verified": evaluator.verified,
                "checkpoint": relative_to_root(evaluator.checkpoint),
                "checkpoint_sha256": self._hash(evaluator.checkpoint),
                "scaler_sha256": self._hash(evaluator.scaler),
                "skeleton_model": item.skeleton.model.name,
            }
        return networks

    def save(self, exercise, decision, recording, errors=0, mosaic=None):
        folder = self._new_folder(exercise)
        payload = {
            "exercise": exercise,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "decision": decision.decision,
            "display": decision.display,
            "explanation": decision.explanation,
            "needs_review": decision.needs_review,
            "votes": {"valid": decision.valid_votes, "invalid": decision.invalid_votes,
                      "insufficient": list(decision.insufficient_views)},
            "views": {view: {"label": v.label, "probability": v.probability, "reason": v.reason,
                             "frames": v.frames} for view, v in decision.verdicts.items()},
            "recording": {"duration_s": recording.duration_s, "frames": recording.frames,
                          "truncated": recording.truncated, "extraction_errors": errors},
            "threshold": self.config.threshold,
            "allow_unverified": self.config.allow_unverified,
            "networks": self._networks(),
        }
        (folder / "decision.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for view in VIEWS:
            np.save(folder / f"keypoints_{view}.npy", recording.sequences[view])
        if self.config.source_path and Path(self.config.source_path).is_file():
            shutil.copy2(self.config.source_path, folder / "config.yaml")
        if mosaic is not None:
            cv2.imwrite(str(folder / "last_frames.jpg"), mosaic)
        return folder
