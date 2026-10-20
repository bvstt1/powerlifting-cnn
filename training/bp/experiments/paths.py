"""Directorio vigente de cada ejecución monovista BP numerada."""

from pathlib import Path


BASE = Path(__file__).resolve().parent
NUMBERS = {"segcnn": "01", "pose": "02", "fusion": "03", "cv": "04", "final_test": "05"}


def experiment_dir(view, stage):
    if view not in ("left", "right") or stage not in NUMBERS:
        raise ValueError(f"Vista o etapa desconocida: {view}/{stage}")
    suffix = {"pose": "posecnn_compatibility_v1" if view == "left" else "posecnn_common_v1",
              "cv": "cv_v1", "final_test": "final_test_v1",
              "segcnn": "segcnn_v1", "fusion": "fusion_v1"}[stage]
    return BASE / f"bp_{view}_{NUMBERS[stage]}_{suffix}"
