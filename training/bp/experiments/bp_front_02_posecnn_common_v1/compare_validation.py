"""Comparación pareada PoseCNN/SegCNN SOLO Validation; nunca abrir Test."""
import csv
import json
from pathlib import Path

import numpy as np

from .data import COHORTE, RAIZ
from .train import OUT, SEG, metricas, json_file, csv_file


def tabla(path):
    with open(path, encoding="utf-8", newline="") as arch:
        rows = list(csv.DictReader(arch))
    if len(rows) != 99 or len({r["video_id"] for r in rows}) != 99:
        raise RuntimeError(f"Predicciones Validation inesperadas: {path}")
    return {r["video_id"]: r for r in rows}


def main():
    with open(COHORTE, newline="", encoding="utf-8-sig") as arch:
        cohorte = list(csv.DictReader(arch))
    expected = {r["video_id"]: int(r["label"]) for r in cohorte
                if r["common_split"] == "validation" and r["excluido"] == "False"}
    if len(expected) != 99:
        raise RuntimeError("Cohorte Validation cambió")
    psel = json.loads((OUT / "selected_checkpoint.json").read_text(encoding="utf-8"))
    modelos = {"pose": tabla(OUT / f"seed_{psel['seed']}" / "validation_predictions.csv")}
    checkpoints = {"pose": psel["path"]}
    for variante in ("full", "sin_relacion"):
        elegido = json.loads((SEG / variante / "selected_checkpoint.json").read_text(encoding="utf-8"))
        modelos[variante] = tabla(
            SEG / variante / f"seed_{elegido['seed']}" / "validation_predictions.csv")
        checkpoints[variante] = str(SEG / variante / elegido["checkpoint"])
    if any(set(m) != set(expected) for m in modelos.values()):
        raise RuntimeError("IDs de Validation no emparejan")
    rows = []
    probs = {n: [] for n in modelos}
    errors = {n: set() for n in modelos}
    ids = sorted(expected)
    for video_id in ids:
        label = expected[video_id]
        row = {"video_id": video_id, "label": label}
        for nombre, modelo in modelos.items():
            r = modelo[video_id]
            if int(r["label"]) != label:
                raise RuntimeError(f"Label discordante: {video_id}")
            p = float(r["probability_valid"])
            pred = int(r["predicted"])
            if int(p >= 0.5) != pred:
                raise RuntimeError(f"Threshold discordante: {video_id}")
            row[f"{nombre}_prob"] = p
            row[f"{nombre}_pred"] = pred
            row[f"{nombre}_error"] = int(pred != label)
            probs[nombre].append(p)
            if pred != label:
                errors[nombre].add(video_id)
        rows.append(row)
    csv_file(OUT / "paired_validation_predictions.csv", rows)
    y = np.asarray([expected[video_id] for video_id in ids])
    resultados = {
        "n_validation": len(ids), "positive_class": "1=valido",
        "threshold": 0.5, "checkpoints_frozen": checkpoints,
        "metrics": {k: metricas(y, np.asarray(p)) for k, p in probs.items()},
        "comparisons": {}, "shared_all_three": sorted(set.intersection(*errors.values())),
        "at_least_one_correct": len(ids) - len(set.intersection(*errors.values())),
        "test_npys_opened": 0,
    }
    for seg in ("full", "sin_relacion"):
        ep, es = errors["pose"], errors[seg]
        resultados["comparisons"][seg] = {
            "pose_errors": len(ep), "seg_errors": len(es),
            "shared_errors": len(ep & es),
            "shared_error_ids": sorted(ep & es),
            "pose_only_errors": len(ep - es),
            "pose_only_error_ids": sorted(ep - es),
            "seg_only_errors": len(es - ep),
            "seg_only_error_ids": sorted(es - ep),
            "at_least_one_correct": len(ids) - len(ep & es),
            "predictions_disagree": sum(r["pose_pred"] != r[f"{seg}_pred"] for r in rows),
            "pearson_probability_correlation": float(np.corrcoef(probs["pose"], probs[seg])[0, 1]),
        }
    json_file(OUT / "comparison_validation.json", resultados)
    print(json.dumps({"n_validation": resultados["n_validation"],
                      "shared_all_three": resultados["shared_all_three"],
                      "comparisons": resultados["comparisons"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
