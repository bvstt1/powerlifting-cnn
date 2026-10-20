"""Comparación final exclusivamente en Validation, sin leer NPZ de Test."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

SALIDA = Path(__file__).resolve().parent
FUENTE = SALIDA.parents[3] / "segmentation/bp/front/temporal_features_v1"


def main():
    seleccion = {}
    for nombre in ("full", "sin_relacion"):
        m = json.loads((SALIDA / nombre / "selected_checkpoint.json").read_text(encoding="utf-8"))
        seed = m["seed"]
        datos = list(csv.DictReader(open(SALIDA / nombre / f"seed_{seed}" /
                                    "validation_predictions.csv", encoding="utf-8")))
        seleccion[nombre] = {f["video_id"]: f for f in datos}
    if len(seleccion["full"]) != 99 or set(seleccion["full"]) != set(seleccion["sin_relacion"]):
        raise RuntimeError("Validation no corresponde a mismos 99 videos")
    cohorte = {f["video_id"]: f for f in csv.DictReader(
        open(FUENTE / "bp_front_features_common_cohort.csv", encoding="utf-8-sig"))}
    filas = []
    for video_id in sorted(seleccion["full"]):
        a = seleccion["full"][video_id]
        b = seleccion["sin_relacion"][video_id]
        if cohorte[video_id]["common_split"] != "validation" or a["label"] != b["label"]:
            raise RuntimeError(f"Pareja incompatible {video_id}")
        y = int(a["label"])
        pred_a, pred_b = int(a["predicted"]), int(b["predicted"])
        filas.append({"video_id": video_id, "label": y,
                      "quality_seg": cohorte[video_id]["quality_status_seg"],
                      "full_prob": a["probability_valid"], "full_pred": pred_a,
                      "sin_relacion_prob": b["probability_valid"],
                      "sin_relacion_pred": pred_b,
                      "full_error": int(pred_a != y),
                      "sin_relacion_error": int(pred_b != y)})
    with open(SALIDA / "validation_paired_predictions.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)
    distinto = [r for r in filas if r["full_pred"] != r["sin_relacion_pred"]]
    comparacion = {
        "validation_n": 99,
        "full_errores": sum(r["full_error"] for r in filas),
        "sin_relacion_errores": sum(r["sin_relacion_error"] for r in filas),
        "ambos_errores": sum(r["full_error"] and r["sin_relacion_error"] for r in filas),
        "solo_full_error": sum(r["full_error"] and not r["sin_relacion_error"] for r in filas),
        "solo_sin_relacion_error": sum(r["sin_relacion_error"] and not r["full_error"] for r in filas),
        "discrepancias": len(distinto),
        "ids_discrepantes": [r["video_id"] for r in distinto],
        "quality_warning_validation": sum(r["quality_seg"] == "QUALITY_WARNING" for r in filas),
        "separacion_test_intacta": True,
    }
    (SALIDA / "paired_comparison_validation.json").write_text(
        json.dumps(comparacion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparacion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
