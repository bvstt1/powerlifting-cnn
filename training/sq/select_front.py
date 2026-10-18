"""Congela selección SQ front usando SOLO CV de desarrollo; nunca lee test.

python -m training.sq.select_front
"""
import json
from pathlib import Path

import numpy as np

from training.sq.controlled_front import REFERENCE, SELECTED, SETUPS, paths, reference
from training.sq.run_analysis import digest, metrics, read_predictions
from training.sq import train_posecnn as sq


def summarize(rows):
    folds = []
    for fold in range(1, 6):
        subset = [r for r in rows if int(r["fold"]) == fold]
        folds.append(metrics([r["true_label"] for r in subset], [r["probability_valid"] for r in subset]))
    return {key: {"mean": float(np.mean([row[key] for row in folds])),
                  "std": float(np.std([row[key] for row in folds]))} for key in
            ("balanced_accuracy", "f1_macro", "roc_auc", "recall_invalid", "f1_valid")}


def main():
    if SELECTED.exists():
        raise FileExistsError(f"Selección congelada ya existe: {SELECTED}")
    manifest, _ = reference()
    results = {"K10_reference": summarize(read_predictions(REFERENCE["results"] / "cv_predictions.csv"))}
    hashes = {}
    for variant in ("A", "B", "C", "D"):
        p = paths(variant)
        file = p["results"] / "cv_predictions.csv"
        if not file.exists():
            raise FileNotFoundError(file)
        results[variant] = summarize(read_predictions(file))
        hashes[variant] = digest(file)
    # Primario: promedio de balanced accuracy y macro F1; desempate AUC.
    ranking = sorted(("K10_reference", "A", "B", "C", "D"),
                     key=lambda name: (results[name]["balanced_accuracy"]["mean"] +
                                       results[name]["f1_macro"]["mean"],
                                       results[name]["roc_auc"]["mean"]), reverse=True)
    selected = ranking[0]
    if selected == "K10_reference":
        print("Ninguna variante mejora K10 según desarrollo; se conserva referencia, NO nueva evaluación test")
        return
    output = {"selected_variant": selected, "ranking_development_only": ranking,
              "selection_rule": "max media(balanced_accuracy + F1_macro); desempate ROC-AUC CV",
              "cv_comparison": results, "cv_prediction_sha256": hashes,
              "reference_split_sha256": manifest["sha256"]["split.npz"],
              "test_policy": "test consultado previamente en K16/K10: evaluación futura exploratoria, no selección",
              "threshold": 0.5, "combo": "no: A/B no superan referencia consistentemente, C inferior; solo D favorable"}
    sq.save_json(SELECTED, output)
    print("Seleccionado usando solo CV:", selected, "ranking", ranking)
    for name in ranking:
        print(name, results[name])


if __name__ == "__main__":
    main()
