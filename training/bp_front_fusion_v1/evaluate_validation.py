"""FASE E: comparacion de los 7 modelos en los mismos 99 Validation.

Referencias congeladas + fusiones. Bootstrap pareado descriptivo por video.
No abre Test. No reentrena nada.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from late_fusion import metrica

FUSION_DIR = Path(__file__).resolve().parent
RAIZ = FUSION_DIR.parents[1]
RNG = np.random.default_rng(7)


def _csv(ruta, col="probability_valid"):
    with open(ruta, encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    if len(filas) != 99:
        raise ValueError(f"Predicciones inesperadas: {ruta}")
    return ([f["video_id"] for f in filas],
            np.asarray([int(f["label"]) for f in filas], dtype=np.int64),
            np.asarray([float(f[col]) for f in filas], dtype=np.float64))


def _bce(p, y):
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())


def _bootstrap_par(y, pa, pb, n=2000):
    idx = np.arange(len(y))
    ds = []
    for _ in range(n):
        s = RNG.integers(0, len(y), len(y))
        acc_a = ((pa[s] >= 0.5).astype(int) == y[s]).mean()
        acc_b = ((pb[s] >= 0.5).astype(int) == y[s]).mean()
        ds.append(acc_a - acc_b)
    ds = np.asarray(ds)
    return float(ds.mean()), [float(np.percentile(ds, 2.5)),
                              float(np.percentile(ds, 97.5))]


def main():
    with open(FUSION_DIR / "fusion_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    ids_pose, y, p_pose = _csv(
        RAIZ / "training/bp_front_posecnn_common_v1/seed_42/validation_predictions.csv")
    _, _, p_full = _csv(
        RAIZ / "training/bp_front_segcnn_v1/full/seed_1337/validation_predictions.csv")
    _, _, p_sin = _csv(
        RAIZ / "training/bp_front_segcnn_v1/sin_relacion/seed_42/validation_predictions.csv")
    modelos = {"pose": p_pose, "seg_full": p_full, "seg_sin": p_sin}
    for nombre, carpeta, archivo in (
            ("late_full", "late_fusion_full", "validation_predictions.csv"),
            ("late_sin", "late_fusion_sin_relacion", "validation_predictions.csv")):
        ids, yy, pp = _csv(FUSION_DIR / carpeta / archivo, col="p_fusion")
        if ids != ids_pose or not np.array_equal(yy, y):
            raise ValueError(f"IDs tardia {nombre} no emparejan")
        modelos[nombre] = pp
    for nombre, carpeta in (("emb_full", "embedding_fusion_full"),
                            ("emb_sin", "embedding_fusion_sin_relacion")):
        with open(FUSION_DIR / carpeta / "selected_checkpoint.json",
                  encoding="utf-8") as f:
            sel = json.load(f)
        ids, yy, pp = _csv(FUSION_DIR / carpeta / f"seed_{sel['seed']}"
                           / "validation_predictions.csv")
        if ids != ids_pose or not np.array_equal(yy, y):
            raise ValueError(f"IDs embeddings {nombre} no emparejan")
        modelos[nombre] = pp
    if set().union(*[set(np.flatnonzero((p >= 0.5).astype(int) != y).tolist())
                     for p in modelos.values()]) is None:
        raise ValueError("inaccesible")
    filas, errores = [], {}
    for nombre, p in modelos.items():
        m = metrica(y, p)
        m["validation_bce_ref"] = _bce(p, y)
        filas.append({"modelo": nombre, **{k: m[k] for k in (
            "accuracy", "precision", "recall", "f1", "macro_f1",
            "balanced_accuracy", "roc_auc", "pr_auc", "sensibilidad_valido",
            "especificidad_invalido", "tn", "fp", "fn", "tp",
            "validation_bce_ref")}})
        errores[nombre] = set(np.flatnonzero(
            (p >= 0.5).astype(int) != y).tolist())
    with open(FUSION_DIR / "fusion_experiments_comparison.csv", "w",
              encoding="utf-8", newline="") as archivo:
        w = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open(FUSION_DIR / "fusion_validation_predictions.csv", "w",
              encoding="utf-8", newline="") as archivo:
        w = csv.DictWriter(archivo, fieldnames=["video_id", "label"] + [
            f"p_{n}" for n in modelos])
        w.writeheader()
        for i, vid in enumerate(ids_pose):
            fila = {"video_id": vid, "label": int(y[i])}
            fila.update({f"p_{n}": float(modelos[n][i]) for n in modelos})
            w.writerow(fila)
    nombres = list(modelos)
    with open(FUSION_DIR / "fusion_paired_error_analysis.csv", "w",
              encoding="utf-8", newline="") as archivo:
        w = csv.DictWriter(archivo, fieldnames=["video_id", "label"] + [
            f"err_{n}" for n in nombres])
        w.writeheader()
        for i, vid in enumerate(ids_pose):
            fila = {"video_id": vid, "label": int(y[i])}
            fila.update({f"err_{n}": int(i in errores[n]) for n in nombres})
            w.writerow(fila)
    pares = {}
    for a in ("late_full", "late_sin", "emb_full", "emb_sin"):
        pares[a] = {}
        for b in ("pose", "seg_full", "seg_sin"):
            ea, eb = errores[a], errores[b]
            media, ici = _bootstrap_par(
                y, modelos[a], modelos[b])
            pares[a][f"vs_{b}"] = {
                "errores_a": len(ea), "errores_b": len(eb),
                "compartidos": len(ea & eb),
                "solo_a": sorted(ids_pose[i] for i in ea - eb),
                "solo_b": sorted(ids_pose[i] for i in eb - ea),
                "corr_probs": float(np.corrcoef(modelos[a], modelos[b])[0, 1]),
                "bootstrap_diff_acc_media": media,
                "bootstrap_diff_acc_ic95": ici,
            }
    with open(FUSION_DIR / "fusion_pairwise_analysis.json", "w",
              encoding="utf-8") as archivo:
        json.dump(pares, archivo, indent=2)
    # Seleccion prefijada: Macro-F1, desempate especificidad, luego BalAcc.
    orden = sorted(filas, key=lambda r: (
        r["macro_f1"], r["especificidad_invalido"],
        r["balanced_accuracy"]), reverse=True)
    with open(FUSION_DIR / "fusion_selected_candidate.json", "w",
              encoding="utf-8") as archivo:
        json.dump({"criterio": cfg["seleccion_candidato"],
                   "ranking_macro_f1": [(r["modelo"], r["macro_f1"],
                                         r["especificidad_invalido"],
                                         r["balanced_accuracy"]) for r in orden],
                   "candidato": orden[0]["modelo"],
                   "nota": "Seleccion descriptiva en Validation reutilizado; "
                           "no es evaluacion externa"}, archivo, indent=2)
    print(json.dumps([orden[0]] + [
        {k: r[k] for k in ("modelo", "accuracy", "f1", "macro_f1",
                           "balanced_accuracy", "roc_auc", "fp", "fn")}
        for r in filas], indent=2))


if __name__ == "__main__":
    main()
