"""FASES D–G: análisis de predicciones Test ya escritas, sin abrir fuentes Test.

Los intervalos son bootstrap PAREADO estratificado (39 negativos/61 positivos
si así lo contiene el holdout), 5000 replicaciones, semilla 2026.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from common import OUT, RAIZ, POSE, SEG, FUSION, guardar_csv, guardar_json, metrica, sha256

N = 5000
SEED = 2026
MODELOS = {"pose": "PoseCNN Common", "seg": "SegCNN SIN", "fusion": "Fusión tardía Pose+SIN"}


def leer_csv(path):
    with open(path, encoding="utf-8", newline="") as arch:
        return list(csv.DictReader(arch))


def bootstrap(y, probs):
    rng = np.random.default_rng(SEED)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(neg) or not len(pos):
        raise RuntimeError("Bootstrap necesita ambas clases")
    idx = np.concatenate([rng.choice(neg, (N, len(neg)), replace=True),
                          rng.choice(pos, (N, len(pos)), replace=True)], axis=1)
    scores = {}
    for name, p in probs.items():
        pred = (p[idx] >= 0.5)
        Y = y[idx]
        tn = ((Y == 0) & ~pred).sum(axis=1).astype(float)
        fp = ((Y == 0) & pred).sum(axis=1).astype(float)
        fn = ((Y == 1) & ~pred).sum(axis=1).astype(float)
        tp = ((Y == 1) & pred).sum(axis=1).astype(float)
        acc = (tn + tp) / len(y)
        f1_valid = 2 * tp / (2 * tp + fp + fn).clip(min=1)
        f1_invalid = 2 * tn / (2 * tn + fp + fn).clip(min=1)
        macro = (f1_valid + f1_invalid) / 2
        spec = tn / (tn + fp)
        bal = (spec + tp / (tp + fn)) / 2
        scores[name] = {"accuracy": acc, "macro_f1": macro,
                        "balanced_accuracy": bal, "specificity_invalid": spec}
    def ci(arr):
        return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
    result = {"method": "bootstrap pareado estratificado por clase, resampleo de videos completos",
              "replicas": N, "seed": SEED, "classes": {"0": len(neg), "1": len(pos)},
              "95pct_method": "percentiles 2.5 / 97.5",
              "intervalos_modelos": {name: {k: ci(a) for k, a in vals.items()}
                                     for name, vals in scores.items()},
              "diferencias_fusion_menos": {
                  other: {k: {"valor_observado": float(np.mean(scores["fusion"][k] - scores[other][k])),
                              "ic95": ci(scores["fusion"][k] - scores[other][k])}
                          for k in ("accuracy", "macro_f1", "balanced_accuracy", "specificity_invalid")}
                  for other in ("pose", "seg")},
              "auc_ci": "No calculado; ROC-AUC informado solo como estimación puntual de probabilidades",
              "interpretacion": "Descriptivo; independencia por atleta/sesión no acreditada, IC pueden ser optimistas"}
    return result


def graficos(y, probs, metrics):
    pdir = OUT / "figures"
    pdir.mkdir(exist_ok=True)
    for nombre, m in metrics.items():
        cm = np.asarray(m["matriz_confusion_0_1"])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicción (0 inválido, 1 válido)",
               ylabel="Real", title=MODELOS[nombre])
        fig.tight_layout()
        fig.savefig(pdir / f"confusion_{nombre}.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 1, 21)
        for label, etiqueta in ((0, "Inválido"), (1, "Válido")):
            ax.hist(probs[nombre][y == label], bins=bins, alpha=0.65,
                    label=etiqueta)
        ax.axvline(0.5, color="black", linestyle="--")
        ax.set(xlabel="P(válido)", ylabel="Videos Test",
               title=MODELOS[nombre])
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / f"probabilities_{nombre}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, pp in probs.items():
        fpr, tpr, _ = roc_curve(y, pp)
        ax.plot(fpr, tpr, label=f"{MODELOS[nombre]} (AUC {metrics[nombre]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "roc_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, pp in probs.items():
        prec, rec, _ = precision_recall_curve(y, pp)
        ax.plot(rec, prec, label=f"{MODELOS[nombre]} (AP {metrics[nombre]['pr_auc_average_precision']:.3f})")
    ax.axhline(float(y.mean()), ls="--", color="gray", label="Prevalencia válidos")
    ax.set(xlabel="Recall válidos", ylabel="Precision válidos", title="PR Test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "pr_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k in zip(axes, ("accuracy", "macro_f1")):
        ax.bar([MODELOS[n] for n in MODELOS], [metrics[n][k] for n in MODELOS])
        ax.set_ylim(0, 1)
        ax.set_ylabel(k)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(pdir / "metrics_comparison.png", dpi=160)
    plt.close(fig)


def main():
    protocol = OUT / "bp_front_final_test_protocol_frozen.json"
    expected_sha = (OUT / "bp_front_final_test_protocol_frozen.sha256").read_text(
        encoding="utf-8").split()[0]
    if sha256(protocol) != expected_sha:
        raise RuntimeError("Protocolo ha cambiado")
    cfg = json.loads(protocol.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != cfg["sha256_artifacts"]["analysis_code"]["sha256"]:
        raise RuntimeError("Código analítico cambió respecto al protocolo")
    log = json.loads((OUT / "bp_front_final_test_execution_log.json").read_text(encoding="utf-8"))
    pred_path = OUT / "bp_front_final_test_predictions.csv"
    if (log["state"] != "COMPLETED_ONE_PASS" or
            sha256(pred_path) != log["result_sha256"] or
            log["protocol_sha256"] != expected_sha):
        raise RuntimeError("Resultados Test o log no verificables")
    rows = leer_csv(pred_path)
    if len(rows) != 100 or [r["video_id"] for r in rows] != cfg["test_ids"]:
        raise RuntimeError("100 IDs Test congelados no coinciden")
    y = np.asarray([int(r["label_real"]) for r in rows], dtype=int)
    if set(y) != {0, 1}:
        raise RuntimeError("Una clase ausente en Test")
    probs = {k: np.asarray([float(r["p_" + k]) for r in rows])
             for k in MODELOS}
    for r in rows:
        if abs(float(r["p_fusion"]) - 0.5 * (float(r["p_pose"]) + float(r["p_seg"]))) > 1e-12:
            raise RuntimeError("Probabilidad de fusion distinta a 50/50")
        for k in MODELOS:
            if int(r["pred_" + k]) != int(float(r["p_" + k]) >= 0.5):
                raise RuntimeError("Predicción no usa umbral congelado")
    metrics = {k: metrica(y, p) for k, p in probs.items()}
    guardar_json(OUT / "bp_front_final_test_metrics.json",
                 {"models": metrics, "classes": {"invalid_0": int((y == 0).sum()),
                                                 "valid_1": int((y == 1).sum())},
                  "protocol_sha256": expected_sha,
                  "selected_model_still": cfg["candidato_preseleccionado"],
                  "pr_auc_definition": "Average Precision scikit-learn de probabilidades (no área trapezoidal PR)"})
    guardar_csv(OUT / "bp_front_final_test_confusion_matrices.csv", [
        {"model": k, "tn": m["tn"], "fp": m["fp"],
         "fn": m["fn"], "tp": m["tp"],
         "fpr": m["fpr"], "fnr": m["fnr"]} for k, m in metrics.items()])
    base = metrics["fusion"]
    guardar_csv(OUT / "bp_front_final_test_comparison.csv", [
        {"model": k, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
         "balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"],
         "fp": m["fp"], "fn": m["fn"],
         "fusion_minus_model_accuracy": base["accuracy"] - m["accuracy"],
         "fusion_minus_model_macro_f1": base["macro_f1"] - m["macro_f1"]}
        for k, m in metrics.items()])
    val_csvs = {
        "pose": POSE / "seed_42/validation_predictions.csv",
        "seg": SEG / "sin_relacion/seed_42/validation_predictions.csv",
        "fusion": FUSION / "late_fusion_sin_relacion/validation_predictions.csv"}
    val_metrics = {}
    for k, p in val_csvs.items():
        v = leer_csv(p)
        field = "p_fusion" if k == "fusion" else "probability_valid"
        val_metrics[k] = metrica(
            [int(x["label"]) for x in v], [float(x[field]) for x in v])
    campos = ("accuracy", "f1", "macro_f1", "balanced_accuracy", "roc_auc",
              "pr_auc_average_precision", "sensibilidad_valido",
              "especificidad_invalido", "tn", "fp", "fn", "tp")
    guardar_csv(OUT / "bp_front_validation_vs_test.csv", [
        {"model": k, **{f"validation_{c}": val_metrics[k][c] for c in campos},
         **{f"test_{c}": metrics[k][c] for c in campos},
         **{f"test_minus_validation_{c}": metrics[k][c] - val_metrics[k][c]
            for c in campos if c not in ("tn", "fp", "fn", "tp")}}
        for k in MODELOS])
    paired = []
    for i, r in enumerate(rows):
        ep, es, ef = [int(int(r["pred_" + k]) != y[i]) for k in MODELOS]
        disagreement = ("ambos_base_iguales" if r["pred_pose"] == r["pred_seg"]
                        else "pose_y_seg_discrepan")
        paired.append({"video_id": r["video_id"], "label_real": y[i],
                       "p_pose": r["p_pose"], "p_seg": r["p_seg"],
                       "p_fusion": r["p_fusion"],
                       "pred_pose": r["pred_pose"], "pred_seg": r["pred_seg"],
                       "pred_fusion": r["pred_fusion"],
                       "error_pose": ep, "error_seg": es, "error_fusion": ef,
                       "tipo_desacuerdo": disagreement,
                       "fusion_aprueba_invalido": int(y[i] == 0 and r["pred_fusion"] == "1"),
                       "fusion_cerca_umbral_0_1": int(abs(probs["fusion"][i] - 0.5) <= 0.1)})
    guardar_csv(OUT / "bp_front_final_test_paired_errors.csv", paired)
    names = np.asarray([r["video_id"] for r in rows])
    errores = {k: set(names[(p >= .5).astype(int) != y]) for k, p in probs.items()}
    compartir = {"los_tres": sorted(set.intersection(*errores.values())),
                 "pose_exclusivos": sorted(errores["pose"] - errores["seg"] - errores["fusion"]),
                 "seg_exclusivos": sorted(errores["seg"] - errores["pose"] - errores["fusion"]),
                 "fusion_exclusivos": sorted(errores["fusion"] - errores["pose"] - errores["seg"]),
                 "fusion_corrige_pose": sorted(errores["pose"] - errores["fusion"]),
                 "fusion_corrige_seg": sorted(errores["seg"] - errores["fusion"]),
                 "fusion_falla_pose_correcta": sorted(errores["fusion"] - errores["pose"]),
                 "fusion_falla_seg_correcta": sorted(errores["fusion"] - errores["seg"]),
                 "desacuerdos_pose_seg": sorted(names[(probs["pose"] >= .5) != (probs["seg"] >= .5)].tolist()),
                 "invalidos_aprobados_fusion": sorted(names[(y == 0) & (probs["fusion"] >= .5)].tolist()),
                 "cerca_umbral_0_1": sorted(names[np.abs(probs["fusion"] - .5) <= .1].tolist()),
                 "corr_p_pose_p_seg": float(np.corrcoef(probs["pose"], probs["seg"])[0, 1])}
    guardar_json(OUT / "bp_front_final_test_error_summary.json", compartir)
    ci = bootstrap(y, probs)
    # Valor observado de diferencias, no media bootstrap (sesgo MC).
    for b in ("pose", "seg"):
        for k, orig in (("accuracy", "accuracy"), ("macro_f1", "macro_f1"),
                        ("balanced_accuracy", "balanced_accuracy"),
                        ("specificity_invalid", "especificidad_invalido")):
            ci["diferencias_fusion_menos"][b][k]["valor_observado"] = (
                metrics["fusion"][orig] - metrics[b][orig])
    guardar_json(OUT / "bp_front_final_test_confidence_intervals.json", ci)
    graficos(y, probs, metrics)
    print(json.dumps({"n": len(rows), "classes": {"0": int((y == 0).sum()),
                                                  "1": int((y == 1).sum())},
                      "metrics": metrics, "errores": compartir},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
