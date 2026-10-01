"""Analisis descriptivo de calidad y comparacion V2/V3 ya entrenados."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_utils import compute_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
QUALITY_PATH = SCRIPT_DIR / "data" / "dl" / "manual_quality_audit.csv"
RESULTS_ROOT = SCRIPT_DIR / "results" / "dl"
PROFILES = ("left", "right", "side")
V2 = "baseline_v2"
V3 = "baseline_v3_manual_audit"
V2_CV = "baseline_v2_cv"
V3_CV = "baseline_v3_manual_audit_cv"


def subset_metrics(frame):
    if len(frame) == 0:
        return {"n": 0, "accuracy": None, "f1": None, "roc_auc": None, "errors": 0}

    metrics = compute_metrics(
        frame["true_label"].to_numpy(),
        frame["prediction"].to_numpy(),
        frame["probability"].to_numpy(),
    )
    return {
        "n": int(len(frame)),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "roc_auc": (
            None if metrics["roc_auc"] is None else float(metrics["roc_auc"])
        ),
        "errors": int(sum(frame["true_label"] != frame["prediction"])),
    }


def quality_analysis():
    quality = pd.read_csv(QUALITY_PATH, dtype={"physical_event": str})
    warning_keys = set(
        zip(
            quality.loc[quality["quality_status"] == "QUALITY_WARNING", "physical_event"],
            quality.loc[quality["quality_status"] == "QUALITY_WARNING", "view"],
        )
    )
    excluded_keys = set(
        zip(
            quality.loc[quality["quality_status"] == "QUALITY_EXCLUDED", "physical_event"],
            quality.loc[quality["quality_status"] == "QUALITY_EXCLUDED", "view"],
        )
    )
    all_results = {}

    for profile in PROFILES:
        path = RESULTS_ROOT / profile / V3 / "predictions.csv"
        predictions = pd.read_csv(path, dtype={"physical_event": str})
        keys = list(zip(predictions["physical_event"], predictions["source_view"]))
        assert not (set(keys) & excluded_keys), (
            f"{profile}: muestra QUALITY_EXCLUDED llego al test"
        )
        predictions["quality_subset"] = [
            "QUALITY_WARNING" if key in warning_keys else "CLEAN" for key in keys
        ]
        predictions.to_csv(
            RESULTS_ROOT / profile / V3 / "predictions_with_quality.csv",
            index=False,
        )
        result = {
            subset: subset_metrics(
                predictions[predictions["quality_subset"] == subset]
            )
            for subset in ("CLEAN", "QUALITY_WARNING")
        }
        all_results[f"dl_{profile}"] = result
        (RESULTS_ROOT / profile / V3 / "quality_subset_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    return all_results


def scalar(data, key):
    return float(np.asarray(data[key]).reshape(-1)[0])


def comparison(quality_results):
    rows = []

    for profile in PROFILES:
        v2_metrics = np.load(RESULTS_ROOT / profile / V2 / "metrics.npz")
        v3_metrics = np.load(RESULTS_ROOT / profile / V3 / "metrics.npz")
        v2_cv = json.loads(
            (RESULTS_ROOT / profile / V2_CV / "cv_summary.json").read_text()
        )
        v3_cv = json.loads(
            (RESULTS_ROOT / profile / V3_CV / "cv_summary.json").read_text()
        )
        rows.append({
            "profile": f"dl_{profile}",
            "test": {
                "v2_f1": scalar(v2_metrics, "test_f1"),
                "v3_f1": scalar(v3_metrics, "test_f1"),
                "v2_auc": scalar(v2_metrics, "test_roc_auc"),
                "v3_auc": scalar(v3_metrics, "test_roc_auc"),
            },
            "cv": {
                "v2_f1_mean": v2_cv["f1"]["mean"],
                "v2_f1_std": v2_cv["f1"]["std"],
                "v3_f1_mean": v3_cv["f1"]["mean"],
                "v3_f1_std": v3_cv["f1"]["std"],
                "v2_auc_mean": v2_cv["roc_auc"]["mean"],
                "v2_auc_std": v2_cv["roc_auc"]["std"],
                "v3_auc_mean": v3_cv["roc_auc"]["mean"],
                "v3_auc_std": v3_cv["roc_auc"]["std"],
            },
            "quality_test": quality_results[f"dl_{profile}"],
        })

    output_dir = RESULTS_ROOT / V3
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v2_v3_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    report = [
        "# Baseline V3 - Post Manual Label Audit",
        "",
        "## Test V2 vs V3",
        "",
        "| Perfil | V2 F1 | V3 F1 | V2 AUC | V3 AUC |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        test = row["test"]
        report.append(
            f"| {row['profile']} | {test['v2_f1']:.4f} | {test['v3_f1']:.4f} | "
            f"{test['v2_auc']:.4f} | {test['v3_auc']:.4f} |"
        )

    report.extend([
        "",
        "## CV V2 vs V3",
        "",
        "| Perfil | V2 F1 | V3 F1 | V2 AUC | V3 AUC |",
        "|---|---:|---:|---:|---:|",
    ])

    for row in rows:
        cv = row["cv"]
        report.append(
            f"| {row['profile']} | {cv['v2_f1_mean']:.4f} +/- {cv['v2_f1_std']:.4f} | "
            f"{cv['v3_f1_mean']:.4f} +/- {cv['v3_f1_std']:.4f} | "
            f"{cv['v2_auc_mean']:.4f} +/- {cv['v2_auc_std']:.4f} | "
            f"{cv['v3_auc_mean']:.4f} +/- {cv['v3_auc_std']:.4f} |"
        )

    report.extend([
        "",
        "La arquitectura no cambio. Las diferencias reflejan ground truth auditado, descarte del evento 302 y nuevos split/folds.",
    ])
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return rows


if __name__ == "__main__":
    quality_results = quality_analysis()
    rows = comparison(quality_results)
    print(json.dumps({"quality": quality_results, "comparison": rows}, indent=2))
