"""Traslada artefactos PoseCNN historicos sin modificar sus bytes.

Uso: python -B -m training.legacy.migrate_artifact_layout [--apply]
Los manifests antiguos no se reescriben; las rutas antiguas quedan registradas
en el plan mostrado en pantalla. Ejecutar desde la raiz del repositorio.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "training"


def moves():
    changes = []

    def add(old, new):
        changes.append((BASE / old, BASE / new))

    # BP front historico: left/right ya usan posecnn y runs.
    for name in ("bp_front_pose_audit_summary.json", "bp_front_pose_audit.csv",
                 "bp_front_pose_preprocessing_manifest.csv", "bp_front_pose_split_arrays.npz",
                 "bp_front_pose_split.csv", "processed_sequences.npz", "raw_sequences.npz"):
        add(f"data/bp/front/{name}", f"data/bp/front/posecnn/{name}")
    for name in ("bp_front_pose_best_model.pth", "bp_front_pose_config.json",
                 "bp_front_pose_scaler.pkl", "cv"):
        add(f"models/bp/front/posecnn/{name}",
            f"models/bp/front/posecnn/runs/legacy_default/{name}")
    for name in ("analysis", "bp_front_pose_confusion_matrix.npy",
                 "bp_front_pose_cv_predictions.csv", "bp_front_pose_cv_results.csv",
                 "bp_front_pose_metrics.json", "bp_front_pose_test_confusion_matrix.png",
                 "bp_front_pose_test_predictions.csv", "bp_front_pose_training_curves.png",
                 "bp_front_pose_training_history.csv"):
        add(f"results/bp/front/posecnn/{name}",
            f"results/bp/front/posecnn/runs/legacy_default/{name}")

    for name in ("event_split.npz", "cv_event_folds.csv", "cv_event_folds_v3.csv",
                 "discordant_view_labels.csv", "manual_label_audit.csv",
                 "manual_quality_audit.csv"):
        add(f"data/dl/{name}", f"data/dl/shared/{name}")
    for view in ("front", "left", "right", "side"):
        for stage in ("preprocessed", "normalized", "split", "augmented"):
            add(f"data/dl/{view}/{stage}", f"data/dl/{view}/posecnn/{stage}")
        model_root = ("best_model.pth", "history.npz") if view == "front" else (
            "best_model.pth", "history.npz", "run_config.json")
        for name in model_root:
            add(f"models/dl/{view}/{name}",
                f"models/dl/{view}/posecnn/runs/legacy_default/{name}")
        for run in ("baseline_v2", "baseline_v2_cv",
                    "baseline_v3_manual_audit", "baseline_v3_manual_audit_cv"):
            if view != "front":
                add(f"models/dl/{view}/{run}",
                    f"models/dl/{view}/posecnn/runs/{run}")
                add(f"results/dl/{view}/{run}",
                    f"results/dl/{view}/posecnn/runs/{run}")
        old_results = ("cross_validation_boxplot.png", "misclassified.csv",
                       "predictions.csv", "report.md") if view == "front" else (
            "cross_validation_boxplot.png", "cross_validation_metrics.npz",
            "misclassified.csv", "predictions.csv", "report.md", "train_val_loss.png")
        for name in old_results:
            add(f"results/dl/{view}/{name}",
                f"results/dl/{view}/posecnn/runs/legacy_default/{name}")
    for run in ("baseline_v2", "baseline_v2_cv", "baseline_v3_manual_audit",
                "baseline_v3_manual_audit_cv"):
        add(f"results/dl/{run}", f"results/dl/shared/posecnn/runs/{run}")
    return changes


def checksum(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.digest()


def files(path):
    return sorted(p.relative_to(path) for p in path.rglob("*") if p.is_file()) if path.is_dir() else [Path(".")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Inventariar destinos migrados y sus hashes")
    parser.add_argument("--check", action="store_true", help="Comprobar el inventario existente sin modificarlo")
    args = parser.parse_args()
    changes = moves()
    target = BASE / "docs" / "migration_artifact_manifest.json"
    if args.check:
        manifest = json.loads(target.read_text(encoding="utf-8"))
        rows = manifest["files"]
        if len({row["new"] for row in rows}) != len(rows):
            raise RuntimeError("Destinos repetidos en inventario")
        for row in rows:
            path = ROOT / row["new"]
            if not path.is_file() or checksum(path).hex() != row["sha256"]:
                raise RuntimeError(f"Hash distinto o archivo ausente: {path}")
        print(f"Inventario historico OK: {len(rows)} archivos")
        return
    if args.verify:
        if any(src.exists() or not dst.exists() for src, dst in changes):
            raise RuntimeError("Migracion incompleta: verificar orígenes y destinos")
        rows = []
        for src, dst in changes:
            for part in files(dst):
                path = dst / part if dst.is_dir() else dst
                rows.append({"old": str(src.relative_to(ROOT) / part if dst.is_dir() else src.relative_to(ROOT)).replace('\\', '/'),
                             "new": str(path.relative_to(ROOT)).replace('\\', '/'),
                             "sha256": checksum(path).hex()})
        if target.exists():
            raise FileExistsError(f"No sobrescribir inventario historico: {target}")
        target.write_text(json.dumps({"purpose": "Inventario posterior a migracion, no protocolo nuevo",
                                      "files": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Destinos presentes: {len(changes)}; archivos inventariados: {len(rows)}")
        return
    missing = [str(src) for src, _ in changes if not src.exists()]
    occupied = [str(dst) for _, dst in changes if dst.exists()]
    if missing or occupied:
        raise RuntimeError(f"Migracion bloqueada: faltantes={missing}; colisiones={occupied}")
    names = [str(dst).casefold() for _, dst in changes]
    if len(names) != len(set(names)):
        raise RuntimeError("Dos fuentes apuntan al mismo destino")
    print(f"Movimientos: {len(changes)}. Faltantes: 0; colisiones: 0.")
    if not args.apply:
        for src, dst in changes:
            print(src.relative_to(ROOT), "->", dst.relative_to(ROOT))
        return
    for src, dst in changes:
        snapshot = {str(part): checksum(src / part if src.is_dir() else src)
                    for part in files(src)}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if snapshot != {str(part): checksum(dst / part if dst.is_dir() else dst)
                        for part in files(dst)}:
            raise RuntimeError(f"Bytes alterados: {src} -> {dst}")
    print(f"Migrados y comprobados: {len(changes)} movimientos sin alterar bytes.")


if __name__ == "__main__":
    main()
