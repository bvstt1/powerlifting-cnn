"""FASE A+B. Congelar protocolo antes de abrir CUALQUIER secuencia Test.

Lee solo CSV de cohorte, videos mp4 para SHA-256 y Train/Val para reproducir
sus 99 probabilidades. Fallos de integridad bloquean Test.
"""
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch

from common import (RAIZ, OUT, POSE, SEG, FUSION, COHORTE, POSE_CKPT,
                    SEG_CKPT, SCALER_POSE, SCALER_SEG, TOLERANCIA_VAL,
                    cohortes, cargar_modelos, guardar_json, metrica, sha256)


def _csv_pred(path):
    with open(path, newline="", encoding="utf-8") as archivo:
        rows = list(csv.DictReader(archivo))
    if len(rows) != 99:
        raise RuntimeError(f"Validation distinto de 99: {path}")
    return rows


def video_hashes(particiones):
    """Video byte-identical por SHA-256; evidencia de duplicación exacta.

    No abre secuencias .npy/.npz Test; los hashes de mp4 sí se leen pre-Test.
    """
    tabla = defaultdict(list)
    pesos = {}
    for split in ("train", "validation", "test"):
        for r in particiones[split]:
            path = Path(r["video_path"])
            if (not path.is_file() or path.resolve().parent !=
                    (RAIZ / "dataset/bp/front").resolve() or
                    path.stem != r["video_id"]):
                raise RuntimeError(f"Video fuente no acreditado: {r['video_id']}")
            digest = sha256(path)
            tabla[digest].append((split, r["video_id"]))
            pesos[r["video_id"]] = {"sha256": digest,
                                    "size_bytes": path.stat().st_size}
    duplicados = {k: v for k, v in tabla.items() if len(v) > 1}
    fuga = {k: v for k, v in duplicados.items()
            if len({s for s, _ in v}) > 1}
    if fuga:
        raise RuntimeError(f"Duplicados MP4 exactos ENTRE splits: {fuga}")
    return pesos, duplicados


@torch.inference_mode()
def reproducir_val(particiones, dispositivo):
    sys.path.insert(0, str(FUSION))
    from fusion_dataset import cargar_fusion
    datos = cargar_fusion("sin_relacion")["validation"]
    ids = [r["video_id"] for r in particiones["validation"]]
    y = np.asarray([int(r["label"]) for r in particiones["validation"]])
    if datos["ids"] != ids or not np.array_equal(datos["y"], y):
        raise RuntimeError("Pose/Seg Validation no corresponde al split")
    pose, seg = cargar_modelos(dispositivo)
    p_pose = []
    for i in range(0, 99, 16):
        inp = torch.as_tensor(datos["pose"][i:i + 16], device=dispositivo)
        p_pose.extend(pose(inp).flatten().cpu().numpy().tolist())
    p_seg = []
    for i in range(0, 99, 128):
        inp = torch.as_tensor(datos["seg"][i:i + 128], device=dispositivo)
        with torch.autocast(device_type=dispositivo.type,
                            enabled=dispositivo.type == "cuda"):
            logits = seg(inp)
            probs = torch.sigmoid(logits.float())
        p_seg.extend(probs.flatten().cpu().numpy().tolist())
    p_pose, p_seg = np.asarray(p_pose), np.asarray(p_seg)
    p_fusion = 0.5 * p_pose + 0.5 * p_seg
    csvs = {
        "pose": _csv_pred(POSE / "seed_42/validation_predictions.csv"),
        "seg_sin": _csv_pred(SEG / "sin_relacion/seed_42/validation_predictions.csv"),
        "fusion": _csv_pred(FUSION / "late_fusion_sin_relacion/validation_predictions.csv"),
    }
    salida = {}
    for name, p, campo in (("pose", p_pose, "probability_valid"),
                           ("seg_sin", p_seg, "probability_valid"),
                           ("fusion", p_fusion, "p_fusion")):
        referencia = csvs[name]
        if [r["video_id"] for r in referencia] != ids or any(
                int(r["label"]) != yy for r, yy in zip(referencia, y)):
            raise RuntimeError(f"IDs/labels Validation discordantes {name}")
        pcsv = np.asarray([float(r[campo]) for r in referencia])
        diff = float(np.max(np.abs(p - pcsv)))
        if diff > TOLERANCIA_VAL or np.any(
                (p >= 0.5) != (pcsv >= 0.5)):
            raise RuntimeError(f"No reproduce {name}: diff={diff}")
        salida[name] = {"max_abs_prob_diff": diff, "metrics": metrica(y, p)}
    previos = {
        "pose": json.loads((POSE / "validation_recheck.json").read_text(encoding="utf-8"))["metrics"],
        "seg_sin": json.loads((SEG / "sin_relacion/seed_42/metrics.json").read_text(encoding="utf-8"))["validation_metrics"],
        "fusion": json.loads((FUSION / "late_fusion_sin_relacion/metrics.json").read_text(encoding="utf-8"))["validation_metrics"],
    }
    for name, prev in previos.items():
        for k in ("accuracy", "f1", "macro_f1", "balanced_accuracy", "roc_auc",
                  "tn", "fp", "fn", "tp"):
            if abs(salida[name]["metrics"][k] - prev[k]) > 1e-8:
                raise RuntimeError(f"Métrica Validation no reproducida: {name}/{k}")
    return salida


def main():
    if (OUT / "bp_front_final_test_protocol_frozen.json").exists():
        raise RuntimeError("Protocolo ya congelado: no sobrescribir")
    torch.set_num_threads(4)
    particiones = cohortes()
    previo = json.loads((FUSION / "fusion_source_manifest.json").read_text(
        encoding="utf-8"))["base_antes"]
    for rel, esperado in previo.items():
        if sha256(RAIZ / rel) != esperado:
            raise RuntimeError(f"SHA previo no coincide: {rel}")
    seleccionado = json.loads((FUSION / "fusion_selected_candidate.json").read_text(
        encoding="utf-8"))
    if seleccionado["candidato"] != "late_sin" or seleccionado["criterio"]["umbral"] != 0.5:
        raise RuntimeError("Candidato preseleccionado cambió")
    pose = json.loads((POSE / "selected_checkpoint.json").read_text(encoding="utf-8"))
    seg = json.loads((SEG / "sin_relacion/selected_checkpoint.json").read_text(encoding="utf-8"))
    channels = json.loads(SCALER_SEG.read_text(encoding="utf-8"))
    if (pose["seed"] != 42 or seg["seed"] != 42 or
            pose["threshold"] != 0.5 or seg["threshold"] != 0.5 or
            len(channels["indices"]) != 31 or channels["canales_total"] != 65 or
            channels["constantes_train"] != ["profile_ambiguous"] or
            len(channels["nombres_mascara"]) != 31 or
            len(channels["nombres_procedencia"]) != 3):
        raise RuntimeError("Canales/umbral/checkpoints alterados")
    esperado_pose = ["left_elbow", "left_shoulder", "right_shoulder",
                     "right_elbow", "right_wrist", "left_wrist"]
    config_pose = (POSE / "training_config.json")
    if json.loads(config_pose.read_text(encoding="utf-8"))["K"] != 6:
        raise RuntimeError("K de Pose cambió")
    video_sha, duplicados = video_hashes(particiones)
    print("Hashes MP4 pre-Test:", len(video_sha),
          "duplicados exactos dentro de split:", len(duplicados), flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reproduccion = reproducir_val(particiones, dev)
    print("Validation reproducida:", {k: v["max_abs_prob_diff"]
                                       for k, v in reproduccion.items()}, flush=True)

    # Listar rutas y hashes (sin abrir .npy/.npz Test).
    esenciales = {
        "cohort_csv": COHORTE, "pose_checkpoint": POSE_CKPT,
        "seg_checkpoint": SEG_CKPT, "pose_scaler": SCALER_POSE,
        "seg_channels_scaler": SCALER_SEG,
        "seg_schema": RAIZ / "segmentation/bp/front/temporal_features_v1/bp_front_features_schema.json",
        "pose_yolo_extractor": RAIZ / "extract_keypoints/bp/extract_keypoints_yolo_bp.py",
        "pose_model_code": POSE / "model.py",
        "seg_model_code": SEG / "model.py",
        "pose_preprocess_code": POSE / "data.py",
        "seg_preprocess_code": SEG / "data.py",
        "fusion_selected_candidate": FUSION / "fusion_selected_candidate.json",
        "pose_training_config": config_pose,
        "seg_training_config": SEG / "training_config.json",
        "fusion_config": FUSION / "fusion_config.json",
        "pose_validation_csv": POSE / "seed_42/validation_predictions.csv",
        "seg_validation_csv": SEG / "sin_relacion/seed_42/validation_predictions.csv",
        "fusion_validation_csv": FUSION / "late_fusion_sin_relacion/validation_predictions.csv",
        "evaluation_code": OUT / "evaluate_test.py",
        "preflight_code": Path(__file__), "analysis_code": OUT / "analyze.py",
        "common_code": OUT / "common.py",
    }
    hashes = {}
    for k, p in esenciales.items():
        if not p.is_file():
            raise RuntimeError(f"Falta archivo fundamental: {p}")
        hashes[k] = {"path": str(p.relative_to(RAIZ)), "sha256": sha256(p)}
    manifesto = {
        "status": "PREFLIGHT_OK", "hashes_fundamentales": hashes,
        "sha256_videos_incluidos": video_sha,
        "duplicados_mp4_intra_split": {k: v for k, v in duplicados.items()},
        "validation_reproducida": reproduccion,
        "no_se_abrieron_secuencias_test": True,
    }
    guardar_json(OUT / "bp_front_final_test_source_manifest.json", manifesto)
    protocolo = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "FROZEN_BEFORE_TEST_SEQUENCES",
        "cohorte": str(COHORTE.relative_to(RAIZ)),
        "particiones": {k: len(v) for k, v in particiones.items()},
        "test_ids": [r["video_id"] for r in particiones["test"]],
        "exclusiones": ["bp_349", "bp_352"],
        "positive_class": "1=valido; negativo 0=invalido",
        "split_sha256": hashes["cohort_csv"]["sha256"],
        "sha256_pretest_manifest": sha256(OUT / "bp_front_final_test_source_manifest.json"),
        "sha256_artifacts": hashes,
        "modelos": ["pose", "seg_sin", "late_sin"],
        "checkpoints": {"pose": hashes["pose_checkpoint"],
                        "seg_sin": hashes["seg_checkpoint"]},
        "preprocesamiento": {"pose": "centro/escala interhombros por video; grid linspace(0,T-1,170); scaler StandardScaler fit Train guardado; gap ausente sentinel",
                              "seg_sin": "31 canales fijados (indices/nombres), 31 availability, 3 bar-provenance; remuestreo con abstencion; mediana/media/std Train guardados; sin relacion barra-silueta"},
        "pose_keypoints_orden": esperado_pose,
        "seg_feature_indices": channels["indices"],
        "seg_feature_names": channels["nombres"],
        "seg_input_channels": 65,
        "fusion_formula": "p_fusion = 0.5*p_pose + 0.5*p_seg_sin",
        "threshold": 0.5,
        "criterio_seleccion_previo": seleccionado["criterio"],
        "candidato_preseleccionado": "late_sin",
        "metricas": ["accuracy", "precision", "recall", "f1", "macro_f1",
                     "balanced_accuracy", "roc_auc_probabilities", "average_precision",
                     "sensitivity_valid", "specificity_invalid", "tn", "fp", "fn", "tp", "fpr", "fnr"],
        "ic_95_policy": {"metodo": "bootstrap pareado estratificado por clase, unidad video, percentil 2.5/97.5",
                          "replicas": 5000, "seed": 2026, "intervalos_exploratorios": True,
                          "auc": "sin IC; punto calculado de probabilidades"},
        "version_codigo_evaluacion_sha256": hashes["evaluation_code"]["sha256"],
        "tolerancia_validacion": TOLERANCIA_VAL,
        "validation_max_diff": {k: v["max_abs_prob_diff"] for k, v in reproduccion.items()},
        "duplicados_exactos_entre_particiones": 0,
        "unica_evaluacion_confirmatoria": True,
        "prohibido": ["reentrenamiento", "ajuste_de_umbral", "cambiar_pesos_fusion",
                       "calibracion", "cambiar_split", "reextraer_video", "seleccion_por_test"],
        "entorno": {"python": platform.python_version(),
                    "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                    "device": str(dev)},
        "limitaciones": ["no hay prueba de independencia por atleta/sesion/evento",
                         "validacion reutilizada en decisiones de clasificador/fusion",
                         "exposicion historica de algunos Test durante desarrollo de visores",
                         "YOLO extractor entrenado con procedencia incompleta"],
    }
    archivo = OUT / "bp_front_final_test_protocol_frozen.json"
    guardar_json(archivo, protocolo)
    (OUT / "bp_front_final_test_protocol_frozen.sha256").write_text(
        sha256(archivo) + "  " + archivo.name + "\n", encoding="utf-8")
    print("PROTOCOLO CONGELADO", archivo, "SHA256", sha256(archivo), flush=True)


if __name__ == "__main__":
    main()
