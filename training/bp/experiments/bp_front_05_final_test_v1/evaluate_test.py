"""FASE C: una evaluación confirmatoria Test del protocolo BP_FRONT congelado.

Uso: python -m training.bp.experiments.bp_front_05_final_test_v1.evaluate_test
No entrena ni reajusta modelos, scalers, umbral ni particiones.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .common import (RAIZ, OUT, POSE, SEG, COHORTE, SCALER_POSE, SCALER_SEG,
                    cohortes, cargar_modelos, modulo, ruta_seg_test,
                    ruta_pose_test, guardar_csv, guardar_json, sha256)


def protocolo_verificado():
    p = OUT / "bp_front_final_test_protocol_frozen.json"
    sf = OUT / "bp_front_final_test_protocol_frozen.sha256"
    if not p.is_file() or not sf.is_file():
        raise RuntimeError("PRE-TEST ausente: prohibido abrir secuencias Test")
    digest = sf.read_text(encoding="utf-8").split()[0]
    if sha256(p) != digest:
        raise RuntimeError("Protocolo congelado modificado")
    spec = json.loads(p.read_text(encoding="utf-8"))
    if (spec["estado"] != "FROZEN_BEFORE_TEST_SEQUENCES" or
            spec["candidato_preseleccionado"] != "late_sin" or
            spec["fusion_formula"] != "p_fusion = 0.5*p_pose + 0.5*p_seg_sin" or
            spec["threshold"] != 0.5):
        raise RuntimeError("Política congelada inconsistente")
    if sha256(Path(__file__)) != spec["version_codigo_evaluacion_sha256"]:
        raise RuntimeError("Código de evaluación cambió desde congelamiento")
    for campo, info in spec["sha256_artifacts"].items():
        if sha256(RAIZ / info["path"]) != info["sha256"]:
            raise RuntimeError(f"Fuente esencial cambió tras freeze: {campo}")
    manifest = OUT / "bp_front_final_test_source_manifest.json"
    if sha256(manifest) != spec["sha256_pretest_manifest"]:
        raise RuntimeError("Manifest pre-Test alterado")
    return spec, digest


def scaler_pose():
    with np.load(SCALER_POSE, allow_pickle=False) as sc:
        s = StandardScaler()
        s.mean_ = sc["mean"].copy()
        s.scale_ = sc["scale"].copy()
        s.var_ = sc["var"].copy()
        s.n_features_in_ = s.mean_.size
        s.n_samples_seen_ = int(sc["n_samples"][0])
    if s.n_samples_seen_ != 462 or s.mean_.size != 170 * 6 * 3:
        raise RuntimeError("Scaler Pose Train incompatible")
    return s


def cargar_test(rows, spec):
    pose_data = modulo("test_pose_data", POSE / "data.py")
    seg_data = modulo("test_seg_data", SEG / "data.py")
    scaler = scaler_pose()
    cfg = json.loads(SCALER_SEG.read_text(encoding="utf-8"))
    if (cfg["indices"] != spec["seg_feature_indices"] or
            cfg["nombres"] != spec["seg_feature_names"] or
            cfg["canales_total"] != 65):
        raise RuntimeError("Canales Seg no coinciden con freeze")
    schema = json.loads((RAIZ / "segmentation/bp/front/temporal_features_v1"
                         / "bp_front_features_schema.json").read_text(encoding="utf-8"))
    nombres = [f["nombre"] for f in schema["features"]]
    xp_all, xs_all, estado, fuentes = [], [], [], []
    for r in rows:
        vid = r["video_id"]
        pose_path = ruta_pose_test(r)
        seg_path = ruta_seg_test(r)
        if (str(pose_path.resolve()) != str(Path(r["pose_path"]).resolve()) or
                str(seg_path.resolve()) != str(Path(r["features_path"]).resolve()) or
                not pose_path.is_file() or not seg_path.is_file()):
            raise RuntimeError(f"Archivo Test inesperado: {vid}")
        pose_raw = np.load(pose_path, allow_pickle=False)
        if pose_raw.shape != (int(r["T"]), 6, 3):
            raise RuntimeError(f"Shape Pose inesperado {vid}: {pose_raw.shape}")
        pnorm, p_mask, _, _ = pose_data.secuencia_a_170(pose_raw)
        px = pose_data.transformar(pnorm[None], scaler)[0]
        with np.load(seg_path, allow_pickle=False) as seq:
            if (seq["features"].shape != (int(r["T"]), 38) or
                    list(seq["feature_names"]) != nombres):
                raise RuntimeError(f"Shape/esquema Seg inesperado: {vid}")
            value, avail, prov = seg_data.remuestrear(seq)
        ii = cfg["indices"]
        vv, mm = value[:, ii], avail[:, ii]
        med = np.asarray(cfg["mediana_train"], dtype=np.float32)
        mu = np.asarray(cfg["media_train"], dtype=np.float32)
        std = np.asarray(cfg["std_train"], dtype=np.float32)
        inp = np.where(mm, vv, med[None, :])
        inp = (inp - mu[None, :]) / std[None, :]
        seg_x = np.concatenate((inp, mm.astype(np.float32), prov[:, :3]), axis=1)
        if seg_x.shape != (170, 65) or not np.isfinite(seg_x).all():
            raise RuntimeError(f"Input Seg inesperado: {vid}")
        xp_all.append(px.transpose(2, 0, 1))
        xs_all.append(seg_x.T)
        estado.append({"video_id": vid, "pose_missing_keypoints_resampled": int(
            (~p_mask.astype(bool)).sum()),
            "seg_missing_feature_values_resampled": int((~mm).sum()),
            "quality_status_seg": r["quality_status_seg"], "T_original": int(r["T"])})
        fuentes.append({"video_id": vid, "pose_sha256": sha256(pose_path),
                        "seg_features_sha256": sha256(seg_path)})
    xp = np.ascontiguousarray(np.stack(xp_all, axis=0), dtype=np.float32)
    xs = np.ascontiguousarray(np.stack(xs_all, axis=0), dtype=np.float32)
    if xp.shape != (100, 3, 170, 6) or xs.shape != (100, 65, 170):
        raise RuntimeError(f"Shapes Test finales incorrectos: {xp.shape}, {xs.shape}")
    return xp, xs, estado, fuentes


@torch.inference_mode()
def ejecutar(xp, xs, dispositivo):
    torch.set_num_threads(4)
    pose, seg = cargar_modelos(dispositivo)
    pp, ps = [], []
    for i in range(0, 100, 16):
        b = torch.as_tensor(xp[i:i + 16], device=dispositivo)
        pp.extend(pose(b).flatten().float().cpu().numpy().tolist())
    for i in range(0, 100, 128):
        b = torch.as_tensor(xs[i:i + 128], device=dispositivo)
        with torch.autocast(device_type=dispositivo.type,
                            enabled=dispositivo.type == "cuda"):
            logits = seg(b)
            ps.extend(torch.sigmoid(logits.float()).flatten().cpu().numpy().tolist())
    pp, ps = np.asarray(pp, dtype=np.float64), np.asarray(ps, dtype=np.float64)
    return pp, ps, 0.5 * (pp + ps)


def main():
    spec, digest = protocolo_verificado()  # Antes de abrir Test
    rows = cohortes()["test"]
    if [r["video_id"] for r in rows] != spec["test_ids"] or len(rows) != 100:
        raise RuntimeError("IDs Test difieren del freeze")
    resultado = OUT / "bp_front_final_test_predictions.csv"
    if resultado.exists():
        raise RuntimeError("Evaluación Test ya existe: no repetir")
    log = OUT / "bp_front_final_test_execution_log.json"
    guardar_json(log, {"state": "STARTED", "protocol_sha256": digest,
                       "time_utc": datetime.now(timezone.utc).isoformat(),
                       "test_npy_npz_opened": 0})
    try:
        print("Protocolo hash validado; abriendo 100 secuencias Test por única vez.",
              flush=True)
        xp, xs, estado, fuentes = cargar_test(rows, spec)
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pp, ps, pf = ejecutar(xp, xs, dispositivo)
        salida = []
        for r, p1, p2, p3 in zip(rows, pp, ps, pf):
            salida.append({"video_id": r["video_id"], "label_real": int(r["label"]),
                           "p_pose": float(p1), "p_seg": float(p2),
                           "p_fusion": float(p3),
                           "pred_pose": int(p1 >= 0.5),
                           "pred_seg": int(p2 >= 0.5),
                           "pred_fusion": int(p3 >= 0.5)})
        guardar_csv(OUT / "bp_front_final_test_input_quality.csv", estado)
        guardar_csv(OUT / "bp_front_final_test_input_hashes.csv", fuentes)
        guardar_csv(resultado, salida)
        guardar_json(log, {"state": "COMPLETED_ONE_PASS",
                           "protocol_sha256": digest,
                           "time_utc": datetime.now(timezone.utc).isoformat(),
                           "n_evaluated": len(salida), "device": str(dispositivo),
                           "result_sha256": sha256(resultado),
                           "test_inference_passes": 1})
        print("EVALUACIÓN ÚNICA completada:", len(salida),
              "output_sha256:", sha256(resultado), flush=True)
    except Exception as exc:
        guardar_json(log, {"state": "FAILED_NO_PROTOCOL_CHANGE",
                           "protocol_sha256": digest,
                           "error": f"{type(exc).__name__}: {exc}",
                           "time_utc": datetime.now(timezone.utc).isoformat(),
                           "partial_predictions_exist": resultado.exists()})
        raise


if __name__ == "__main__":
    main()
