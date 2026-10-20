"""Verificación final de artefactos Pose/Seg protegidos sin acceder Test NPZ/NPY."""
import json

import numpy as np

from training.bp.experiments.bp_side_segcnn_v1.train import metrics
from .core import ROOT, paths, rows_csv, save_json, sha


def verify(view):
    p=paths(view)
    out=p["left_out"] if view=="left" else p["right_out"]
    left=view=="left"
    source_audit=json.loads((p["source"]/"source_audit.json").read_text(encoding="utf-8"))
    pose_audit=json.loads((p["pose_data"]/"audit_summary.json").read_text(encoding="utf-8"))
    split={r["video_id"]:r["split"] for r in rows_csv(p["pose_data"]/"split.csv")}
    if source_audit["cross_split_exact_duplicates"]:
        raise RuntimeError("MP4 idénticos en particiones distintas")
    for pair in pose_audit["exact_pose_duplicates"]:
        roles={split.get(vid,"unassigned") for vid in pair}
        if len(roles & {"train","validation","test"})>1:
            raise RuntimeError("Pose idéntica cruza particiones: "+str(pair))
    if left:
        left_audit=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
        original=left_audit["source"]["source_sha256"]
        source={"pose_split":p["pose_data"]/"split.csv","labels":p["labels"],
                "historical_config":p["historical_config"],
                "model_code":ROOT/"training/common/train_utils.py",
                "pose_preprocess_code":ROOT/"training/bp/train_posecnn.py",
                "pose_preprocessing_csv":p["pose_data"]/"preprocessing.csv",
                "pose_audit_csv":p["pose_data"]/"audit.csv",
                "current_cohort":p["cohort"]}
        for name,path in source.items():
            if sha(path)!=original[name]:raise RuntimeError("Fuente LEFT mutada: "+name)
        if (sha(p["historical_ckpt"])!=left_audit["checkpoint_sha256"] or
            sha(p["historical_scaler"])!=left_audit["scaler_sha256"] or
            left_audit["status"]!="LEFT_POSECNN_HISTORICAL_COMPATIBLE"):
            raise RuntimeError("Checkpoint/scaler LEFT histórico mutado")
        expected=91;pred=out/"validation_predictions.csv"
        audit_paths=left_audit["source"]["train_validation_npy_sha256"]
    else:
        a=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
        source=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
        historical=json.loads((out/"historical_right_audit.json").read_text(encoding="utf-8"))
        selected=json.loads((out/"selected_checkpoint.json").read_text(encoding="utf-8"))
        if (sha(p["historical_ckpt"])!=historical["historical_checkpoint_sha256"] or
            sha(p["historical_scaler"])!=historical["historical_scaler_sha256"] or
            sha(p["pose_data"]/"split.csv")!=source["source_sha256"]["pose_split"] or
            sha(p["labels"])!=source["source_sha256"]["labels"] or
            sha(out/"scaler_train_original.pkl")!=a["new_scaler_sha256"] or
            a["new_best_checkpoint_sha256"]!=selected["checkpoint_sha256"] or
            selected["checkpoint_sha256"]!=sha(out/selected["checkpoint"])):
            raise RuntimeError("Checkpoint/scaler RIGHT mutado")
        if a["status"]!="RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE":
            raise RuntimeError("Entrenamiento RIGHT no completado")
        expected=93;pred=out/f'seed_{selected["seed"]}'/"validation_predictions.csv"
        audit_paths=source["train_validation_npy_sha256"]
    for vid,digest in audit_paths.items():
        if sha(p["keypoints"]/f"{vid}.npy")!=digest:
            raise RuntimeError("Keypoints Train/Validation mutados: "+vid)
    comparison=json.loads((out/"validation_pose_seg_comparison.json").read_text(encoding="utf-8"))
    rows=rows_csv(pred)
    if (len(rows)!=expected or len({r["video_id"] for r in rows})!=expected or
        comparison["validation_videos"]!=expected or comparison["test_npy_opened"]!=0):
        raise RuntimeError("Validation no corresponde a cohortes")
    y=np.array([int(r["label"]) for r in rows]);prob=np.array([float(r["probability_valid"]) for r in rows])
    m=metrics(y,prob)
    for key in ("accuracy","f1","roc_auc","TN","FP","FN","TP"):
        if not np.isclose(m[key],comparison["metrics"]["pose"][key],rtol=0,atol=1e-9):
            raise RuntimeError("Métrica Pose cambia: "+key)
    for variant in ("full","sin_relacion"):
        sel=json.loads((p["seg_train"]/variant/"selected_checkpoint.json").read_text(encoding="utf-8"))
        if sha(p["seg_train"]/sel["checkpoint"])!=sel["checkpoint_sha256"]:
            raise RuntimeError("SegCNN mutada: "+variant)
    result={"view":view,"status":"INTEGRITY_OK","train_validation_npy_hashes_verified":len(audit_paths),
            "validation_predictions_verified":expected,"selected_pose_checkpoint_sha256":
            sha(p["historical_ckpt"]) if left else sha(out/selected["checkpoint"]),
            "segcnn_checkpoints_unmodified":True,"cross_split_exact_duplicates":0,
            "source_audit_sha256":sha(p["source"]/"source_audit.json"),
            "pose_audit_summary_sha256":sha(p["pose_data"]/"audit_summary.json"),
            "test_npy_opened":0,"test_features_opened":0}
    save_json(out/"final_verification.json",result)
    return result


def main():
    print(verify("left"));print(verify("right"))


if __name__=="__main__":main()
