"""Comparación documental PoseCNN/SegCNN por vista en Validation, SIN fusión."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from training.common.train_utils import PoseCNN
from training.bp_side_segcnn_v1.train import metrics
from .core import ROOT, paths, predict, rows_csv, save_csv, save_json, sha, verify_embedding


def read_predictions(path,positive="label"):
    rows=rows_csv(path)
    if len(rows)!=len({r["video_id"] for r in rows}):raise RuntimeError("IDs predicción duplicados")
    return {r["video_id"]:{"label":int(r[positive]),"prob":float(r["probability_valid"]),
                            "pred":int(r["predicted"])} for r in rows}


def compare(view,pose,seg,p_out):
    ids=set(pose)
    if not ids or any(set(s)!=ids for s in seg.values()):
        raise RuntimeError("Predicciones Validation sin mismo video_id")
    names=["pose","seg_full","seg_sin_relacion"]
    byname={"pose":pose,"seg_full":seg["full"],"seg_sin_relacion":seg["sin_relacion"]}
    y=np.asarray([pose[vid]["label"] for vid in sorted(ids)])
    if any(x[vid]["label"]!=pose[vid]["label"] for x in seg.values() for vid in ids):
        raise RuntimeError("Labels Pose/Seg distintas")
    if len(np.unique(y))!=2:raise RuntimeError("Falta alguna clase")
    probabilities={name:np.asarray([byname[name][vid]["prob"] for vid in sorted(ids)]) for name in names}
    results={name:metrics(y,probabilities[name]) for name in names}
    errors={name:{vid for vid in ids if byname[name][vid]["pred"]!=pose[vid]["label"]} for name in names}
    rows=[]
    for vid in sorted(ids):
        r={"video_id":vid,"label":pose[vid]["label"]}
        for name in names:
            r[name+"_prob"]=byname[name][vid]["prob"]
            r[name+"_pred"]=byname[name][vid]["pred"]
            r[name+"_error"]=int(vid in errors[name])
            r[name+"_fp"]=int(pose[vid]["label"]==0 and byname[name][vid]["pred"]==1)
        rows.append(r)
    save_csv(p_out/"validation_pose_seg_paired.csv",rows)
    pair={}
    for name in ("seg_full","seg_sin_relacion"):
        p,q=errors["pose"],errors[name]
        correlation=float(np.corrcoef(probabilities["pose"],probabilities[name])[0,1])
        pair[name]={"shared_errors":sorted(p&q),"pose_only_errors":sorted(p-q),
                    "seg_only_errors":sorted(q-p),
                    "prediction_disagreements":sorted(vid for vid in ids if pose[vid]["pred"]!=byname[name][vid]["pred"]),
                    "probability_pearson_r":correlation,
                    "shared_count":len(p&q),"pose_only_count":len(p-q),"seg_only_count":len(q-p)}
    save_json(p_out/"validation_pose_seg_comparison.json",{
        "view":view,"validation_videos":len(ids),"positive_class":"1=valido","threshold":.5,
        "metrics":results,"paired":pair,"three_way_shared_errors":sorted(set.intersection(*errors.values())),
        "pose_only_vs_both_seg_errors":sorted(errors["pose"]-errors["seg_full"]-errors["seg_sin_relacion"]),
        "test_npy_opened":0,"fusion_model_trained":False})
    return results,pair


def right_historical_audit(p,out):
    from sklearn.preprocessing import StandardScaler
    old=torch.load(p["historical_ckpt"],map_location="cpu",weights_only=True)
    model=PoseCNN(T=170,K=8,C=3)
    model.load_state_dict(old,strict=True)  # Solo auditoría. No llega al nuevo entrenamiento.
    with open(p["historical_scaler"],"rb") as file:scaler=pickle.load(file)
    history=rows_csv(p["historical_history"])
    info={"historical_checkpoint_sha256":sha(p["historical_ckpt"]),
          "historical_scaler_sha256":sha(p["historical_scaler"]),
          "historical_config_sha256":sha(p["historical_config"]),
          "historical_history_sha256":sha(p["historical_history"]),
          "architecture_state_keys_match":True,"historical_train_count_scaler":int(scaler.n_samples_seen_),
          "historical_scaler_features":int(scaler.n_features_in_),
          "history_best_validation_epoch":int(min(history,key=lambda row:float(row["val_loss"]))["epoch"]),
          "new_train_count":431,"historical_model_not_reused_for_training":True,
          "historical_test_metrics_not_read":True}
    if info["historical_train_count_scaler"]!=432 or info["historical_scaler_features"]!=170*8*3:
        raise RuntimeError("Scaler histórico inesperado")
    save_json(out/"historical_right_audit.json",info)
    return info


def run_left():
    p=paths("left");out=p["left_out"]
    status=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
    if status["status"]!="LEFT_POSECNN_HISTORICAL_COMPATIBLE":
        print("LEFT bloqueada: "+status["status"]);return None
    if (sha(p["historical_ckpt"])!=status["checkpoint_sha256"] or
        sha(p["historical_scaler"])!=status["scaler_sha256"] or
        sha(out/"validation_predictions.csv")!=status["valid_prediction_sha256"]):
        raise RuntimeError("Fuente histórica LEFT cambió")
    source=json.loads((p["seg_train"]/"source_manifest.json").read_text(encoding="utf-8"))
    pose=read_predictions(out/"validation_predictions.csv")
    seg={}
    for variant in ("full","sin_relacion"):
        checkpoint=json.loads((p["seg_train"]/variant/"selected_checkpoint.json").read_text(encoding="utf-8"))
        model=p["seg_train"]/checkpoint["checkpoint"]
        if sha(model)!=checkpoint["checkpoint_sha256"]:raise RuntimeError("SegCNN checkpoint alterado")
        seg[variant]=read_predictions(p["seg_train"]/variant/f'seed_{checkpoint["seed"]}'/"validation_predictions.csv")
    results,pair=compare("left",pose,seg,out)
    manifest={"status":"LEFT_POSECNN_HISTORICAL_COMPATIBLE","checkpoint_sha256":status["checkpoint_sha256"],
              "scaler_sha256":status["scaler_sha256"],"cohort_sha256":sha(p["cohort"]),
              "split_sha256":sha(p["pose_data"]/"split.csv"),"segcnn_source_manifest_sha256":sha(p["seg_train"]/"source_manifest.json"),
              "model_code_sha256":sha(ROOT/"training/common/train_utils.py"),"test_npy_opened":0}
    save_json(out/"protected_sources.json",manifest)
    print("left paired",{k:(len(v["shared_errors"]),len(v["pose_only_errors"]),len(v["seg_only_errors"])) for k,v in pair.items()},flush=True)
    return results,pair


def run_right():
    p=paths("right");out=p["right_out"]
    historical=right_historical_audit(p,out)
    source=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    selected=json.loads((out/"selected_checkpoint.json").read_text(encoding="utf-8"))
    if selected["seed"] not in (42,1337):raise RuntimeError("Seed seleccionada inválida")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file=out/selected["checkpoint"]
    if sha(model_file)!=selected["checkpoint_sha256"]:raise RuntimeError("Checkpoint RIGHT alterado")
    with np.load(out/"development_original.npz",allow_pickle=False) as z:
        validation=z["X_validation"]
        valids=z["ids_validation"].tolist()
    with open(out/"scaler_train_original.pkl","rb") as file:scaler=pickle.load(file)
    from training.bp.train_posecnn import transform
    x=transform(scaler,validation)
    for seed in (42,1337):
        other=out/f"seed_{seed}"/"best_model.pth"
        result=json.loads((out/f"seed_{seed}"/"metrics.json").read_text(encoding="utf-8"))
        if result["checkpoint_sha256"]!=sha(other):raise RuntimeError("Checksum seed RIGHT difiere")
        model=PoseCNN(T=170,K=8,C=3).to(device)
        model.load_state_dict(torch.load(other,map_location=device,weights_only=True),strict=True)
        pp=predict(model,x,device)
        original=read_predictions(out/f"seed_{seed}"/"validation_predictions.csv")
        difference=max(abs(float(pp[i])-original[v]["prob"]) for i,v in enumerate(valids))
        if difference>1e-6 or len(original)!=93:
            raise RuntimeError("Probabilidades RIGHT no reproducibles")
        embed=verify_embedding(model,x,device)
        embed["validation_probability_replay_max_abs_diff"]=difference
        embed["checkpoint_sha256"]=result["checkpoint_sha256"]
        save_json(out/f"seed_{seed}"/"embeddings_audit.json",embed)
    pose=read_predictions(out/f'seed_{selected["seed"]}'/"validation_predictions.csv")
    seg={}
    for variant in ("full","sin_relacion"):
        ckpt=json.loads((p["seg_train"]/variant/"selected_checkpoint.json").read_text(encoding="utf-8"))
        if sha(p["seg_train"]/ckpt["checkpoint"])!=ckpt["checkpoint_sha256"]:
            raise RuntimeError("SegCNN RIGHT alterada")
        seg[variant]=read_predictions(p["seg_train"]/variant/f'seed_{ckpt["seed"]}'/"validation_predictions.csv")
    results,pair=compare("right",pose,seg,out)
    integrity={"status":"RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE",
               "new_source_manifest_sha256":sha(out/"source_manifest.json"),
               "new_scaler_sha256":sha(out/"scaler_train_original.pkl"),
               "new_best_checkpoint_sha256":selected["checkpoint_sha256"],
               "historical_checkpoint_sha256":historical["historical_checkpoint_sha256"],
               "historical_scaler_sha256":historical["historical_scaler_sha256"],
               "segcnn_source_manifest_sha256":sha(p["seg_train"]/"source_manifest.json"),
               "train":431,"validation":93,"excluded_duplicate":"bp_354",
               "validation_replayed_seed_42_and_1337":True,
               "test_npy_opened":0,"test_features_opened":0}
    save_json(out/"integrity.json",integrity)
    print("right paired",{k:(len(v["shared_errors"]),len(v["pose_only_errors"]),len(v["seg_only_errors"])) for k,v in pair.items()},flush=True)
    return results,pair


def main():
    run_left();run_right()


if __name__=="__main__":main()
