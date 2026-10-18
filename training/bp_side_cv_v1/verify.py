"""Verificación independiente de hashes, estadísticos por fold y OOF. No abre Test."""
import json
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from training.bp_posecnn_common_v1.core import paths, rows_csv, save_json
from training.bp_side_segcnn_v1.data import sha
from training.bp_front_segcnn_v1.model import SegCNNTemporal
from training.common.train_utils import PoseCNN
from training.bp_side_fusion_v1.models import FusionHead
from .protocol import VARIANT, N_DEV, frozen_contract, output, snapshot


def run(view):
    out=output(view);p=paths(view)
    manifest=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    summary=json.loads((out/"cv_integrity.json").read_text(encoding="utf-8"))
    if (manifest["source_sha256"]!=snapshot(view) or
        sha(out/"development_unscaled.npz")!=manifest["development_array_sha256"] or
        sha(out/"fold_assignments.csv")!=manifest["fold_assignments_sha256"] or
        summary["test_npy_opened"]!=0 or summary["development_n"]!=N_DEV[view]):
        raise RuntimeError("CV preflight no íntegro")
    with np.load(out/"development_unscaled.npz",allow_pickle=False) as z:
        raw_pose=z["pose_original"].copy();raw_seg=z["seg_values"].copy()
        available=z["seg_availability"].astype(bool);ids=z["ids"].tolist()
    if "bp_354" in ids and view=="right":raise RuntimeError("Duplicado en development")
    expected_pose=manifest["pose_npy_train_validation_sha256"]
    expected_seg=manifest["seg_npz_train_validation_sha256"]
    if set(ids)!=set(expected_pose) or set(ids)!=set(expected_seg):
        raise RuntimeError("Fuente CV incluye video ajeno")
    pose_digests={r["video_id"]:r["sha256"] for r in rows_csv(p["pose_data"]/"audit.csv") if r["video_id"] in set(ids)}
    if len(set(pose_digests.values()))!=len(ids):
        raise RuntimeError("Duplicados exactos Pose dentro de development")
    for vid in ids:
        if (sha(p["keypoints"]/f"{vid}.npy")!=expected_pose[vid] or
            sha(p["source"]/"sequences"/f"{vid}_temporal.npz")!=expected_seg[vid]):
            raise RuntimeError("Fuente Pose/Seg cambió: "+vid)
    assignments=rows_csv(out/"fold_assignments.csv")
    index={vid:i for i,vid in enumerate(ids)}
    contract=frozen_contract(view)
    audited=[]
    for fold in range(1,6):
        d=out/f"fold_{fold:02d}"
        record=json.loads((d/"fold_metrics.json").read_text(encoding="utf-8"))
        inner=[index[r["video_id"]] for r in assignments if int(r["fold"])==fold and r["role"]=="inner_train"]
        outer=[index[r["video_id"]] for r in assignments if int(r["fold"])==fold and r["role"]=="outer_evaluation"]
        if set(inner)&set(outer) or len(inner)!=record["inner_train_n"] or len(outer)!=record["outer_n"]:
            raise RuntimeError("Roles del fold incompatibles")
        old_p=torch.load(d/"pose/best_internal_validation.pt",map_location="cpu",weights_only=True)
        old_s=torch.load(d/"seg/best_internal_validation.pt",map_location="cpu",weights_only=True)
        PoseCNN(T=170,K=8,C=3).load_state_dict(old_p["state_dict"],strict=True)
        SegCNNTemporal(contract["C"]).load_state_dict(old_s["state_dict"],strict=True)
        if (record["pose_training"]["checkpoint_sha256"]!=sha(d/"pose/best_internal_validation.pt") or
            record["seg_training"]["checkpoint_sha256"]!=sha(d/"seg/best_internal_validation.pt")):
            raise RuntimeError("Pesos del fold mutados")
        if view=="left":
            head=torch.load(d/"head/best_internal_validation.pt",map_location="cpu",weights_only=True)
            FusionHead().load_state_dict(head["state_dict"],strict=True)
            if record["fusion_training"]["checkpoint_sha256"]!=sha(d/"head/best_internal_validation.pt"):
                raise RuntimeError("Cabeza fold mutada")
        with open(d/"pose/scaler_inner_train.pkl","rb") as file:scaler=pickle.load(file)
        recomputed=StandardScaler().fit(raw_pose[inner].reshape(len(inner),-1))
        if scaler.n_samples_seen_!=len(inner) or not np.allclose(scaler.mean_,recomputed.mean_,atol=1e-10,rtol=0) or not np.allclose(scaler.scale_,recomputed.scale_,atol=1e-10,rtol=0):
            raise RuntimeError("Scaler Pose fold no procede únicamente de inner_train")
        stats=json.loads((d/"seg/inner_train_preprocessing.json").read_text(encoding="utf-8"))
        selected=contract["selected_feature_indices"]
        if stats["input_C_frozen"]!=contract["C"] or len(stats["statistics"])!=len(selected):
            raise RuntimeError("Topología Seg fold cambió")
        for k,i in enumerate(selected):
            obs=raw_seg[inner,:,i][available[inner,:,i]].astype(np.float64)
            item=stats["statistics"][k]
            med=float(np.median(obs)) if len(obs) else 0.
            mean=float(obs.mean()) if len(obs) else 0.
            std=float(obs.std()) if len(obs) else 1.
            std=std if std>=1e-6 else 1.
            if (item["name"]!=contract["selected_feature_names"][k] or
                item["n_observed_inner_train"]!=len(obs) or
                not np.allclose([item["median"],item["mean"],item["std"]],[med,mean,std],atol=1e-9,rtol=0)):
                raise RuntimeError("Estadístico Seg fold externo o incorrecto")
        audited.append({"fold":fold,"pose_scaler_inner_train_only":len(inner),
                        "seg_scaler_inner_train_only":len(inner),
                        "seg_channels_frozen":contract["C"],"outer_evaluation_only":len(outer),
                        "checkpoints_new_and_strict_load":True})
    if snapshot(view)!=manifest["source_sha256"]:
        raise RuntimeError("Checkpoint final original alterado")
    result={"view":view,"status":"CV_INTEGRITY_OK","folds":audited,
            "development_pose_npy_hashes_verified":len(expected_pose),
            "development_seg_npz_hashes_verified":len(expected_seg),
            "exact_pose_duplicates_within_development":0,
            "final_preselected_source_hashes_unchanged":True,
            "test_npy_opened":0,"test_seg_npz_opened":0}
    save_json(out/"final_verification.json",result)
    print(view,"folds verified",len(audited),"pose/seg source hashes",len(expected_pose),len(expected_seg),flush=True)
    return result


def main():
    run("left");run("right")


if __name__=="__main__":main()
