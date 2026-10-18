"""Preparación y contratos de CV: todo ajuste estadístico ocurre en inner train."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from training.bp_posecnn_common_v1.core import ROOT, load_development, paths, rows_csv, save_csv, save_json, sha
from training.bp_side_segcnn_v1.data import cohort, load_development as load_seg

VARIANT={"left":"full","right":"sin_relacion"}
N_DEV={"left":513,"right":524}


def output(view):
    if view not in VARIANT:raise ValueError(view)
    return ROOT/"training"/f"bp_{view}_cv_v1"


def frozen_contract(view):
    p=paths(view);variant=VARIANT[view]
    contract=json.loads((p["seg_train"]/variant/"channels_and_train_preprocessing.json").read_text(encoding="utf-8"))
    expected=138 if view=="left" else 90
    if contract["variant"]!=variant or contract["C"]!=expected:
        raise RuntimeError("Esquema SegCNN congelado inesperado")
    return contract


def source_paths(view):
    p=paths(view);variant=VARIANT[view]
    seg=p["seg_train"]
    return {"cohort":p["cohort"],"pose_split":p["pose_data"]/"split.csv",
            "pose_labels":p["labels"],"pose_preprocess_code":ROOT/"training/bp/train_posecnn.py",
            "pose_architecture_code":ROOT/"training/common/train_utils.py",
            "seg_schema":p["schema"],"seg_features_code":ROOT/"training/bp_side_segcnn_v1/data.py",
            "seg_architecture_code":ROOT/"training/bp_front_segcnn_v1/model.py",
            "seg_contract":seg/variant/"channels_and_train_preprocessing.json",
            "seg_source_manifest":seg/"source_manifest.json",
            "final_pose_checkpoint":p["historical_ckpt"] if view=="left" else p["right_out"]/"seed_1337/best_model.pth",
            "final_seg_checkpoint":(seg/"full/seed_42/best_validation_loss.pt" if view=="left" else
                                    seg/"sin_relacion/seed_1337/best_validation_loss.pt"),
            "final_fusion_head":p["left_out"].parent/"bp_left_fusion_v1/embedding_fusion_full/seed_1337/best_validation_loss.pt" if view=="left" else
                               seg/"sin_relacion/seed_1337/best_validation_loss.pt"}


def snapshot(view):
    return {k:sha(path) for k,path in source_paths(view).items()}


def folds(ids,y,view,video_hashes=None):
    if len(ids)!=N_DEV[view] or len(set(ids))!=len(ids):raise RuntimeError("ID development inesperado")
    if view=="right" and "bp_354" in ids:raise RuntimeError("Duplicado right reincorporado")
    if video_hashes is not None:
        same=defaultdict(list)
        for vid in ids:same[video_hashes[vid]].append(vid)
        if any(len(v)>1 for v in same.values()):
            raise RuntimeError("Duplicados MP4 dentro de development: "+str([v for v in same.values() if len(v)>1]))
    indices=np.arange(len(ids));result=[]
    # Group IDs físicos no acreditados: estratificación por video tras verificar
    # que no queda ningún duplicado exacto dentro de development.
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(outer_train,outer_eval) in enumerate(splitter.split(indices,y),1):
        inner_fit,inner_validation=train_test_split(outer_train,test_size=.15,
                                 stratify=y[outer_train],random_state=42+fold)
        role={ids[j]:"inner_train" for j in inner_fit}
        role.update({ids[j]:"inner_validation" for j in inner_validation})
        role.update({ids[j]:"outer_evaluation" for j in outer_eval})
        if len(role)!=len(ids) or set(inner_fit)&set(inner_validation) or set(inner_fit)&set(outer_eval) or set(inner_validation)&set(outer_eval):
            raise RuntimeError("Folds superpuestos")
        result.append({"fold":fold,"inner_train":np.sort(inner_fit),"inner_validation":np.sort(inner_validation),
                       "outer_eval":np.sort(outer_eval),"roles":role})
    evaluations=[int(i) for f in result for i in f["outer_eval"]]
    if sorted(evaluations)!=indices.tolist():raise RuntimeError("OOF no cubre development exactamente")
    return result


def prepare(view):
    out=output(view);out.mkdir(parents=True,exist_ok=True)
    groups,names,p,info=cohort(view)
    if info["test_npz_opened"]!=0 or len(groups["train"])+len(groups["validation"])!=N_DEV[view]:
        raise RuntimeError("Cohorte development incompatible")
    if view=="right" and "bp_354" in {r["video_id"] for role in groups for r in groups[role]}:
        raise RuntimeError("Duplicado excluido reaparece")
    for role in ("train","validation"):
        if any(r["source_identity"]!="unverified_multiview" for r in groups[role]):
            raise RuntimeError("Nuevo group_id encontrado; revisar protocolo antes de CV")
    original_snapshot=snapshot(view)
    pose_data,pose_audit,_=load_development(view)
    seg_data,seg_hashes=load_seg(groups,names,p)
    ids=pose_data["train"][2]+pose_data["validation"][2]
    y=np.concatenate((pose_data["train"][1],pose_data["validation"][1]))
    xpose=np.concatenate((pose_data["train"][0],pose_data["validation"][0]))
    segx=np.concatenate((seg_data["train"][0],seg_data["validation"][0]))
    avail=np.concatenate((seg_data["train"][1],seg_data["validation"][1]))
    prov=np.concatenate((seg_data["train"][2],seg_data["validation"][2]))
    if (ids!=seg_data["train"][4]+seg_data["validation"][4] or
        not np.array_equal(y,np.concatenate((seg_data["train"][3],seg_data["validation"][3]))) or
        xpose.shape!=(len(ids),170,8,3) or segx.shape!=(len(ids),170,62) or
        avail.shape!=segx.shape or prov.shape!=segx.shape):
        raise RuntimeError("Pose/Seg no comparten development")
    source_rows={r["video_id"]:r for r in rows_csv(p["source"]/"source_audit.csv")}
    hashes={vid:source_rows[vid]["video_sha256"] for vid in ids}
    planned=folds(ids,y,view,hashes)
    assignments=[]
    distributions=[]
    for f in planned:
        for role in ("inner_train","inner_validation","outer_eval"):
            idx=f[role]
            counts=Counter(y[idx].tolist())
            if set(counts)!={0,1}:raise RuntimeError("Fold sin dos clases")
            distributions.append({"fold":f["fold"],"role":role,"n":len(idx),"invalid_0":counts[0],"valid_1":counts[1]})
        assignments.extend({"fold":f["fold"],"video_id":vid,"label":int(y[ids.index(vid)]),
            "role":f["roles"][vid],"physical_group_status":"unverified","group_key":"video_only_no_verified_pairs"}
                           for vid in ids)
    save_csv(out/"fold_assignments.csv",assignments)
    save_csv(out/"fold_class_distributions.csv",distributions)
    with open(out/"development_unscaled.npz","wb") as file:
        np.savez_compressed(file,pose_original=xpose,seg_values=segx,seg_availability=avail,
                            seg_provenance=prov,y=y,ids=np.asarray(ids),names=np.asarray(names))
    manifest={"view":view,"variant":VARIANT[view],"development_n":len(ids),
              "original_cohort":{"train":len(groups["train"]),"validation":len(groups["validation"]),
                                 "test_ids_only":91 if view=="left" else 93},
              "pose_npy_train_validation_sha256":pose_audit["train_validation_npy_sha256"],
              "seg_npz_train_validation_sha256":seg_hashes,
              "source_sha256":original_snapshot,"development_array_sha256":sha(out/"development_unscaled.npz"),
              "cv_code_sha256":{name:sha(Path(__file__).parent/name) for name in ("protocol.py","train.py")},
              "fold_assignments_sha256":sha(out/"fold_assignments.csv"),
              "fold_distribution_sha256":sha(out/"fold_class_distributions.csv"),
              "grouping":"video-level StratifiedKFold seed42; no credible physical group IDs; exact development MP4 duplicates zero",
              "verified_groups_within_dev":0,"test_npy_opened":0,"test_seg_npz_opened":0}
    save_json(out/"source_manifest.json",manifest)
    if snapshot(view)!=original_snapshot:raise RuntimeError("Fuentes finales alteradas")
    print(view,"dev",len(ids),"fold eval sizes",[len(f["outer_eval"]) for f in planned],
          "class",Counter(y.tolist()),flush=True)
    return manifest


def seg_fold_transform(x,a,p,contract,fit_idx,other_idx):
    """Transforma inner train + subconjunto sin usar el segundo para fit.

    Se conserva la topología/orden histórico C=138/90, incluso si un canal se
    vuelve constante o vacío dentro del inner train; mediana=0 cuando no hay
    observaciones y std=1 si no hay dispersión. Nunca se borra un canal por
    distribución de outer evaluation.
    """
    selected=contract["selected_feature_indices"]
    keys=contract["selected_feature_names"]
    train_x=x[fit_idx][:,:,selected];train_a=a[fit_idx][:,:,selected].astype(bool)
    stats=[]
    for j,name in enumerate(keys):
        values=train_x[:,:,j][train_a[:,:,j]].astype(np.float64)
        if len(values):
            med=float(np.median(values));mean=float(values.mean());std=float(values.std())
            if std<1e-6:std=1.
        else:med=mean=0.;std=1.
        stats.append({"name":name,"median":med,"mean":mean,"std":std,"n_observed_inner_train":len(values)})
    med=np.array([r["median"] for r in stats],np.float32)
    mean=np.array([r["mean"] for r in stats],np.float32)
    std=np.array([r["std"] for r in stats],np.float32)
    keep=contract["input_channel_indices_before_constant_filter"]
    def transform_idx(indices):
        val=x[indices][:,:,selected];valid=a[indices][:,:,selected].astype(bool)
        orig=p[indices][:,:,selected]
        numeric=(np.where(valid,val,med[None,None,:])-mean[None,None,:])/std[None,None,:]
        bits=[np.isin(orig,c).astype(np.float32) for c in (2,(3,6),4,5,7)]
        all_channels=np.concatenate((numeric,valid.astype(np.float32),*bits),axis=2).astype(np.float32)
        transformed=np.ascontiguousarray(all_channels[:,:,keep].transpose(0,2,1))
        if transformed.shape[1:]!=(contract["C"],170) or not np.isfinite(transformed).all():
            raise RuntimeError("Transform Seg fold incompatible")
        return transformed
    return transform_idx(fit_idx),transform_idx(other_idx),stats
