"""Insumos Train/Validation PoseCNN; prohíbe cargar NPY Test y checkpoints Test."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from training.bp.train_posecnn import POINTS, preprocess, FRAMES
from training.common.train_utils import PoseCNN
from training.bp.experiments.bp_side_segcnn_v1.data import cohort, paths as seg_paths, sha

ROOT = Path(__file__).resolve().parents[4]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows: raise ValueError("CSV vacío")
    with path.open("w", newline="", encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def rows_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def paths(view):
    p=seg_paths(view)
    p.update({"keypoints":ROOT/"keypoints/bp/yolo"/view,
              "pose_data":ROOT/"training/data/bp"/view/"posecnn",
              "historical_config":p["historical"]/"config.json",
              "historical_ckpt":p["historical"]/"best_model.pth",
              "historical_scaler":p["historical"]/"scaler.pkl",
              "historical_history":p["historical"]/"history.csv",
              "left_out":ROOT/"training/bp/experiments/bp_left_02_posecnn_compatibility_v1",
              "right_out":ROOT/"training/bp/experiments/bp_right_02_posecnn_common_v1",
               "seg_train":ROOT/"training/bp/experiments"/f"bp_{view}_01_segcnn_v1"})
    return p


def get_members(view):
    groups,names,p,info=cohort(view)
    poses=paths(view)
    split={row["video_id"]:row for row in rows_csv(poses["pose_data"]/"split.csv")}
    manifest={row["video_id"]:row for row in rows_csv(poses["pose_data"]/"preprocessing.csv")}
    labels={row["video_id"]:row for row in rows_csv(p["labels"])}
    rows=rows_csv(p["cohort"])
    excluded={r["video_id"] for r in rows if r["supervised_member"]=="False"}
    if view=="right" and ("bp_354" not in excluded or split["bp_354"]["split"]!="train"):
        raise ValueError("Duplicado RIGHT no excluido")
    if view=="left" and len(excluded)!=7 or view=="right" and len(excluded)!=13:
        raise ValueError("Exclusiones cambiadas")
    for role in ("train","validation"):
        for r in groups[role]:
            vid=r["video_id"]
            if (split[vid]["split"]!=role or split[vid]["label"]!=labels[vid]["label"] or
                manifest[vid]["status"]!="usable"):
                raise ValueError("Split/labels/calidad incompatibles: "+vid)
    # Para Test se cotejan ÚNICAMENTE IDs de splits; ningún video/NPY de Test.
    current_test={r["video_id"] for r in rows if r["split"]=="test" and r["supervised_member"]=="True"}
    old_test={vid for vid,r in split.items() if r["split"]=="test"}
    if current_test!=old_test or (set(split)&excluded)-({"bp_354"} if view=="right" else set()):
        raise ValueError("IDs Test/excluidos incompatibles")
    mismatches={role:sorted({v for v,r in split.items() if r["split"]==role}-
                            {r["video_id"] for r in groups[role]}) for role in groups}
    if mismatches!={"train":[],"validation":[]} and mismatches!={"train":["bp_354"],"validation":[]}:
        raise ValueError("Particiones incompatibles")
    if view=="left" and mismatches["train"] or view=="right" and mismatches["train"]!=["bp_354"]:
        raise ValueError("Membresía histórica inesperada")
    return groups,poses,info,manifest,labels,split,mismatches


def pose_digest(raw):
    digest=hashlib.sha256();digest.update(str(raw.shape).encode());digest.update(np.ascontiguousarray(raw).tobytes())
    return digest.hexdigest()


def load_development(view, check_history=False):
    """Lee NPY exclusivamente de train/validation de la cohorte común."""
    groups,p,source,pre,labels,split,mismatch=get_members(view)
    audit={r["video_id"]:r for r in rows_csv(p["pose_data"]/"audit.csv")}
    input_audit={r["video_id"]:r for r in rows_csv(p["source"].parent/"manifests/input_audit.csv")}
    loaded={};record=[];hashes={}
    for role in ("train","validation"):
        xs=[];ys=[];ids=[]
        for r in groups[role]:
            vid=r["video_id"]
            file=p["keypoints"]/f"{vid}.npy"
            raw=np.load(file,allow_pickle=False)
            if (raw.shape!=(int(r["T"]),8,3) or raw.ndim!=3 or np.isinf(raw).any() or
                raw.shape[0]!=int(input_audit[vid]["frames_decoded"]) or
                input_audit[vid]["keypoints_match_frames"]!="True" or
                pose_digest(raw)!=pre[vid]["sha256"] or pose_digest(raw)!=audit[vid]["sha256"]):
                raise ValueError("Pose original/frames/hash histórico difieren: "+vid)
            seq,center,scale=preprocess(raw)
            if seq.shape!=(170,8,3) or not np.isfinite(seq).all() or not np.allclose(center,[float(pre[vid]["center_x"]),float(pre[vid]["center_y"])],atol=2e-6,rtol=0) or not np.isclose(scale,float(pre[vid]["scale"]),atol=2e-6,rtol=0):
                raise ValueError("Preprocesamiento histórico no reproducible: "+vid)
            xs.append(seq);ys.append(int(labels[vid]["label"]));ids.append(vid)
            hashes[vid]=sha(file)
            record.append({"video_id":vid,"split":role,"label":int(labels[vid]["label"]),
                           "raw_T":len(raw),"npy_sha256":hashes[vid],"historical_pose_digest":pose_digest(raw),
                           "center_x":float(center[0]),"center_y":float(center[1]),"scale":float(scale),
                           "historical_preprocessing_match":True,
                           "grid":"linspace(0,T_original-1,170)"})
        loaded[role]=(np.stack(xs),np.asarray(ys,dtype=np.int64),ids)
        if set(ys)!={0,1}: raise RuntimeError("Clases incompletas")
    source_hashes={"current_cohort":sha(p["cohort"]),"pose_split":sha(p["pose_data"]/"split.csv"),
                   "pose_preprocessing_csv":sha(p["pose_data"]/"preprocessing.csv"),
                   "pose_audit_csv":sha(p["pose_data"]/"audit.csv"),"labels":sha(p["labels"]),
                   "historical_config":sha(p["historical_config"]),"model_code":sha(ROOT/"training/common/train_utils.py"),
                   "pose_preprocess_code":sha(ROOT/"training/bp/train_posecnn.py"),
                   "segcnn_cohort":sha(p["cohort"]),
                   "segcnn_selected_full":sha(p["seg_train"]/"full/selected_checkpoint.json"),
                   "segcnn_selected_ablated":sha(p["seg_train"]/"sin_relacion/selected_checkpoint.json")}
    return loaded,{"view":view,"cohort":{"train":len(loaded["train"][2]),"validation":len(loaded["validation"][2]),
                            "test_ids_only":len([s for s in split.values() if s["split"]=="test"]),
                            "unassigned":len([r for r in rows_csv(p["cohort"]) if r["split"]=="unassigned"])},
                   "historical_minus_current":mismatch,"source_sha256":source_hashes,
                   "train_validation_npy_sha256":hashes,"test_npy_opened":0,"test_features_opened":0},record


class PoseEmbeddingAdapter(torch.nn.Module):
    """Interfaz externa; estado y forward de PoseCNN históricos inalterados."""
    def __init__(self, model):
        super().__init__();self.model=model

    def forward(self,x):
        return self.model(x)

    def extract_embeddings(self,x):
        h=self.model.pool(self.model.features(x))
        h=self.model.classifier[0](h)
        h=self.model.classifier[1](h)
        h=self.model.classifier[2](h)
        return h

    def classifier_from_embeddings(self,embedding):
        return self.model.classifier[3:](embedding)


def verify_embedding(model,processed,device):
    model.eval();adapter=PoseEmbeddingAdapter(model).to(device).eval()
    x=torch.as_tensor(processed[:min(8,len(processed))],device=device).permute(0,3,1,2).contiguous()
    if x.shape[1:]!=(3,170,8):raise RuntimeError("Orden de tensores lateral incorrecto")
    with torch.no_grad():
        original=model(x)
        restored=adapter.classifier_from_embeddings(adapter.extract_embeddings(x))
        embed=adapter.extract_embeddings(x)
    maximum=float((original-restored).abs().max().cpu())
    if embed.shape!=(len(x),64) or maximum>1e-7 or not torch.equal(original,restored):
        raise RuntimeError("Embedding cambia probabilidades originales")
    return {"input_shape":list(x.shape),"embedding_shape":list(embed.shape),
            "embedding_dimension":int(embed.shape[1]),"max_abs_probability_difference":maximum,
            "exact_probability_match":True,"model_code_unchanged":True}


def predict(model,X,device,batch=16):
    model.eval();all_probs=[];criterion=torch.nn.BCELoss();total=0
    with torch.no_grad():
        for first in range(0,len(X),batch):
            xb=torch.as_tensor(X[first:first+batch],device=device).permute(0,3,1,2).contiguous()
            preds=model(xb).reshape(-1)
            all_probs.extend(preds.cpu().numpy().tolist())
    return np.asarray(all_probs,dtype=np.float64)
