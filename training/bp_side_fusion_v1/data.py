"""Carga exclusivamente Train/Validation monovista y reproducción de bases."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from training.bp.train_posecnn import transform
from training.bp_front_segcnn_v1.model import SegCNNTemporal
from training.bp_posecnn_common_v1.core import load_development, paths, predict, rows_csv, save_json
from training.bp_right_posecnn_common_v1.train import verify_prepared as verify_right_pose_prepared
from training.bp_side_segcnn_v1.data import cohort, sha
from training.bp_side_segcnn_v1.train import metrics
from training.common.train_utils import PoseCNN
from .models import SegEmbeddingAdapter, verify_adapters, late_fusion


def fusion_out(view):
    return paths(view)["left_out"].parent / f"bp_{view}_fusion_v1"


def snapshot(view):
    p=paths(view);files={
        "cohort":p["cohort"],"split":p["pose_data"]/"split.csv","labels":p["labels"],
        "pose_code":p["left_out"].parent/"common/train_utils.py",
        "pose_preprocess_code":p["left_out"].parent/"bp/train_posecnn.py",
        "seg_model_code":p["left_out"].parent/"bp_front_segcnn_v1/model.py",
        "seg_data_code":p["left_out"].parent/"bp_side_segcnn_v1/data.py",
        "seg_source_manifest":p["seg_train"]/"source_manifest.json",
        "pose_checkpoint":p["historical_ckpt"] if view=="left" else p["right_out"]/"seed_1337/best_model.pth",
        "pose_scaler":p["historical_scaler"] if view=="left" else p["right_out"]/"scaler_train_original.pkl",
        "pose_validation_predictions":p["left_out"]/"validation_predictions.csv" if view=="left" else p["right_out"]/"seed_1337/validation_predictions.csv",
        "pose_integrity":p["left_out"]/"final_verification.json" if view=="left" else p["right_out"]/"final_verification.json",
    }
    for variant in ("full","sin_relacion"):
        base=p["seg_train"]/variant
        sel=json.loads((base/"selected_checkpoint.json").read_text(encoding="utf-8"))
        files[f"seg_{variant}_checkpoint"]=p["seg_train"]/sel["checkpoint"]
        files[f"seg_{variant}_selected"]=base/"selected_checkpoint.json"
        files[f"seg_{variant}_prepared"]=base/"prepared_development.npz"
        files[f"seg_{variant}_channels"]=base/"channels_and_train_preprocessing.json"
        files[f"seg_{variant}_validation_predictions"]=base/f'seed_{sel["seed"]}'/"validation_predictions.csv"
    return {name:{"path":str(path.resolve()),"sha256":sha(path)} for name,path in files.items()}


def get_pose(view,coh,device):
    p=paths(view)
    if view=="left":
        source=json.loads((p["left_out"]/"integrity.json").read_text(encoding="utf-8"))
        if source["status"]!="LEFT_POSECNN_HISTORICAL_COMPATIBLE":raise RuntimeError("LEFT no compatible")
        loaded,_,_=load_development("left")
        with open(p["historical_scaler"],"rb") as f:scaler=pickle.load(f)
        if int(scaler.n_samples_seen_)!=422:raise RuntimeError("Scaler LEFT no corresponde a Train")
        ckpt=p["historical_ckpt"]
    else:
        prepared,scaler,src=verify_right_pose_prepared()
        loaded={role:(prepared[f'X_{role}'],prepared[f'y_{role}'],prepared[f'ids_{role}'].tolist())
                for role in ("train","validation")}
        if int(scaler.n_samples_seen_)!=431:raise RuntimeError("Scaler RIGHT no corresponde a Train")
        ckpt=p["right_out"]/"seed_1337/best_model.pth"
        selection=json.loads((p["right_out"]/"selected_checkpoint.json").read_text(encoding="utf-8"))
        if selection["seed"]!=1337 or selection["checkpoint_sha256"]!=sha(ckpt):
            raise RuntimeError("PoseCNN RIGHT no coincide con selección validada")
    model=PoseCNN(T=170,K=8,C=3).to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    result={}
    for role in ("train","validation"):
        x,y,ids=loaded[role]
        expected=[r["video_id"] for r in coh[role]]
        if list(ids)!=expected or len(set(ids))!=len(ids):raise RuntimeError("Orden Pose/cohorte distinto")
        result[role]={"X":np.ascontiguousarray(transform(scaler,x).transpose(0,3,1,2)),
                      "y":y,"ids":list(ids)}
        if result[role]["X"].shape!=(len(expected),3,170,8):raise RuntimeError("Input Pose inválido")
    return model,result


def get_seg(view,variant,coh,device):
    p=paths(view);base=p["seg_train"]/variant
    selected=json.loads((base/"selected_checkpoint.json").read_text(encoding="utf-8"))
    ckpt=p["seg_train"]/selected["checkpoint"]
    if sha(ckpt)!=selected["checkpoint_sha256"]:raise RuntimeError("SegCNN checkpoint alterado")
    channels=json.loads((base/"channels_and_train_preprocessing.json").read_text(encoding="utf-8"))
    state=torch.load(ckpt,map_location=device,weights_only=True)
    if ((state["view"],state["variant"],state["seed"],state["channels_total"])!=(view,variant,selected["seed"],channels["C"]) or
        state["input_channels"]!=channels["input_channel_names"]):
        raise RuntimeError("SegCNN contrato/checkpoint incompatibles")
    model=SegCNNTemporal(channels["C"]).to(device)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    with np.load(base/"prepared_development.npz",allow_pickle=False) as z:
        data={role:{"X":z[f"{role}_X"],"y":z[f"{role}_y"],
                    "ids":z[f"{role}_ids"].tolist()} for role in ("train","validation")}
    for role in data:
        r=data[role]
        if (r["X"].shape!=(len(coh[role]),channels["C"],170) or
            r["ids"]!=[row["video_id"] for row in coh[role]] or not np.isfinite(r["X"]).all()):
            raise RuntimeError("SegCNN tensor/IDs incompatibles")
    return model,data,selected


def read_prob(path):
    rows=rows_csv(path)
    if len({r["video_id"] for r in rows})!=len(rows):raise RuntimeError("IDs predicción repetidos")
    return {r["video_id"]:(int(r["label"]),float(r["probability_valid"])) for r in rows}


@torch.no_grad()
def seg_infer(model,x,device):
    model.eval();values=[]
    for a in range(0,len(x),128):
        xb=torch.as_tensor(x[a:a+128],device=device)
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
            out=model(xb)
        values.extend(torch.sigmoid(out.float()).cpu().numpy().tolist())
    return np.asarray(values,np.float64)


@torch.no_grad()
def get_embeddings(pose,seg,xp,xs,device):
    from training.bp_posecnn_common_v1.core import PoseEmbeddingAdapter
    pose.eval();seg.eval()
    pa=PoseEmbeddingAdapter(pose).to(device).eval()
    sa=SegEmbeddingAdapter(seg).to(device).eval()
    result=[]
    for a in range(0,len(xp),128):
        p=torch.as_tensor(xp[a:a+128],device=device)
        s=torch.as_tensor(xs[a:a+128],device=device)
        zp=pa.extract_embeddings(p)
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
            zs=sa.extract_embeddings(s)
        result.append(torch.cat((zp.float(),zs.float()),dim=1).cpu().numpy())
    z=np.ascontiguousarray(np.concatenate(result),dtype=np.float32)
    if z.shape!=(len(xp),192) or not np.isfinite(z).all():raise RuntimeError("Embeddings no finitos/forma incorrecta")
    return z


def verify_base_prob(view,variant,pose_model,seg_model,pose_data,seg_data,selected,device):
    p=paths(view);ids=pose_data["validation"]["ids"];y=pose_data["validation"]["y"]
    a=predict(pose_model,pose_data["validation"]["X"].transpose(0,2,3,1),device)
    b=seg_infer(seg_model,seg_data["validation"]["X"],device)
    pose_pred=read_prob(p["left_out"]/"validation_predictions.csv" if view=="left" else
                        p["right_out"]/"seed_1337/validation_predictions.csv")
    seg_pred=read_prob(p["seg_train"]/variant/f'seed_{selected["seed"]}'/"validation_predictions.csv")
    if set(ids)!=set(pose_pred) or set(ids)!=set(seg_pred):raise RuntimeError("ID de Prediction distinto")
    diff_pose=max(abs(a[i]-pose_pred[vid][1]) for i,vid in enumerate(ids))
    diff_seg=max(abs(b[i]-seg_pred[vid][1]) for i,vid in enumerate(ids))
    if (diff_pose>1e-6 or diff_seg>1e-6 or
        any(int(y[i])!=pose_pred[vid][0] or int(y[i])!=seg_pred[vid][0] for i,vid in enumerate(ids))):
        raise RuntimeError(f"No se reproducen probabilidades {view}/{variant}: {diff_pose} {diff_seg}")
    ref_pose=metrics(y,a);ref_seg=metrics(y,b)
    for name,score in (("pose",ref_pose),("seg",ref_seg)):
        if name=="pose":
            expected_file=(p["left_out"]/"validation_metrics.json" if view=="left" else
                           p["right_out"]/"seed_1337/metrics.json")
        else:
            expected_file=p["seg_train"]/variant/f'seed_{selected["seed"]}'/"metrics.json"
        report=json.loads(expected_file.read_text(encoding="utf-8"))
        target=report["metrics"] if name=="pose" and view=="left" else report["validation_metrics"]
        if any(not np.isclose(score[k],target[k],rtol=0,atol=1e-9) for k in
               ("accuracy","f1","macro_f1","roc_auc","TN","FP","FN","TP")):
            raise RuntimeError("Métricas base no reproducidas")
    return a,b,{"max_abs_pose_probability_diff":diff_pose,"max_abs_seg_probability_diff":diff_seg,
                "pose_metrics":ref_pose,"seg_metrics":ref_seg,"validation_ids_count":len(ids)}


def prepare(view,config):
    groups,names,p,info=cohort(view)
    out=fusion_out(view);out.mkdir(parents=True,exist_ok=True)
    # Antes de abrir secuencias de desarrollo, se conservan hashes de modelos y fuentes.
    before=snapshot(view)
    save_json(out/"fusion_config.json",config)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose,pose_data=get_pose(view,groups,device)
    val_ids=pose_data["validation"]["ids"]
    for variant in ("full","sin_relacion"):
        seg,seg_data,selected=get_seg(view,variant,groups,device)
        for role in ("train","validation"):
            if (pose_data[role]["ids"]!=seg_data[role]["ids"] or
                not np.array_equal(pose_data[role]["y"],seg_data[role]["y"])):
                raise RuntimeError(f"Pose/Seg no comparten IDs/labels {view}/{variant}/{role}")
        pp,ss,reproduced=verify_base_prob(view,variant,pose,seg,pose_data,seg_data,selected,device)
        adapter=verify_adapters(pose,seg,pose_data["validation"]["X"],seg_data["validation"]["X"],device)
        n_grad_pose=sum(q.numel() for q in pose.parameters() if q.requires_grad)
        n_grad_seg=sum(q.numel() for q in seg.parameters() if q.requires_grad)
        if n_grad_pose or n_grad_seg:raise RuntimeError("Encoder descongelado")
        save_json(out/f"base_reproducibility_{variant}.json",{**reproduced,**adapter,
                  "pose_trainable_parameters":n_grad_pose,"seg_trainable_parameters":n_grad_seg,
                  "selected_seg_seed":selected["seed"],"test_npy_opened":0})
        late_dir=out/f"late_fusion_{variant}";late_dir.mkdir(exist_ok=True)
        combined=late_fusion(pp,ss)
        late_score=metrics(pose_data["validation"]["y"],combined)
        save_json(late_dir/"validation_metrics.json",{"view":view,"variant":variant,"metrics":late_score,
                  "formula":"p=.5*p_pose+.5*p_seg","threshold":.5,"trainable_parameters":0,"test_npy_opened":0})
        from training.bp_posecnn_common_v1.core import save_csv
        save_csv(late_dir/"validation_predictions.csv",[{"video_id":vid,"label":int(y),
            "pose_probability":float(a),"seg_probability":float(b),"probability_valid":float(c),
            "predicted":int(c>=.5),"error":int((c>=.5)!=y),"false_positive":int(y==0 and c>=.5)}
            for vid,y,a,b,c in zip(val_ids,pose_data["validation"]["y"],pp,ss,combined)])
        embed_dir=out/f"embedding_fusion_{variant}";embed_dir.mkdir(exist_ok=True)
        ztr=get_embeddings(pose,seg,pose_data["train"]["X"],seg_data["train"]["X"],device)
        zva=get_embeddings(pose,seg,pose_data["validation"]["X"],seg_data["validation"]["X"],device)
        mean=ztr.mean(0,dtype=np.float64).astype(np.float32)
        std=ztr.std(0,dtype=np.float64).astype(np.float32)
        std=np.maximum(std,1e-6)
        with open(embed_dir/"embedding_scaler_train.npz","wb") as file:
            np.savez_compressed(file,mean=mean,std=std,n_samples_train=len(ztr),dims=np.asarray([64,128]))
        with open(embed_dir/"embeddings_development.npz","wb") as file:
            np.savez_compressed(file,train_Z=((ztr-mean)/std).astype(np.float32),
                train_y=pose_data["train"]["y"],train_ids=np.asarray(pose_data["train"]["ids"]),
                validation_Z=((zva-mean)/std).astype(np.float32),
                validation_y=pose_data["validation"]["y"],validation_ids=np.asarray(val_ids))
        save_json(embed_dir/"embedding_contract.json",{"view":view,"variant":variant,
                  "pose_dim":64,"seg_dim":128,"concat_dim":192,"head_params":6209,
                  "scaler_fit_train_original_embeddings_only":len(ztr),"train_embeddings_are_in_sample":True,
                  "pose_backbone_frozen":True,"seg_backbone_frozen":True,
                  "selected_pose_seed":42 if view=="left" else 1337,"selected_seg_seed":selected["seed"],
                  "scaler_sha256":sha(embed_dir/"embedding_scaler_train.npz"),
                  "embeddings_sha256":sha(embed_dir/"embeddings_development.npz"),
                  "test_embeddings_exported":False})
        print(view,variant,"base reproduced",reproduced["max_abs_pose_probability_diff"],
              reproduced["max_abs_seg_probability_diff"],"late F1",round(late_score["f1"],4),
              "Z",ztr.shape,zva.shape,flush=True)
    after=snapshot(view)
    if before!=after:raise RuntimeError("Fuentes protegidas mutaron durante la preparación")
    source={"view":view,"cohort":{"train":len(groups["train"]),"validation":len(groups["validation"]),
                                   "test_ids_only":91 if view=="left" else 93},
            "source_sha256":before,"code_sha256":{"models.py":sha(Path(__file__).parent/"models.py"),
                            "data.py":sha(Path(__file__)),"train.py":sha(Path(__file__).parent/"train.py")},
            "test_npy_opened":0,"test_features_opened":0,"validation_reproduced_all":True,
            "late_fusion_fixed_weight":.5,"embedding_dim":192,"head_params":6209}
    save_json(out/"fusion_source_manifest.json",source)
    return source
