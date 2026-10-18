"""CV retrospectiva 5-fold BP_LEFT/BP_RIGHT. Cada outer fold queda intacto para evaluación.

python -m training.bp_side_cv_v1.train --view left --phase prepare|train|all
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.bp.train_posecnn import seed_all as pose_seed_all, transform
from training.bp_front_segcnn_v1.model import SegCNNTemporal
from training.bp_posecnn_common_v1.core import PoseEmbeddingAdapter, ROOT, rows_csv, save_csv, save_json
from training.bp_side_fusion_v1.models import FusionHead, late_fusion
from training.bp_side_fusion_v1.train import CONFIG as FUSION_CONFIG, seed_all as head_seed_all
from training.bp_side_segcnn_v1.train import metrics as base_metrics
from training.common.train_utils import PoseCNN, augment_dataset
from training.bp_side_segcnn_v1.data import sha
from .protocol import VARIANT, frozen_contract, output, prepare, seg_fold_transform, snapshot

CONFIG={"n_splits":5,"outer_split":"StratifiedKFold shuffle=True random_state=42; video unit",
        "inner_split":"stratified 15% of outer train; random_state=42+fold",
        "pose_seed":42,"seg_seed":42,"fusion_head_seed":1337,
        "pose":{"T":170,"K":8,"C":3,"loss":"BCELoss","optimizer":"Adam", "lr":.001,
                "weight_decay":.0001,"batch":16,"epochs":200,"early_patience":8,
                "scheduler_patience":4,"scheduler_factor":.5,"augmentation":"original+3; translation scaling noise temporal_shift train only"},
        "seg":{"variant_by_view":VARIANT,"loss":"BCEWithLogitsLoss", "optimizer":"AdamW",
               "lr":.001,"weight_decay":.0001,"batch":32,"epochs":200,"early_patience":15,
               "scheduler_patience":5,"scheduler_factor":.5,
               "frozen_channel_topology":{"left":138,"right":90},"train_constant_channel_policy":"keep frozen topology; std=1 for constant/empty inner train, median=0 if empty"},
        "head":{"view":"left only","dim":192,"params":6209,"optimizer":"AdamW",
                "lr":.001,"weight_decay":.001,"batch":32,"epochs":100,"early_patience":15,
                "scheduler_patience":5,"scheduler_factor":.5,"train_embeddings_are_in_sample":True},
        "right_late_weight":[.5,.5],"threshold":.5,"test_npy_opened":0,"test_seg_npz_opened":0,
        "fold_evaluation_policy":"outer never used for epoch selection/scalers/head fit"}


def pose_prob(model,x,device):
    model.eval();prob=[];lossfn=nn.BCELoss()
    with torch.no_grad():
        for i in range(0,len(x),128):
            xx=torch.as_tensor(x[i:i+128],device=device).permute(0,3,1,2).contiguous()
            prob.extend(model(xx).reshape(-1).cpu().numpy().tolist())
    return np.asarray(prob,np.float64)


def pose_loss(model,x,y,device):
    model.eval();lossfn=nn.BCELoss();total=0.
    with torch.no_grad():
        for i in range(0,len(y),16):
            xx=torch.as_tensor(x[i:i+16],device=device).permute(0,3,1,2).contiguous()
            yy=torch.as_tensor(y[i:i+16],dtype=torch.float32,device=device).unsqueeze(1)
            total+=float(lossfn(model(xx),yy).item())*len(yy)
    return total/len(y)


def train_pose(x,y,fit,val,outer,fold_dir,device):
    path=fold_dir/"pose";path.mkdir(exist_ok=True)
    scaler=StandardScaler().fit(x[fit].reshape(len(fit),-1))
    if int(scaler.n_samples_seen_)!=len(fit):raise RuntimeError("Pose scaler usa muestras externas")
    with open(path/"scaler_inner_train.pkl","wb") as f:pickle.dump(scaler,f)
    aug,labels=augment_dataset(x[fit],y[fit],np.random.RandomState(42),3,
                               transforms=("translation","scaling","noise","temporal_shift"))
    if not np.array_equal(aug[::4],x[fit]) or len(aug)!=4*len(fit):
        raise RuntimeError("Pose augmentation fuera de inner train")
    xtrain=transform(scaler,aug);xval=transform(scaler,x[val])
    pose_seed_all(42)
    model=PoseCNN(T=170,K=8,C=3).to(device)
    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4)
    dl=DataLoader(TensorDataset(torch.from_numpy(xtrain).permute(0,3,1,2).contiguous(),
                                torch.from_numpy(labels.astype(np.float32))),
                  batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    best=float("inf");wait=0;epoch_best=0;history=[];ckpt=path/"best_internal_validation.pt"
    for epoch in range(1,201):
        model.train();losses=[]
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward();optimizer.step()
            losses.append((float(loss.detach()),len(yb)))
        lval=pose_loss(model,xval,y[val],device)
        scheduler.step(lval)
        history.append({"epoch":epoch,"train_aug_loss":sum(a*b for a,b in losses)/len(xtrain),
                        "inner_validation_loss":lval,"lr":optimizer.param_groups[0]["lr"]})
        if lval<best:
            best=lval;epoch_best=epoch;wait=0
            torch.save({"state_dict":model.state_dict(),"best_epoch":epoch,"inner_validation_loss":best,
                        "inner_train_count":len(fit),"seed":42},ckpt)
        else:wait+=1
        if wait>=8:break
    state=torch.load(ckpt,map_location=device,weights_only=True)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    save_csv(path/"learning_curves.csv",history)
    outerx=transform(scaler,x[outer])
    return model,transform(scaler,x[fit]),transform(scaler,x[val]),outerx,{
        "best_epoch":epoch_best,"epochs_run":epoch,"inner_train_original":len(fit),
        "inner_train_augmented":len(aug),"inner_val_count":len(val),"outer_evaluation_count":len(outer),
        "inner_validation_loss":best,"scaler_n_samples_seen":int(scaler.n_samples_seen_),
        "scaler_sha256":sha(path/"scaler_inner_train.pkl"),"checkpoint_sha256":sha(ckpt)}


def seg_loss(model,x,y,device,amp):
    model.eval();criterion=nn.BCEWithLogitsLoss();total=0.
    with torch.no_grad():
        for i in range(0,len(y),128):
            xx=torch.as_tensor(x[i:i+128],device=device)
            yy=torch.as_tensor(y[i:i+128],dtype=torch.float32,device=device)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(xx)
                loss=criterion(logits.float(),yy)
            total+=float(loss.item())*len(yy)
    return total/len(y)


def seg_prob(model,x,device,amp):
    model.eval();ps=[]
    with torch.no_grad():
        for i in range(0,len(x),128):
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(torch.as_tensor(x[i:i+128],device=device))
            ps.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return np.asarray(ps,np.float64)


def train_seg(x,a,p,y,fit,val,outer,contract,fold_dir,device):
    path=fold_dir/"seg";path.mkdir(exist_ok=True)
    together=np.concatenate((val,outer))
    train_x,other,stats=seg_fold_transform(x,a,p,contract,fit,together)
    inner_x=other[:len(val)];outer_x=other[len(val):]
    if (train_x.shape!=(len(fit),contract["C"],170) or
        inner_x.shape[0]!=len(val) or outer_x.shape[0]!=len(outer)):
        raise RuntimeError("Forma de Seg fold inválida")
    save_json(path/"inner_train_preprocessing.json",{"fitted_on_inner_train_only":len(fit),
              "n_features":len(stats),"input_C_frozen":contract["C"],
              "selected_feature_names":contract["selected_feature_names"],
              "input_channel_names":contract["input_channel_names"],"statistics":stats,
              "outer_evaluation_not_used_in_fit":True})
    from training.bp_side_segcnn_v1.train import seed_all
    seed_all(42)
    model=SegCNNTemporal(contract["C"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    criterion=nn.BCEWithLogitsLoss();amp=device.type=="cuda"
    grad_scaler=torch.amp.GradScaler("cuda",enabled=amp)
    dl=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(y[fit].astype(np.float32))),
                  batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    best=float("inf");wait=0;epoch_best=0;history=[];ckpt=path/"best_internal_validation.pt"
    for epoch in range(1,201):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                loss=criterion(model(xb).float(),yb)
            grad_scaler.scale(loss).backward();grad_scaler.step(opt);grad_scaler.update()
        ltr=seg_loss(model,train_x,y[fit],device,amp)
        lval=seg_loss(model,inner_x,y[val],device,amp)
        if not np.isfinite(lval):raise RuntimeError("Seg validation loss no finita")
        scheduler.step(lval)
        history.append({"epoch":epoch,"train_loss":ltr,"inner_validation_loss":lval,
                        "lr":opt.param_groups[0]["lr"]})
        if lval<best-1e-7:
            best=lval;epoch_best=epoch;wait=0
            torch.save({"state_dict":model.state_dict(),"C":contract["C"],"best_epoch":epoch,
                        "inner_validation_loss":best,"seed":42},ckpt)
        else:wait+=1
        if wait>=15:break
    state=torch.load(ckpt,map_location=device,weights_only=True)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    save_csv(path/"learning_curves.csv",history)
    return model,train_x,inner_x,outer_x,{
        "best_epoch":epoch_best,"epochs_run":epoch,"inner_train_count":len(fit),
        "inner_val_count":len(val),"outer_evaluation_count":len(outer),
        "inner_validation_loss":best,"C_frozen":contract["C"],
        "preprocessing_sha256":sha(path/"inner_train_preprocessing.json"),
        "checkpoint_sha256":sha(ckpt)}


@torch.no_grad()
def get_embedding(pose,seg,xp,xs,device):
    pose.eval();seg.eval();pa=PoseEmbeddingAdapter(pose);z=[]
    for i in range(0,len(xp),128):
        zp=pa.extract_embeddings(torch.as_tensor(xp[i:i+128],device=device).permute(0,3,1,2).contiguous())
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
            zs=seg.temporal(torch.as_tensor(xs[i:i+128],device=device)).flatten(1)
        z.append(torch.cat((zp.float(),zs.float()),dim=1).cpu().numpy())
    result=np.concatenate(z).astype(np.float32)
    if result.shape!=(len(xp),192):raise RuntimeError("Embedding fold debe ser 64+128")
    return result


def head_loss(model,x,y,device):
    model.eval();criterion=nn.BCEWithLogitsLoss();total=0.;prob=[]
    with torch.no_grad():
        for i in range(0,len(y),256):
            xx=torch.as_tensor(x[i:i+256],device=device)
            yy=torch.as_tensor(y[i:i+256],dtype=torch.float32,device=device)
            logits=model(xx)
            total+=float(criterion(logits,yy))*len(yy)
            prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return total/len(y),np.asarray(prob,np.float64)


def train_head(pose,seg,xp_fit,xp_val,xp_outer,xs_fit,xs_val,xs_outer,y,fit,val,outer,fold_dir,device):
    path=fold_dir/"head";path.mkdir(exist_ok=True)
    ztr=get_embedding(pose,seg,xp_fit,xs_fit,device)
    zval=get_embedding(pose,seg,xp_val,xs_val,device)
    zouter=get_embedding(pose,seg,xp_outer,xs_outer,device)
    mean=ztr.mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(ztr.std(0,dtype=np.float64).astype(np.float32),1e-6)
    with open(path/"embedding_scaler_inner_train.npz","wb") as file:
        np.savez_compressed(file,mean=mean,std=std,n_fit=len(fit),pose_dim=64,seg_dim=128)
    ztr=(ztr-mean)/std;zval=(zval-mean)/std;zouter=(zouter-mean)/std
    if not np.isfinite(ztr).all() or not np.isfinite(zval).all() or not np.isfinite(zouter).all():
        raise RuntimeError("Embeddings mal escalados")
    from training.bp_side_fusion_v1.train import seed_all
    seed_all(1337)
    model=FusionHead(64,128).to(device)
    params=sum(q.numel() for q in model.parameters() if q.requires_grad)
    if params!=6209 or any(q.requires_grad for q in list(pose.parameters())+list(seg.parameters())):
        raise RuntimeError("Backbones no congelados / cabeza incorrecta")
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    criterion=nn.BCEWithLogitsLoss()
    dl=DataLoader(TensorDataset(torch.from_numpy(ztr),torch.from_numpy(y[fit].astype(np.float32))),
                  batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(1337),num_workers=0)
    best=float("inf");wait=0;epoch_best=0;history=[];ckpt=path/"best_internal_validation.pt"
    for epoch in range(1,101):
        model.train()
        for zb,yb in dl:
            zb,yb=zb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(zb),yb)
            loss.backward();opt.step()
        train_loss,_=head_loss(model,ztr,y[fit],device)
        val_loss,_=head_loss(model,zval,y[val],device)
        sched.step(val_loss)
        history.append({"epoch":epoch,"train_loss":train_loss,"inner_validation_loss":val_loss,
                        "lr":opt.param_groups[0]["lr"]})
        if val_loss<best-1e-7:
            best=val_loss;epoch_best=epoch;wait=0
            torch.save({"state_dict":model.state_dict(),"best_epoch":epoch,
                        "inner_validation_loss":best,"seed":1337,"params":params},ckpt)
        else:wait+=1
        if wait>=15:break
    state=torch.load(ckpt,map_location=device,weights_only=True)
    model.load_state_dict(state["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    save_csv(path/"learning_curves.csv",history)
    _,outerprob=head_loss(model,zouter,y[outer],device)
    return outerprob,{"best_epoch":epoch_best,"epochs_run":epoch,"inner_train_count":len(fit),
                      "inner_validation_count":len(val),"outer_eval_count":len(outer),
                      "inner_validation_loss":best,"embedding_train_in_sample":True,
                      "embedding_scaler_fit_count":len(fit),
                      "scaler_sha256":sha(path/"embedding_scaler_inner_train.npz"),
                      "checkpoint_sha256":sha(ckpt),"head_params":params}


def measure(y,p):
    m=base_metrics(y,p)
    m["FPR"]=m["FP"]/(m["FP"]+m["TN"]) if m["FP"]+m["TN"] else None
    m["FNR"]=m["FN"]/(m["FN"]+m["TP"]) if m["FN"]+m["TP"] else None
    return m


def run_fold(view,fold,arrays,contract,assignment,device):
    out=output(view);d=out/f"fold_{fold:02d}";d.mkdir(exist_ok=True)
    idx={str(vid):i for i,vid in enumerate(arrays["ids"].tolist())}
    fit=np.asarray([idx[r["video_id"]] for r in assignment if r["role"]=="inner_train"],int)
    inner=np.asarray([idx[r["video_id"]] for r in assignment if r["role"]=="inner_validation"],int)
    outer=np.asarray([idx[r["video_id"]] for r in assignment if r["role"]=="outer_evaluation"],int)
    if len(set(fit)&set(inner)) or len(set(fit)&set(outer)) or len(set(inner)&set(outer)):
        raise RuntimeError("Solapamiento de índices")
    y=arrays["y"]
    role_counts={name:{str(k):int(v) for k,v in Counter(y[ids].tolist()).items()} for name,ids in
                 (("inner_train",fit),("inner_validation",inner),("outer_eval",outer))}
    save_json(d/"roles_and_classes.json",{"fold":fold,"classes":role_counts,
              "inner_train_ids":arrays["ids"][fit].tolist(),"inner_validation_ids":arrays["ids"][inner].tolist(),
              "outer_eval_ids":arrays["ids"][outer].tolist(),"outer_not_used_for_early_stopping":True})
    pose,xptr,xpval,xpouter,pose_meta=train_pose(arrays["pose_original"],y,fit,inner,outer,d,device)
    seg,xstr,xsval,xsouter,seg_meta=train_seg(arrays["seg_values"],arrays["seg_availability"],
          arrays["seg_provenance"],y,fit,inner,outer,contract,d,device)
    p_pose=pose_prob(pose,xpouter,device)
    p_seg=seg_prob(seg,xsouter,device,device.type=="cuda")
    if view=="left":
        p_fusion,head_meta=train_head(pose,seg,xptr,xpval,xpouter,xstr,xsval,xsouter,
                                      y,fit,inner,outer,d,device)
        target="embedding_full"
    else:
        p_fusion=late_fusion(p_pose,p_seg)
        head_meta=None;target="late_sin_relacion"
    pred={"pose":p_pose,"seg_"+VARIANT[view]:p_seg,target:p_fusion}
    rows=[];summary={}
    for model,p in pred.items():
        summary[model]=measure(y[outer],p)
        np.save(d/f"confusion_{model}.npy",np.asarray(summary[model]["confusion_matrix_0_1"],np.int64))
        for vid,label,prob in zip(arrays["ids"][outer],y[outer],p):
            rows.append({"fold":fold,"model":model,"video_id":str(vid),"label":int(label),
                        "probability_valid":float(prob),"predicted":int(prob>=.5),
                        "error":int((prob>=.5)!=label),"false_positive":int(label==0 and prob>=.5)})
    save_csv(d/"outer_predictions.csv",rows)
    doc={"fold":fold,"view":view,"variant":VARIANT[view],"outer_n":len(outer),
         "inner_train_n":len(fit),"inner_validation_n":len(inner),
         "pose_training":pose_meta,"seg_training":seg_meta,"fusion_training":head_meta,
         "outer_metrics":summary,"outer_predictions_sha256":sha(d/"outer_predictions.csv"),
         "test_npz_opened":0,"backbones_initialized_from_scratch":True,
         "outer_never_used_for_selection":True}
    save_json(d/"fold_metrics.json",doc)
    print(view,"fold",fold,"outer",len(outer),"Pose F1",round(summary["pose"]["f1"],4),
          "Seg F1",round(summary["seg_"+VARIANT[view]]["f1"],4),
          "fusion F1",round(summary[target]["f1"],4),
          "epochs",pose_meta["best_epoch"],seg_meta["best_epoch"],
          (head_meta["best_epoch"] if head_meta else "fixed"),flush=True)


def train(view):
    out=output(view);manifest=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    if manifest["view"]!=view or snapshot(view)!=manifest["source_sha256"] or manifest["test_npy_opened"]!=0:
        raise RuntimeError("Fuente CV no coincide con manifest")
    if manifest["cv_code_sha256"]!={name:sha(Path(__file__).parent/name) for name in ("protocol.py","train.py")}:
        raise RuntimeError("Código CV cambió después de congelar folds")
    config=json.loads((out/"cv_config.json").read_text(encoding="utf-8"))
    if config!=CONFIG:raise RuntimeError("CV config alterada")
    with np.load(out/"development_unscaled.npz",allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    if sha(out/"development_unscaled.npz")!=manifest["development_array_sha256"]:
        raise RuntimeError("Development no escalado alterado")
    contract=frozen_contract(view)
    if arrays["seg_values"].shape!=(len(arrays["ids"]),170,62) or len(arrays["ids"])!=(513 if view=="left" else 524):
        raise RuntimeError("Forma development CV alterada")
    assignments=rows_csv(out/"fold_assignments.csv")
    if sha(out/"fold_assignments.csv")!=manifest["fold_assignments_sha256"]:
        raise RuntimeError("Folds congelados alterados")
    torch.set_num_threads(4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(1,6):
        d=out/f"fold_{fold:02d}";file=d/"fold_metrics.json"
        if file.exists():
            record=json.loads(file.read_text(encoding="utf-8"))
            if (record["outer_predictions_sha256"]!=sha(d/"outer_predictions.csv") or
                record["pose_training"]["checkpoint_sha256"]!=sha(d/"pose/best_internal_validation.pt") or
                record["seg_training"]["checkpoint_sha256"]!=sha(d/"seg/best_internal_validation.pt") or
                (view=="left" and record["fusion_training"]["checkpoint_sha256"]!=sha(d/"head/best_internal_validation.pt"))):
                raise RuntimeError("Fold existente inválido")
            print(view,"fold",fold,"ya completado/verificado",flush=True);continue
        run_fold(view,fold,arrays,contract,[r for r in assignments if int(r["fold"])==fold],device)
        if snapshot(view)!=manifest["source_sha256"]:
            raise RuntimeError("Checkpoint final alterado durante fold")
    save_json(out/"cv_training_complete.json",{"view":view,"folds":5,"source_sha256":manifest["source_sha256"],
              "test_npy_opened":0,"fold_checkpoints_are_fresh":True})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view",required=True,choices=("left","right"))
    ap.add_argument("--phase",required=True,choices=("prepare","train","all"))
    args=ap.parse_args()
    if args.phase in ("prepare","all"):
        prepare(args.view)
        save_json(output(args.view)/"cv_config.json",CONFIG)
    if args.phase in ("train","all"):train(args.view)


if __name__=="__main__":main()
