"""python -m training.bp.experiments.bp_right_02_posecnn_common_v1.train --phase all|prepare|train

Replica el baseline histórico lateral; únicamente Train/Validation actuales.
"""
import argparse
import csv
import json
import pickle
import platform
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.bp.train_posecnn import seed_all, transform
from training.bp.experiments.bp_side_segcnn_v1.train import metrics
from training.common.train_utils import PoseCNN, augment_dataset
from training.bp.experiments.bp_posecnn_common_v1.core import (ROOT, load_development, paths, predict,
                                                save_csv, save_json, sha, verify_embedding)

HERE=ROOT/"training/bp/experiments/bp_right_02_posecnn_common_v1"
CONFIG={"view":"right","cohort":{"train":431,"validation":93,"test_ids_only":93,"unassigned":12},
        "excluded_duplicate":"bp_354", "duplicate_of":"bp_027", "seed_values":[42,1337],
        "input_order":"(N,3,170,8) x,y,confidence", "T":170,"K":8,"C":3,
        "model":"training.common.train_utils.PoseCNN; same kernels/padding/layers as historical baseline",
        "loss":"BCELoss on sigmoid output", "optimizer":"Adam", "learning_rate":.001,
        "weight_decay":.0001,"batch_size":16,"max_epochs":200,"patience":8,
        "scheduler":"ReduceLROnPlateau factor=.5 patience=4 on weighted Validation loss",
        "augmentation":"train x4 including original; translation,scaling,noise,temporal_shift; no flips/rotations",
        "scaler":"StandardScaler.fit(original 431 Train, flattened 170*8*3); transform augmented train and Validation",
        "normalization":"historical preprocess: median shoulder 3 center and shoulder 3 to hip 2 median distance per video",
        "missing":"historical: interpolate XY gaps <=5 interior only; missing confidence=0",
        "grid":"linspace(0,T_original-1,170) same SegCNN and historical PoseCNN",
        "checkpoint":"lowest weighted Validation BCELoss; first epoch if tie", "threshold":.5,
        "selection":"lowest Validation loss between seeds", "test_npy_opened":0}


def prepare():
    HERE.mkdir(parents=True,exist_ok=True)
    p=paths("right");loaded,source,records=load_development("right")
    if source["historical_minus_current"]!={"train":["bp_354"],"validation":[]}:
        raise RuntimeError("Cohorte RIGHT no excluye únicamente bp_354")
    original,y,train_ids=loaded["train"]
    val,yv,val_ids=loaded["validation"]
    if len(train_ids)!=431 or len(val_ids)!=93 or "bp_354" in train_ids or "bp_027" not in train_ids:
        raise RuntimeError("Split común alterado")
    scaler=StandardScaler().fit(original.reshape(len(original),-1))
    if scaler.n_samples_seen_!=431 or scaler.n_features_in_!=170*8*3:
        raise RuntimeError("Scaler no ajustado a 431 originales")
    with open(HERE/"scaler_train_original.pkl","wb") as file:pickle.dump(scaler,file)
    with open(HERE/"development_original.npz","wb") as file:
        np.savez_compressed(file,X_train=original,y_train=y,ids_train=np.asarray(train_ids),
                            X_validation=val,y_validation=yv,ids_validation=np.asarray(val_ids))
    save_csv(HERE/"train_validation_source_audit.csv",records)
    historical={"checkpoint_sha256":sha(p["historical_ckpt"]),
                "scaler_sha256":sha(p["historical_scaler"]),
                "history_sha256":sha(p["historical_history"]),
                "config_sha256":sha(p["historical_config"]),
                "historical_train":432,"excluded_duplicate":"bp_354","new_train":431,
                "historical_checkpoint_never_loaded_into_new_model":True}
    source.update({"historical_documentation":historical,
                   "new_scaler_sha256":sha(HERE/"scaler_train_original.pkl"),
                   "prepared_original_sha256":sha(HERE/"development_original.npz"),
                   "code_sha256":{"train.py":sha(Path(__file__)),
                                  "core.py":sha(ROOT/"training/bp/experiments/bp_posecnn_common_v1/core.py")},
                   "environment":{"torch":torch.__version__,"numpy":np.__version__,
                                  "python":platform.python_version(),
                                  "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}})
    save_json(HERE/"config.json",CONFIG)
    save_json(HERE/"source_manifest.json",source)
    save_json(HERE/"scaler_audit.json",{"scaler_sha256":source["new_scaler_sha256"],
              "fit_count_original_train":int(scaler.n_samples_seen_),
              "features":int(scaler.n_features_in_),"never_fit_augmented":True,
              "never_fit_validation_test":True,"historical_scaler_not_reused":True})
    print("right prepared Train",len(train_ids),"Validation",len(val_ids),
          "labels",Counter(y),Counter(yv),"bp_354 excluded",flush=True)


def verify_prepared():
    p=paths("right")
    source=json.loads((HERE/"source_manifest.json").read_text(encoding="utf-8"))
    if json.loads((HERE/"config.json").read_text(encoding="utf-8"))!=CONFIG or source["code_sha256"]!={
        "train.py":sha(Path(__file__)),"core.py":sha(ROOT/"training/bp/experiments/bp_posecnn_common_v1/core.py")}:
        raise RuntimeError("Código/config cambiados tras preparar")
    _,new_source,_=load_development("right")
    for k,h in source["source_sha256"].items():
        if h!=new_source["source_sha256"][k]:raise RuntimeError("Fuente cambió: "+k)
    for vid,h in source["train_validation_npy_sha256"].items():
        if sha(p["keypoints"]/f"{vid}.npy")!=h:raise RuntimeError("NPY desarrollo cambió")
    if (sha(HERE/"scaler_train_original.pkl")!=source["new_scaler_sha256"] or
        sha(HERE/"development_original.npz")!=source["prepared_original_sha256"]):
        raise RuntimeError("Datos preparados/scaler cambiaron")
    with np.load(HERE/"development_original.npz",allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    if (d["X_train"].shape!=(431,170,8,3) or d["X_validation"].shape!=(93,170,8,3) or
        "bp_354" in d["ids_train"] or "bp_027" not in d["ids_train"] or
        not np.isfinite(d["X_train"]).all() or not np.isfinite(d["X_validation"]).all()):
        raise RuntimeError("Array de desarrollo incompatible")
    with open(HERE/"scaler_train_original.pkl","rb") as file:scaler=pickle.load(file)
    recomputed=StandardScaler().fit(d["X_train"].reshape(431,-1))
    if (scaler.n_samples_seen_!=431 or not np.allclose(scaler.mean_,recomputed.mean_,rtol=0,atol=1e-10) or
        not np.allclose(scaler.scale_,recomputed.scale_,rtol=0,atol=1e-10)):
        raise RuntimeError("Scaler no reproduce train")
    return d,scaler,source


def eval_loss(model,X,y,device):
    model.eval();crit=nn.BCELoss();total=0.;prob=[]
    with torch.no_grad():
        for a in range(0,len(y),16):
            xb=torch.as_tensor(X[a:a+16],device=device).permute(0,3,1,2).contiguous()
            yb=torch.as_tensor(y[a:a+16],dtype=torch.float32,device=device).unsqueeze(1)
            out=model(xb)
            total+=float(crit(out,yb))*len(yb)
            prob.extend(out.cpu().numpy().reshape(-1).tolist())
    return total/len(y),np.asarray(prob,dtype=np.float64)


def plot(history,cm,out):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for key in ("train_loss_augmented","validation_loss"):
        axes[0].plot([r["epoch"] for r in history],[r[key] for r in history],label=key)
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(cm[i][j]),ha="center",va="center")
    axes[1].set(xlabel="Predicción",ylabel="Real",xticks=[0,1],yticks=[0,1])
    fig.tight_layout();fig.savefig(out/"curves_and_confusion.png",dpi=130);plt.close(fig)


def run_seed(seed,data,scaler,device):
    path=HERE/f"seed_{seed}";path.mkdir(parents=True,exist_ok=True)
    x,y,ids=data["X_train"],data["y_train"],data["ids_train"]
    xv,yv,idv=data["X_validation"],data["y_validation"],data["ids_validation"]
    seed_all(seed)
    # No se carga ni inicializa a partir del checkpoint histórico.
    model=PoseCNN(T=170,K=8,C=3).to(device)
    start_sha=sha(HERE/"scaler_train_original.pkl")
    aug,labels=augment_dataset(x,y,np.random.RandomState(seed),3,
                               transforms=("translation","scaling","noise","temporal_shift"))
    if aug.shape!=(431*4,170,8,3) or not np.array_equal(labels[::4],y) or not np.array_equal(aug[::4],x):
        raise RuntimeError("Augmentation incluye orígenes en posiciones inesperadas")
    X_fit=transform(scaler,aug)
    X_train=transform(scaler,x)
    X_val=transform(scaler,xv)
    loader=DataLoader(TensorDataset(torch.from_numpy(X_fit).permute(0,3,1,2).contiguous(),
                                     torch.from_numpy(labels.astype(np.float32))),
                      batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4)
    best=float("inf");wait=0;best_ep=0;history=[];checkpoint=path/"best_model.pth"
    for ep in range(1,201):
        model.train();losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward();optimizer.step()
            losses.append((float(loss),len(yb)))
        val_loss,prob=eval_loss(model,X_val,yv,device)
        scheduler.step(val_loss)
        train_loss=sum(loss*n for loss,n in losses)/len(X_fit)
        if not np.isfinite(val_loss):raise RuntimeError("Val loss no finita")
        history.append({"epoch":ep,"train_loss_augmented":train_loss,"validation_loss":val_loss,"lr":optimizer.param_groups[0]["lr"]})
        if val_loss<best:
            best=val_loss;best_ep=ep;wait=0
            torch.save(model.state_dict(),checkpoint)
        else:
            wait+=1
        if ep==1 or ep%10==0:
            print("right pose",seed,"epoch",ep,"train",round(train_loss,4),"val",round(val_loss,4),flush=True)
        if wait>=8:break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    train_loss,ptr=eval_loss(model,X_train,y,device)
    val_loss,pv=eval_loss(model,X_val,yv,device)
    mt,mv=metrics(y,ptr),metrics(yv,pv)
    embedding=verify_embedding(model,X_val,device)
    save_json(path/"embeddings_audit.json",embedding)
    result={"view":"right","seed":seed,"best_epoch":best_ep,"epochs_run":ep,
            "best_validation_loss":val_loss,"train_original_loss":train_loss,
            "train_original_metrics":mt,"validation_metrics":mv,"parameters":sum(q.numel() for q in model.parameters() if q.requires_grad),
            "initialization":"fresh torch random with seed, no historical model/BatchNorm/optimizer loaded",
            "train_original_videos":431,"train_augmented_sequences":len(aug),"validation_videos":93,
            "scaler_sha256":start_sha,"checkpoint_sha256":sha(checkpoint),
            "test_npy_opened":0,"threshold":.5}
    save_json(path/"metrics.json",result)
    save_csv(path/"history.csv",history)
    save_csv(path/"validation_predictions.csv",[{"video_id":vid,"label":int(target),"probability_valid":float(prob),
              "predicted":int(prob>=.5),"error":int((prob>=.5)!=target),
              "false_positive":int(target==0 and prob>=.5)} for vid,target,prob in zip(idv,yv,pv)])
    np.save(path/"validation_confusion.npy",np.array(mv["confusion_matrix_0_1"],dtype=np.int64))
    plot(history,mv["confusion_matrix_0_1"],path)
    print("right seed",seed,"best epoch",best_ep,"val_loss",round(val_loss,4),
          "F1",round(mv["f1"],4),"CM",mv["confusion_matrix_0_1"],flush=True)
    return result


def train():
    torch.set_num_threads(4)
    data,scaler,src=verify_prepared()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    for seed in CONFIG["seed_values"]:
        path=HERE/f"seed_{seed}"
        file=path/"metrics.json"
        if file.exists():
            r=json.loads(file.read_text(encoding="utf-8"))
            if (r["seed"]!=seed or r["checkpoint_sha256"]!=sha(path/"best_model.pth") or
                r["scaler_sha256"]!=src["new_scaler_sha256"] or r["train_original_videos"]!=431):
                raise RuntimeError("Seed preexistente incompatible")
            results.append(r);print("right seed",seed,"verificado y conservado",flush=True)
        else:results.append(run_seed(seed,data,scaler,device))
    chosen=min(results,key=lambda r:(r["best_validation_loss"],r["seed"]))
    save_json(HERE/"selected_checkpoint.json",{"seed":chosen["seed"],"checkpoint":f'seed_{chosen["seed"]}/best_model.pth',
             "checkpoint_sha256":chosen["checkpoint_sha256"],"validation_loss":chosen["best_validation_loss"],
             "selection":"lowest Validation loss between seeds; no Test"})
    save_json(HERE/"training_complete.json",{"status":"RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE",
              "seeds":[x["seed"] for x in results],"selected_seed":chosen["seed"],"test_npy_opened":0})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",required=True,choices=("prepare","train","all"))
    args=ap.parse_args()
    if args.phase in ("prepare","all"):prepare()
    if args.phase in ("train","all"):train()


if __name__=="__main__":main()
