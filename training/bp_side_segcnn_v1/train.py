"""SegCNN LEFT/RIGHT independientes; ocho runs Train/Validation, Test NPZ cerrado.

python -m training.bp_side_segcnn_v1.train --view left --phase all
python -m training.bp_side_segcnn_v1.train --view right --phase all
"""
import argparse
import csv
import json
import platform
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from training.bp_front_segcnn_v1.model import SegCNNTemporal
from .data import (ROOT, RELATION_EXCLUDED, T_TARGET, cohort, load_development, paths,
                   prepare, sha)

CONFIG={"variants":["full","sin_relacion"],"seeds":[42,1337],"T":170,
        "batch":32,"max_epochs":200,"early_stopping_patience":15,
        "optimizer":"AdamW","learning_rate":.001,"weight_decay":1e-4,
        "scheduler_patience":5,"scheduler_factor":.5,"threshold":.5,
        "checkpoint_selection":"lowest validation BCE loss, first epoch on ties",
        "seed_selection":"lowest best validation loss; no test selection",
        "architecture":"training.bp_front_segcnn_v1.model.SegCNNTemporal; conv padding=2,2,1; pool 2",
        "grid":"linspace(0,T_original-1,170), same BP lateral PoseCNN; gated missing/provenance",
        "test_policy":"never open Test NPZ; descriptive validation only"}


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def save_csv(path,rows):
    if not rows: raise ValueError("CSV vacío")
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def metrics(y,p):
    pred=(p>=CONFIG["threshold"]).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "f1":float(f1_score(y,pred,zero_division=0)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
            "sensitivity_valid":tp/(tp+fn) if tp+fn else None,
            "specificity_invalid":tn/(tn+fp) if tn+fp else None,
            "TN":tn,"FP":fp,"FN":fn,"TP":tp,
            "confusion_matrix_0_1":[[tn,fp],[fn,tp]]}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def evaluate(model,x,y,device,loss_fn,amp):
    model.eval();probs=[];total=0.
    with torch.no_grad():
        for start in range(0,len(y),128):
            batch=torch.as_tensor(x[start:start+128],device=device)
            labels=torch.as_tensor(y[start:start+128],dtype=torch.float32,device=device)
            with torch.autocast(device_type=device.type,enabled=amp):
                logits=model(batch)
                loss=loss_fn(logits.float(),labels)
            total+=float(loss)*len(labels)
            probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return total/len(y),np.asarray(probs,dtype=np.float64)


def plots(history,y,p,cm,path):
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in ("train_loss","validation_loss"):ax[0].plot([h["epoch"] for h in history],[h[name] for h in history],label=name)
    for name in ("train_f1","validation_f1"):ax[1].plot([h["epoch"] for h in history],[h[name] for h in history],label=name)
    for a in ax:a.legend();a.grid(alpha=.2);a.set_xlabel("Epoch")
    fig.tight_layout();fig.savefig(path/"learning_curves.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,4));ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha="center",va="center")
    ax.set(xlabel="Predicho",ylabel="Real",xticks=[0,1],yticks=[0,1])
    fig.tight_layout();fig.savefig(path/"validation_confusion.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3))
    for label,name in ((0,"inválido"),(1,"válido")):
        ax.hist(p[y==label],bins=np.linspace(0,1,21),alpha=.5,label=name)
    ax.axvline(.5,color="black",linestyle="--");ax.legend();ax.set_xlabel("P(válido)")
    fig.tight_layout();fig.savefig(path/"validation_probabilities.png",dpi=130);plt.close(fig)


def prepare_view(view):
    groups,names,p,src=cohort(view)
    out=p["out"];out.mkdir(parents=True,exist_ok=True)
    data,hashes=load_development(groups,names,p)
    src.update({"train_validation_sequences_sha256":hashes,"train_ids":[r["video_id"] for r in groups["train"]],
                "validation_ids":[r["video_id"] for r in groups["validation"]],
                "labels_train_validation":dict(zip(data["train"][4],data["train"][3].tolist())) |
                                          dict(zip(data["validation"][4],data["validation"][3].tolist())),
                "code_sha256":{f:sha(Path(__file__).resolve().parent/f) for f in ("data.py","train.py")},
                "front_architecture_sha256":sha(ROOT/"training/bp_front_segcnn_v1/model.py"),
                "environment":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
                               "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}})
    save_json(out/"training_config.json",CONFIG)
    save_json(out/"source_manifest.json",src)
    for variant in CONFIG["variants"]:
        outdir=out/variant;outdir.mkdir(exist_ok=True)
        arrays,contract=prepare(data,names,variant)
        save_json(outdir/"channels_and_train_preprocessing.json",contract)
        with open(outdir/"prepared_development.npz","wb") as f:
            np.savez_compressed(f,train_X=arrays["train"][0],train_y=arrays["train"][1],
                                train_ids=np.asarray(arrays["train"][2]),
                                validation_X=arrays["validation"][0],validation_y=arrays["validation"][1],
                                validation_ids=np.asarray(arrays["validation"][2]))
        print(view,variant,"Train/Val",len(arrays["train"][1]),len(arrays["validation"][1]),
              "C",contract["C"],"constant_features",len(contract["excluded_constant_features_train"]),flush=True)
    print(view,"PREPARED; test_npz_opened=0; labels",Counter(data["train"][3]),Counter(data["validation"][3]),flush=True)


def validate_prepared(view):
    groups,names,p,src=cohort(view)
    out=p["out"]
    old=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    if (old["source_sha256"]!=src["source_sha256"] or
        old["code_sha256"]!={f:sha(Path(__file__).resolve().parent/f) for f in ("data.py","train.py")} or
        old["front_architecture_sha256"]!=sha(ROOT/"training/bp_front_segcnn_v1/model.py") or
        json.loads((out/"training_config.json").read_text(encoding="utf-8"))!=CONFIG):
        raise RuntimeError("Código/fuentes/config cambiados tras preparación")
    if set(old["train_ids"])!=set(r["video_id"] for r in groups["train"]) or set(old["validation_ids"])!=set(r["video_id"] for r in groups["validation"]):
        raise RuntimeError("Cohorte alterada")
    # No se abren NPZ Test. Se verifican únicamente hashes de desarrollo.
    for vid,h in old["train_validation_sequences_sha256"].items():
        if sha(p["source"]/"sequences"/f"{vid}_temporal.npz")!=h:raise RuntimeError("Fuente cambió: "+vid)
    result={}
    for variant in CONFIG["variants"]:
        directory=out/variant
        contract=json.loads((directory/"channels_and_train_preprocessing.json").read_text(encoding="utf-8"))
        with np.load(directory/"prepared_development.npz",allow_pickle=False) as z:
            result[variant]=({k:z[k] for k in z.files},contract)
        arr,_=result[variant]
        for role in ("train","validation"):
            if (arr[f"{role}_X"].shape!=(len(groups[role]),contract["C"],T_TARGET) or
                not np.array_equal(arr[f"{role}_ids"],old[f"{role}_ids"]) or
                not np.isfinite(arr[f"{role}_X"]).all() or
                not np.array_equal(arr[f"{role}_y"],[old["labels_train_validation"][vid] for vid in arr[f"{role}_ids"]])):
                raise RuntimeError("Tensor desarrollo inconsistente")
    return out,result


def train_run(view,variant,seed,out,prepared,contract,device):
    d=out/variant/f"seed_{seed}";d.mkdir(parents=True,exist_ok=True)
    xtr,ytr,ids_tr=(prepared[f"train_{k}"] for k in ("X","y","ids"))
    xva,yva,ids_va=(prepared[f"validation_{k}"] for k in ("X","y","ids"))
    seed_all(seed)
    model=SegCNNTemporal(contract["C"]).to(device)
    param=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=CONFIG["scheduler_factor"],
                                                          patience=CONFIG["scheduler_patience"])
    loss_fn=torch.nn.BCEWithLogitsLoss();amp=device.type=="cuda"
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr.astype(np.float32))),
                      batch_size=CONFIG["batch"],shuffle=True,generator=generator,num_workers=0)
    best=float("inf");best_ep=0;wait=0;history=[];checkpoint=d/"best_validation_loss.pt"
    for ep in range(1,CONFIG["max_epochs"]+1):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):logits=model(xb);loss=loss_fn(logits.float(),yb)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        tr_loss,tr_p=evaluate(model,xtr,ytr,device,loss_fn,amp)
        va_loss,va_p=evaluate(model,xva,yva,device,loss_fn,amp)
        if not np.isfinite(va_loss):raise RuntimeError("Validation loss no finita")
        scheduler.step(va_loss)
        history.append({"epoch":ep,"train_loss":tr_loss,"validation_loss":va_loss,
                        "train_f1":metrics(ytr,tr_p)["f1"],"validation_f1":metrics(yva,va_p)["f1"],
                        "lr":optimizer.param_groups[0]["lr"]})
        if va_loss<best-1e-7:
            best=va_loss;best_ep=ep;wait=0
            torch.save({"state_dict":model.state_dict(),"channels_total":contract["C"],
                        "input_channels":contract["input_channel_names"],"seed":seed,"epoch":ep,
                        "variant":variant,"view":view,"validation_loss":best,"threshold":.5},checkpoint)
        else:wait+=1
        if ep==1 or ep%10==0:print(view,variant,seed,"ep",ep,"train",round(tr_loss,4),"val",round(va_loss,4),flush=True)
        if wait>=CONFIG["early_stopping_patience"]:break
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    if (state["view"],state["variant"],state["seed"],state["channels_total"])!=(view,variant,seed,contract["C"]):
        raise RuntimeError("Checkpoint equivocado")
    model.load_state_dict(state["state_dict"])
    tr_loss,tr_p=evaluate(model,xtr,ytr,device,loss_fn,amp)
    va_loss,va_p=evaluate(model,xva,yva,device,loss_fn,amp)
    mt,mv=metrics(ytr,tr_p),metrics(yva,va_p)
    result={"view":view,"variant":variant,"seed":seed,"parameters":param,"C":contract["C"],
            "best_epoch":best_ep,"epochs_run":ep,"train_loss":tr_loss,"validation_loss":va_loss,
            "train_metrics":mt,"validation_metrics":mv,"checkpoint_sha256":sha(checkpoint),
            "checkpoint":str(checkpoint.relative_to(out)),"positive_class":"1=valido",
            "threshold":.5,"train_minus_validation_f1":mt["f1"]-mv["f1"],
            "validation_minus_train_loss":va_loss-tr_loss,"test_npz_opened":0}
    save_json(d/"metrics.json",result)
    save_csv(d/"learning_curves.csv",history)
    save_csv(d/"validation_predictions.csv",[{"video_id":vid,"label":int(y),"probability_valid":float(p),
               "predicted":int(p>=.5),"error":int((p>=.5)!=y),"false_positive":int(y==0 and p>=.5)}
               for vid,y,p in zip(ids_va,yva,va_p)])
    np.save(d/"validation_confusion.npy",np.array(mv["confusion_matrix_0_1"],dtype=np.int64))
    plots(history,yva,va_p,mv["confusion_matrix_0_1"],d)
    print(view,variant,seed,"best_ep",best_ep,"val_loss",round(va_loss,4),"F1",round(mv["f1"],4),flush=True)
    return result


def train_view(view):
    torch.set_num_threads(4)
    out,prepared=validate_prepared(view)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant,(arr,contract) in prepared.items():
        for seed in CONFIG["seeds"]:
            d=out/variant/f"seed_{seed}";metrics_file=d/"metrics.json"
            if metrics_file.exists():
                previous=json.loads(metrics_file.read_text(encoding="utf-8"))
                checkpoint=d/"best_validation_loss.pt"
                if ((previous["view"],previous["variant"],previous["seed"],previous["C"])!=(view,variant,seed,contract["C"]) or
                    not checkpoint.exists() or previous["checkpoint_sha256"]!=sha(checkpoint) or
                    not (d/"validation_predictions.csv").exists()):
                    raise RuntimeError("Run existente incompatible, no sobrescribir: "+str(d))
                print(view,variant,seed,"checkpoint previo verificado",flush=True);continue
            train_run(view,variant,seed,out,arr,contract,device)
        scores=[json.loads((out/variant/f"seed_{seed}"/"metrics.json").read_text(encoding="utf-8")) for seed in CONFIG["seeds"]]
        winner=min(scores,key=lambda r:(r["validation_loss"],r["seed"]))
        save_json(out/variant/"seeds_metrics.json",scores)
        save_json(out/variant/"selected_checkpoint.json",{"seed":winner["seed"],"checkpoint":winner["checkpoint"],
                  "selection":"minimum Validation loss, seed tie ascending","validation_loss":winner["validation_loss"],
                  "checkpoint_sha256":winner["checkpoint_sha256"]})
    save_json(out/"training_complete.json",{"view":view,"variants":CONFIG["variants"],"seeds":CONFIG["seeds"],
              "test_npz_opened":0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view",required=True,choices=("left","right"))
    parser.add_argument("--phase",required=True,choices=("prepare","train","all"))
    args=parser.parse_args()
    if args.phase in ("prepare","all"):prepare_view(args.view)
    if args.phase in ("train","all"):train_view(args.view)


if __name__=="__main__":main()
