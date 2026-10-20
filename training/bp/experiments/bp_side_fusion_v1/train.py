"""Ocho cabezas de embeddings; backbones preextraídos y congelados."""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.bp.experiments.bp_posecnn_common_v1.core import save_csv, save_json
from training.bp.experiments.bp_side_segcnn_v1.data import sha
from training.bp.experiments.bp_side_segcnn_v1.train import metrics
from .models import FusionHead
from .data import fusion_out, snapshot

CONFIG={"variants":["full","sin_relacion"],"seeds":[42,1337],"late_weights":[.5,.5],
        "pose_dim":64,"seg_dim":128,"embedding_dim":192,"head_params":6209,
        "batch_size":32,"max_epochs":100,"patience":15,"scheduler_patience":5,
        "scheduler_factor":.5,"optimizer":"AdamW","lr":.001,"weight_decay":.001,
        "dropout":.40,"threshold":.5,"loss":"BCEWithLogitsLoss",
        "epoch_selection":"lowest Validation loss, first epoch for ties",
        "seed_selection":"lowest best Validation loss; no Test",
        "candidate_selection":{"primary":"macro_f1","secondary":"specificity_invalid",
                               "tertiary":"balanced_accuracy","exact_tie":"simpler model",
                               "threshold":.5},
        "no_backbone_finetune":True,"train_embeddings_in_sample":True,
        "test_npy_opened":0,"test_features_opened":0}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def evaluate(model,z,y,device,loss_fn):
    model.eval();prob=[];total=0.
    with torch.no_grad():
        for a in range(0,len(y),256):
            xx=torch.as_tensor(z[a:a+256],device=device)
            yy=torch.as_tensor(y[a:a+256],dtype=torch.float32,device=device)
            logits=model(xx)
            total+=float(loss_fn(logits,yy).item())*len(yy)
            prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return total/len(y),np.asarray(prob,dtype=np.float64)


def draw(history,y,probs,cm,path):
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in ("train_loss","validation_loss"):
        ax[0].plot([r["epoch"] for r in history],[r[name] for r in history],label=name)
    for name in ("train_f1","validation_f1"):
        ax[1].plot([r["epoch"] for r in history],[r[name] for r in history],label=name)
    for a in ax:a.grid(alpha=.2);a.legend();a.set_xlabel("Epoch")
    fig.tight_layout();fig.savefig(path/"learning_curves.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(4,4));ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i][j]),ha="center",va="center")
    ax.set(xlabel="Predicho",ylabel="Real",xticks=[0,1],yticks=[0,1]);fig.tight_layout()
    fig.savefig(path/"validation_confusion.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3))
    for label,name in ((0,"inválido"),(1,"válido")):
        ax.hist(probs[y==label],bins=np.linspace(0,1,21),alpha=.5,label=name)
    ax.axvline(.5,color="black",linestyle="--");ax.legend();fig.tight_layout()
    fig.savefig(path/"validation_probabilities.png",dpi=130);plt.close(fig)


def load_embeddings(view,variant):
    base=fusion_out(view)/f"embedding_fusion_{variant}"
    contract=json.loads((base/"embedding_contract.json").read_text(encoding="utf-8"))
    if (contract["view"]!=view or contract["variant"]!=variant or contract["concat_dim"]!=192 or
        contract["head_params"]!=6209 or contract["embeddings_sha256"]!=sha(base/"embeddings_development.npz") or
        contract["scaler_sha256"]!=sha(base/"embedding_scaler_train.npz") or
        not contract["pose_backbone_frozen"] or not contract["seg_backbone_frozen"]):
        raise RuntimeError("Contrato embeddings incompleto")
    with np.load(base/"embedding_scaler_train.npz",allow_pickle=False) as sc:
        if int(sc["n_samples_train"])!=(422 if view=="left" else 431) or sc["mean"].shape!=(192,) or sc["std"].shape!=(192,):
            raise RuntimeError("Scaler embeddings no ajustado solo Train")
    with np.load(base/"embeddings_development.npz",allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if (data["train_Z"].shape!=((422 if view=="left" else 431),192) or
        data["validation_Z"].shape!=((91 if view=="left" else 93),192) or
        not np.isfinite(data["train_Z"]).all() or not np.isfinite(data["validation_Z"]).all() or
        len(set(data["train_ids"].tolist()))!=len(data["train_ids"])):
        raise RuntimeError("Tensores embeddings incompletos")
    return data,contract


def run_seed(view,variant,seed,data,contract,device):
    parent=fusion_out(view)/f"embedding_fusion_{variant}";directory=parent/f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    seed_all(seed)
    head=FusionHead(64,128).to(device)
    n=sum(p.numel() for p in head.parameters() if p.requires_grad)
    if n!=6209:raise RuntimeError("Cabeza difiere de protocolo")
    ztr,ytr=data["train_Z"],data["train_y"]
    zva,yva=data["validation_Z"],data["validation_y"]
    if set(ytr.tolist())!={0,1} or set(yva.tolist())!={0,1}:
        raise RuntimeError("Faltan clases")
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=5,factor=.5)
    loss_fn=nn.BCEWithLogitsLoss()
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(ztr),torch.from_numpy(ytr.astype(np.float32))),
                      batch_size=32,shuffle=True,generator=generator,num_workers=0)
    best=float("inf");best_ep=0;wait=0;history=[];file=directory/"best_validation_loss.pt"
    for ep in range(1,101):
        head.train()
        for zb,yb in loader:
            zb,yb=zb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(head(zb),yb)
            loss.backward();optimizer.step()
        ltr,ptr=evaluate(head,ztr,ytr,device,loss_fn)
        lva,pva=evaluate(head,zva,yva,device,loss_fn)
        if not np.isfinite(lva):raise RuntimeError("Validation loss no finita")
        scheduler.step(lva)
        history.append({"epoch":ep,"train_loss":ltr,"validation_loss":lva,
                        "train_f1":metrics(ytr,ptr)["f1"],"validation_f1":metrics(yva,pva)["f1"],
                        "lr":optimizer.param_groups[0]["lr"]})
        if lva<best-1e-7:
            best=lva;best_ep=ep;wait=0
            torch.save({"head_state_dict":head.state_dict(),"view":view,"variant":variant,
                        "seed":seed,"epoch":ep,"validation_loss":best,"threshold":.5,
                        "embedding_scaler_sha256":contract["scaler_sha256"],
                        "embedding_development_sha256":contract["embeddings_sha256"],
                        "head_params":n},file)
        else:wait+=1
        if ep==1 or ep%10==0:
            print(view,variant,"head",seed,"epoch",ep,"train_loss",round(ltr,4),
                  "val_loss",round(lva,4),flush=True)
        if wait>=15:break
    state=torch.load(file,map_location=device,weights_only=True)
    if (state["view"],state["variant"],state["seed"],state["head_params"])!=(view,variant,seed,6209):
        raise RuntimeError("Checkpoint cabeza ajena")
    head.load_state_dict(state["head_state_dict"],strict=True)
    ltr,ptr=evaluate(head,ztr,ytr,device,loss_fn)
    lva,pva=evaluate(head,zva,yva,device,loss_fn)
    result={"view":view,"variant":variant,"seed":seed,"head_trainable_parameters":n,
            "pose_trainable_parameters":0,"seg_trainable_parameters":0,
            "best_epoch":best_ep,"epochs_run":ep,"train_loss":ltr,"validation_loss":lva,
            "train_metrics":metrics(ytr,ptr),"validation_metrics":metrics(yva,pva),
            "train_minus_validation_f1":metrics(ytr,ptr)["f1"]-metrics(yva,pva)["f1"],
            "validation_minus_train_loss":lva-ltr,
            "checkpoint_sha256":sha(file),"embedding_scaler_sha256":contract["scaler_sha256"],
            "test_npy_opened":0}
    save_json(directory/"metrics.json",result)
    save_csv(directory/"learning_curves.csv",history)
    save_csv(directory/"validation_predictions.csv",[{"video_id":str(vid),"label":int(y),
        "probability_valid":float(p),"predicted":int(p>=.5),"error":int((p>=.5)!=y),
        "false_positive":int(y==0 and p>=.5)}
        for vid,y,p in zip(data["validation_ids"],yva,pva)])
    np.save(directory/"validation_confusion.npy",np.asarray(result["validation_metrics"]["confusion_matrix_0_1"],dtype=np.int64))
    draw(history,yva,pva,result["validation_metrics"]["confusion_matrix_0_1"],directory)
    print(view,variant,"seed",seed,"best",best_ep,"loss",round(lva,4),"F1",round(result["validation_metrics"]["f1"],4),flush=True)
    return result


def train_view(view):
    torch.set_num_threads(4)
    out=fusion_out(view)
    source=json.loads((out/"fusion_source_manifest.json").read_text(encoding="utf-8"))
    if (source["view"]!=view or source["source_sha256"]!=snapshot(view) or source["test_npy_opened"]!=0 or
        json.loads((out/"fusion_config.json").read_text(encoding="utf-8"))!=CONFIG or
        source["code_sha256"]!={name:sha(Path(__file__).parent/name) for name in ("models.py","data.py","train.py")}):
        raise RuntimeError("Fuentes/config cambian después de preparación")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant in CONFIG["variants"]:
        data,contract=load_embeddings(view,variant)
        scores=[]
        for seed in CONFIG["seeds"]:
            directory=out/f"embedding_fusion_{variant}"/f"seed_{seed}"
            old=directory/"metrics.json"
            if old.exists():
                result=json.loads(old.read_text(encoding="utf-8"))
                if (result["checkpoint_sha256"]!=sha(directory/"best_validation_loss.pt") or
                    result["embedding_scaler_sha256"]!=contract["scaler_sha256"] or
                    (result["view"],result["variant"],result["seed"])!=(view,variant,seed)):
                    raise RuntimeError("Run existente incompatible")
                print(view,variant,seed,"checkpoint anterior verificado",flush=True)
            else:result=run_seed(view,variant,seed,data,contract,device)
            scores.append(result)
        chosen=min(scores,key=lambda x:(x["validation_loss"],x["seed"]))
        parent=out/f"embedding_fusion_{variant}"
        save_json(parent/"seeds_metrics.json",scores)
        save_json(parent/"selected_checkpoint.json",{"seed":chosen["seed"],
                  "checkpoint":f'seed_{chosen["seed"]}/best_validation_loss.pt',
                  "validation_loss":chosen["validation_loss"],"checkpoint_sha256":chosen["checkpoint_sha256"],
                  "selection":"minimum Validation loss then seed ascending"})
    if snapshot(view)!=source["source_sha256"]:raise RuntimeError("Fuentes protegidas mutaron durante entrenamiento")
    save_json(out/"heads_training_complete.json",{"status":"HEADS_TRAINED_VALIDATION_ONLY",
              "view":view,"runs":4,"backbones_frozen":True,"test_npy_opened":0})
