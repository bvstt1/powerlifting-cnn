"""Evaluación final monovista congelada. No entrenamiento ni combinación LEFT/RIGHT.

python -B -m training.bp.experiments.bp_lateral_final_test_v1.run --view left --stage freeze
python -B -m training.bp.experiments.bp_lateral_final_test_v1.run --view left --stage validation
python -B -m training.bp.experiments.bp_lateral_final_test_v1.run --view left --stage test
"""
import argparse
import datetime
import json
import os
import pickle
import platform
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix)
from training.bp.train_posecnn import preprocess, transform, POINTS
from training.bp.experiments.bp_posecnn_common_v1.core import paths, rows_csv
from training.bp.experiments.bp_side_segcnn_v1.data import cohort, resample, sha
from training.bp.experiments.bp_side_fusion_v1.data import snapshot, fusion_out
from training.bp.experiments.bp_side_fusion_v1.models import PoseEmbeddingAdapter, SegEmbeddingAdapter, FusionHead
from training.bp.experiments.bp_front_01_segcnn_v1.model import SegCNNTemporal
from training.common.train_utils import PoseCNN

ROOT=Path(__file__).resolve().parents[4]
LIMITACIONES=[
    "Test lateral expuesto históricamente a evaluaciones PoseCNN; no totalmente inédito para el proyecto.",
    "Independencia de atletas, sesiones y eventos físicos no acreditada; vistas no emparejadas por numeración.",
    "Segmentación parcialmente asistida por keypoints Pose: modalidades no independientes.",
    "Proyecciones barra–torso 2D no demuestran contacto físico.",
    "Validation reutilizado para seleccionar checkpoints, variantes y cabezas.",
    "CV retrospectiva de robustez, no selección completamente anidada previa a exposición Test.",
    "Independencia de datasets históricos YOLO respecto a evaluación no plenamente acreditada.",
    "IC exploratorios por video suponen independencia no acreditada; no corrigen selección ni comparaciones múltiples."]


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def outdir(view): return ROOT/"training/bp/experiments"/f"bp_{view}_05_final_test_v1"
def readj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def savej(p,obj): Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def models(view): return ["pose","seg_full","late_full","embedding_full"] if view=="left" else ["pose","seg_sin_relacion","late_sin_relacion"]


def protect_inputs(view,after=False):
    """Hash de bytes sin decodificar, únicamente DESPUÉS del congelamiento."""
    verify(view)
    out=outdir(view);p=paths(view)
    folders=[ROOT/"dataset/bp"/view,p["keypoints"],p["source"].parent/"raw",p["source"]]
    hashes={}
    for folder in folders:
        assert folder.is_dir(),str(folder)
        for file in sorted(folder.rglob("*")):
            if file.is_file() and "__pycache__" not in str(file):
                hashes[str(file.relative_to(ROOT))]=sha(file)
    target=out/("protected_inputs_after.json" if after else "protected_inputs_before.json")
    assert not target.exists(),"No sobrescribir auditoría de integridad"
    if after:
        before=readj(out/"protected_inputs_before.json")
        assert hashes==before["sha256"],"Insumos protegidos modificados"
    savej(target,dict(at=now(),sha256=hashes,files=len(hashes),status="INTEGRITY_OK"))
    print(view,"INPUT_INTEGRITY",len(hashes),"after" if after else "before",flush=True)


def metrics(y,p):
    pred=np.asarray(p)>=.5
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        macro_f1=float(f1_score(y,pred,labels=[0,1],average="macro",zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if len(set(y))==2 else None,
        sensitivity=tp/(tp+fn) if tp+fn else None,specificity=tn/(tn+fp) if tn+fp else None,
        fpr=fp/(tn+fp) if tn+fp else None,fnr=fn/(tp+fn) if tp+fn else None,tn=tn,fp=fp,fn=fn,tp=tp)


def references(view):
    p=paths(view); f=fusion_out(view); variant="full" if view=="left" else "sin_relacion"
    seed=42 if view=="left" else 1337
    refs={"pose":p["left_out"]/"validation_predictions.csv" if view=="left" else p["right_out"]/"seed_1337/validation_predictions.csv",
        "seg_"+variant:p["seg_train"]/variant/f"seed_{seed}/validation_predictions.csv",
        "late_"+variant:f/f"late_fusion_{variant}/validation_predictions.csv"}
    if view=="left":refs["embedding_full"]=f/"embedding_fusion_full/seed_1337/validation_predictions.csv"
    return refs


def freeze(view):
    out=outdir(view);out.mkdir(exist_ok=True)
    target=out/f"bp_{view}_final_test_protocol_frozen.json"
    if target.exists():raise RuntimeError("Protocolo existente: no sobrescribir")
    groups,names,_,info=cohort(view)
    p=paths(view);f=fusion_out(view)
    original=readj(f/"fusion_source_manifest.json")
    actual=snapshot(view)
    assert actual==original["source_sha256"], "Fuentes distintas al manifiesto de fusión"
    assert readj(f/"fusion_integrity_verification.json")["status"]=="INTEGRITY_OK"
    selection=readj(f/"fusion_selected_candidate.json")
    principal="embedding_full" if view=="left" else "seg_sin_relacion"
    assert selection["winner"]==principal
    assert sha(Path(selection["frozen_checkpoint"]["path"]))==selection["frozen_checkpoint"]["sha256"]
    rows=rows_csv(p["cohort"])
    groups["test"]=[r for r in rows if r["split"]=="test" and r["supervised_member"]=="True"]
    expected=(422,91,91) if view=="left" else (431,93,93)
    assert tuple(len(groups[k]) for k in ("train","validation","test"))==expected
    all_ids=[r["video_id"] for role in groups for r in groups[role]]
    assert len(set(all_ids))==sum(expected)
    assert view!="right" or "bp_354" not in all_ids
    labels={r["video_id"]:int(r["label"]) for r in rows_csv(p["labels"])}
    split={r["video_id"]:r for r in rows_csv(p["split"])}
    for role,rs in groups.items():
        for r in rs: assert split[r["video_id"]]["split"]==role and int(split[r["video_id"]]["label"])==labels[r["video_id"]]
    audit_path=p["source"]/"source_audit.csv"
    audit={r["video_id"]:r for r in rows_csv(audit_path)}
    duplicates={}
    for role,rs in groups.items():
        for r in rs:
            h=audit[r["video_id"]].get("video_sha256","")
            if h:duplicates.setdefault(h,[]).append({"id":r["video_id"],"split":role})
    pairs=[v for v in duplicates.values() if len(v)>1]
    cross=[v for v in pairs if len({r["split"] for r in v})>1]
    duplicate_report=dict(source=str(audit_path),coverage=sum(len(v) for v in duplicates.values()),expected=sum(expected),
        exact_duplicates=pairs,cross_partition_duplicates=cross,physical_independence="not_verified",
        method="SHA-256 MP4 ya disponible; no decodificación ni nuevo hash de videos")
    duplicate_report["coverage_status"]="verified" if duplicate_report["coverage"]==sum(expected) else "not_verified_incomplete_hashes"
    savej(out/"duplicate_audit.json",duplicate_report)
    assert not cross, "Fuga demostrada por hash de video entre particiones"
    variant="full" if view=="left" else "sin_relacion"
    contract_path=p["seg_train"]/variant/"channels_and_train_preprocessing.json"
    contract=readj(contract_path)
    assert contract["C"]==(138 if view=="left" else 90)
    files={k:Path(v["path"]) for k,v in actual.items()}
    files.update(candidate=f/"fusion_selected_candidate.json",fusion_manifest=f/"fusion_source_manifest.json",
        fusion_integrity=f/"fusion_integrity_verification.json",source_audit=audit_path,
        schema=p["schema"],source_config=p["frozen"],pose_audit=p["pose_data"]/"audit.csv",
        pose_preprocessing_manifest=p["pose_data"]/"preprocessing.csv",
        pose_original_config=p["historical_config"],fusion_models=ROOT/"training/bp/experiments/bp_side_fusion_v1/models.py",
        historical_replay_code=ROOT/"training/bp/experiments/bp_side_fusion_v1/verify.py",
        evaluator=Path(__file__),analysis_code=Path(__file__).with_name("analyze.py"))
    if view=="left":
        files["head"]=Path(selection["frozen_checkpoint"]["path"])
        files["embedding_scaler"]=Path(selection["frozen_embedding_scaler"]["path"])
        assert sha(files["embedding_scaler"])==selection["frozen_embedding_scaler"]["sha256"]
    # Proteger también CV y documentos históricos; no abrir secuencias Test.
    for folder in (f,p["seg_train"],ROOT/"training/bp/experiments"/f"bp_{view}_04_cv_v1",p["left_out"] if view=="left" else p["right_out"]):
        for file in folder.rglob("*"):
            if file.is_file() and "__pycache__" not in str(file):files[str(file.relative_to(ROOT))]=file
    protected={str(file.relative_to(ROOT)):sha(file) for file in set(files.values())}
    savej(out/"source_manifest.json",dict(created_utc=now(),protected_sha256=protected,
        fusion_snapshot=actual,input_hash_evidence=str(audit_path),test_sequences_opened=0))
    with open(actual["pose_scaler"]["path"],"rb") as handle:scaler=pickle.load(handle)
    assert int(scaler.n_samples_seen_)==expected[0] and scaler.n_features_in_==4080
    # Validar contratos de pesos, sin secuencias Test.
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base=PoseCNN(T=170,K=8,C=3)
    base.load_state_dict(torch.load(actual["pose_checkpoint"]["path"],map_location="cpu",weights_only=True),strict=True)
    assert sha(Path(actual["pose_checkpoint"]["path"]))==selection["backbone_checkpoint_sha256"]["pose"]
    state=torch.load(actual[f"seg_{variant}_checkpoint"]["path"],map_location="cpu",weights_only=True)
    assert state["input_channels"]==contract["input_channel_names"] and state["channels_total"]==contract["C"]
    assert state["view"]==view and state["seed"]==(42 if view=="left" else 1337)
    SegCNNTemporal(contract["C"]).load_state_dict(state["state_dict"],strict=True)
    if view=="left":
        head=FusionHead();state=torch.load(files["head"],map_location="cpu",weights_only=True)
        head.load_state_dict(state["head_state_dict"],strict=True)
        assert sum(q.numel() for q in head.parameters())==6209
        assert state["embedding_scaler_sha256"]==sha(files["embedding_scaler"])
        with np.load(files["embedding_scaler"],allow_pickle=False) as z:
            assert z["mean"].shape==z["std"].shape==(192,) and int(z["n_samples_train"])==422
    protocol=dict(version="V1",frozen_utc=now(),view=view,principal=principal,models=models(view),
        secondary=[m for m in models(view) if m!=principal],test_ids=[r["video_id"] for r in groups["test"]],
        validation_ids=[r["video_id"] for r in groups["validation"]],counts=dict(zip(("train","validation","test"),expected)),
        threshold=.5,late_formula="0.5*p_pose + 0.5*p_seg",positive_class="1=valido; FP=invalido aprobado",
        checkpoints={k:v for k,v in actual.items() if "checkpoint" in k},candidate=selection,
        pose_shape=[3,170,8],pose_keypoint_order=POINTS,
        pose_preprocessing="preprocess histórico: huecos XY interiores <=5; confianza ausente 0; centro mediano hombro3, escala hombro3-cadera2; linspace(0,T-1,170); scaler congelado 4080 features",
        seg_contract=contract,raw_feature_order=names,
        embedding="LEFT: Pose64 + Seg128; scaler Train congelado; MLP192->32->1; seed1337" if view=="left" else None,
        inference=dict(device=str(device),pose_batch=16,seg_batch=128,embedding_batch=128,pose_fp32=True,seg_amp=device.type=="cuda",eval=True,inference_mode=True,augmentation=False),
        validation_tolerance=dict(base_probability_atol=1e-6,fusion_probability_atol=2e-5,rtol=0,decisions="exact",metrics_atol=1e-9,
            evidence="bp_side_fusion_v1/data.py:149 y verify.py:51"),
        metrics=list(metrics(np.array([0,1]),np.array([.1,.9]))),
        confidence_intervals=dict(method="paired ordinary video bootstrap percentile",replicates=10000,seed=20261015,
            confidence=.95,metrics=["accuracy","f1","macro_f1","balanced_accuracy","specificity","roc_auc"],
            differences=["accuracy","macro_f1","specificity"],single_class="discard replica for all metrics and log count",binomial="Wilson 95% para FPR y especificidad"),
        paired_comparisons=([ ["pose","seg_full"],["pose","late_full"],["pose","embedding_full"],["seg_full","embedding_full"],["late_full","embedding_full"] ] if view=="left" else
            [["pose","seg_sin_relacion"],["pose","late_sin_relacion"],["seg_sin_relacion","late_sin_relacion"]]),
        near_threshold="abs(p-0.5)<=0.05",resume="solo fallo técnico, mismos hashes/protocolo; no repetir inferencia completada",
        protected_manifest_sha256=sha(out/"source_manifest.json"),duplicate_audit=duplicate_report,
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,cuda=torch.version.cuda),
        limitations=LIMITACIONES)
    savej(target,protocol)
    (out/f"bp_{view}_final_test_protocol_frozen.sha256").write_text(sha(target)+"\n",encoding="ascii")
    savej(out/"execution_state.json",dict(status="PROTOCOL_FROZEN",test_inference_runs=0,at=now()))
    print(view,"PROTOCOL_FROZEN; duplicados entre particiones:",cross,flush=True)


def verify(view):
    out=outdir(view);path=out/f"bp_{view}_final_test_protocol_frozen.json"
    assert sha(path)==(out/f"bp_{view}_final_test_protocol_frozen.sha256").read_text().strip()
    protocol=readj(path);source=readj(out/"source_manifest.json")
    assert sha(out/"source_manifest.json")==protocol["protected_manifest_sha256"]
    changed=[p for p,h in source["protected_sha256"].items() if sha(ROOT/p)!=h]
    assert not changed,changed
    return protocol,source


class Evaluator:
    def __init__(self,view,protocol,source):
        self.view=view;self.p=paths(view);self.protocol=protocol
        self.device=torch.device(protocol["inference"]["device"])
        self.contract=protocol["seg_contract"];self.names=protocol["raw_feature_order"]
        snap=source["fusion_snapshot"]
        self.pose=PoseCNN(T=170,K=8,C=3).to(self.device)
        self.pose.load_state_dict(torch.load(snap["pose_checkpoint"]["path"],map_location=self.device,weights_only=True))
        with open(snap["pose_scaler"]["path"],"rb") as f:self.scaler=pickle.load(f)
        variant="full" if view=="left" else "sin_relacion"
        state=torch.load(snap[f"seg_{variant}_checkpoint"]["path"],map_location=self.device,weights_only=True)
        self.seg=SegCNNTemporal(self.contract["C"]).to(self.device)
        self.seg.load_state_dict(state["state_dict"])
        self.head=None
        if view=="left":
            candidate=protocol["candidate"]
            state=torch.load(candidate["frozen_checkpoint"]["path"],map_location=self.device,weights_only=True)
            self.head=FusionHead().to(self.device);self.head.load_state_dict(state["head_state_dict"])
            with np.load(candidate["frozen_embedding_scaler"]["path"],allow_pickle=False) as z:
                self.mean=z["mean"];self.std=z["std"]
        for m in (self.pose,self.seg,self.head):
            if m is not None:m.eval().requires_grad_(False)

    def load(self,role):
        ids=self.protocol[role+"_ids"]
        rows={r["video_id"]:r for r in rows_csv(self.p["cohort"])}
        audit={r["video_id"]:r for r in rows_csv(self.p["source"]/"source_audit.csv")}
        labels={r["video_id"]:int(r["label"]) for r in rows_csv(self.p["labels"])}
        xp=[];xs=[];hashes={};quality=[]
        for vid in ids:
            r=rows[vid];assert r["split"]==role and r["supervised_member"]=="True"
            pp=self.p["keypoints"]/f"{vid}.npy";sp=self.p["source"]/"sequences"/f"{vid}_temporal.npz"
            hp,hs=sha(pp),sha(sp)
            assert hp==audit[vid]["pose_sha256"] and hs==r["features_sha256"],f"Hash fuente alterado: {vid}"
            hashes[str(pp.relative_to(ROOT))]=hp;hashes[str(sp.relative_to(ROOT))]=hs
            raw=np.load(pp,allow_pickle=False)
            assert raw.shape==(int(r["T"]),8,3)
            x,_,_=preprocess(raw)
            xp.append(transform(self.scaler,x[None])[0])
            with np.load(sp,allow_pickle=False) as z:
                assert str(z["view"])==self.view and str(z["video_id"])==vid
                assert len(z["features"])==len(raw)
                v,a,pr=resample(z,self.names)
            c=self.contract;ix=c["selected_feature_indices"];stats=c["train_statistics"];chosen=c["selected_feature_names"]
            med,mean,std=[np.array([stats[n][key] for n in chosen],np.float32) for key in ("median","mean","std")]
            vv=v[:,ix];aa=a[:,ix].astype(bool);orig=pr[:,ix]
            numeric=(np.where(aa,vv,med)-mean)/std
            flags=[np.isin(orig,k).astype(np.float32) for k in (2,(3,6),4,5,7)]
            combined=np.concatenate([numeric,aa.astype(np.float32),*flags],axis=1)
            xs.append(combined[:,c["input_channel_indices_before_constant_filter"]].T)
            quality.append(dict(video_id=vid,view=self.view,T_original=len(raw),pose_nan_values=int(np.isnan(raw).sum()),seg_missing_fraction=float(1-a.mean())))
        xp=np.ascontiguousarray(np.stack(xp).transpose(0,3,1,2));xs=np.ascontiguousarray(np.stack(xs))
        assert xp.shape==(len(ids),3,170,8) and xs.shape==(len(ids),self.contract["C"],170)
        assert np.isfinite(xp).all() and np.isfinite(xs).all()
        savej(outdir(self.view)/f"{role}_input_hashes.json",hashes)
        pd.DataFrame(quality).to_csv(outdir(self.view)/f"{role}_input_metadata.csv",index=False)
        return ids,np.array([labels[v] for v in ids]),xp,xs

    @torch.inference_mode()
    def predict(self,xp,xs):
        pp=[];ss=[];zz=[]
        for start in range(0,len(xp),16):
            xb=torch.as_tensor(xp[start:start+16],device=self.device)
            pp.extend(self.pose(xb).flatten().cpu().tolist())
        for start in range(0,len(xs),128):
            xb=torch.as_tensor(xs[start:start+128],device=self.device)
            with torch.autocast(device_type=self.device.type,enabled=self.device.type=="cuda"):
                logits=self.seg(xb)
            ss.extend(torch.sigmoid(logits.float()).cpu().tolist())
            if self.head is not None:
                zp=PoseEmbeddingAdapter(self.pose).extract_embeddings(torch.as_tensor(xp[start:start+128],device=self.device))
                with torch.autocast(device_type=self.device.type,enabled=self.device.type=="cuda"):
                    zs=SegEmbeddingAdapter(self.seg).extract_embeddings(xb)
                z=torch.cat([zp.float(),zs.float()],dim=1).cpu().numpy()
                assert z.shape[1]==192
                logits=self.head(torch.as_tensor((z-self.mean)/self.std,device=self.device))
                zz.extend(torch.sigmoid(logits).cpu().tolist())
        variant="full" if self.view=="left" else "sin_relacion"
        probs={"pose":np.array(pp),"seg_"+variant:np.array(ss),"late_"+variant:.5*(np.array(pp)+np.array(ss))}
        if zz:probs["embedding_full"]=np.array(zz)
        assert set(probs)==set(models(self.view))
        return probs


def evaluate(view,role):
    protocol,source=verify(view);out=outdir(view)
    if (out/f"{role}_predictions.csv").exists():raise RuntimeError("Predicciones existentes: no repetir inferencia")
    if role=="test":
        assert readj(out/"validation_reproducibility.json")["passed"]
        assert readj(out/"protected_inputs_before.json")["status"]=="INTEGRITY_OK"
        state=readj(out/"execution_state.json")
        assert state["test_inference_runs"]==0,"Test ya iniciado: requiere revisión técnica documentada"
        savej(out/"execution_state.json",dict(status="TEST_STARTED",test_inference_runs=1,at=now()))
    evaluator=Evaluator(view,protocol,source)
    ids,y,xp,xs=evaluator.load(role)
    probabilities=evaluator.predict(xp,xs)
    records=[];scores=[]
    for model,p in probabilities.items():
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        scores.append(dict(model=model,n=len(y),**metrics(y,p)))
        records.extend(dict(video_id=v,view=view,model=model,label=int(label),probability_valid=float(q),predicted=int(q>=.5),
            protocol_sha256=sha(out/f"bp_{view}_final_test_protocol_frozen.json")) for v,label,q in zip(ids,y,p))
    pd.DataFrame(records).to_csv(out/f"{role}_predictions.csv",index=False)
    pd.DataFrame(scores).to_csv(out/f"{role}_metrics.csv",index=False)
    savej(out/f"{role}_metrics.json",scores)
    if role=="validation":
        checks=[];original_scores=[]
        for model,path in references(view).items():
            original=pd.read_csv(path)
            assert list(original.video_id)==ids and np.array_equal(original.label,y),f"Orden/labels distintos {model}"
            p=probabilities[model];old=original.probability_valid.to_numpy()
            tol=1e-6 if model.startswith(("pose","seg_")) else 2e-5
            delta=float(np.max(np.abs(p-old)))
            actual_metrics=metrics(y,p);old_metrics=metrics(y,old)
            ok=delta<=tol and np.array_equal(p>=.5,old>=.5) and all(np.isclose(actual_metrics[k],old_metrics[k],atol=1e-9,rtol=0) for k in actual_metrics)
            checks.append(dict(model=model,max_abs_probability_difference=delta,tolerance=tol,passed=bool(ok),reference_sha256=sha(path)))
            original_scores.append(dict(model=model,n=len(y),**old_metrics))
        passed=all(r["passed"] for r in checks)
        savej(out/"validation_reproducibility.json",dict(passed=passed,checks=checks,at=now(),test_opened=False))
        pd.DataFrame(original_scores).to_csv(out/"validation_original_metrics.csv",index=False)
        assert passed,checks
        savej(out/"execution_state.json",dict(status="VALIDATION_REPRODUCED",test_inference_runs=0,at=now()))
    else:
        savej(out/"execution_state.json",dict(status="TEST_INFERENCE_COMPLETE",test_inference_runs=1,at=now(),
            n=len(y),valid=int(y.sum()),invalid=int((y==0).sum()),prediction_sha256=sha(out/"test_predictions.csv")))
    verify(view)
    print(view,role,"COMPLETE",flush=True)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--view",choices=["left","right"],required=True)
    parser.add_argument("--stage",choices=["freeze","protect","validation","test"],required=True);a=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    try:
        if a.stage=="freeze":freeze(a.view)
        elif a.stage=="protect":protect_inputs(a.view)
        else:evaluate(a.view,a.stage)
    except Exception:
        out=outdir(a.view);out.mkdir(exist_ok=True)
        savej(out/f"blocked_{a.stage}.json",dict(status=f"{a.view.upper()}_FINAL_TEST_BLOCKED",stage=a.stage,at=now(),error=traceback.format_exc()))
        raise


if __name__=="__main__":main()
