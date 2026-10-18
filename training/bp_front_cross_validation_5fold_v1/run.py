"""CV retrospectiva FRONT; acceso a secuencias limitado a development.

python -m training.bp_front_cross_validation_5fold_v1.run
"""
import csv
import hashlib
import json
import os
import platform
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from training.bp_front_posecnn_common_v1 import data as pose
from training.bp_front_posecnn_common_v1.model import BPFrontPoseCNN
from training.bp_front_segcnn_v1 import data as seg
from training.bp_front_segcnn_v1.model import SegCNNTemporal

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODELS = ["posecnn", "segcnn_full", "segcnn_sin_relacion",
          "late_fusion_full", "late_fusion_sin_relacion"]
SEEDS = dict(zip(MODELS[:3], [42, 1337, 42]))


def guardar(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def metricas(y, p):
    pred = p >= .5
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred), f1=f1_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
        balanced_accuracy=balanced_accuracy_score(y, pred), roc_auc=roc_auc_score(y, p),
        average_precision=average_precision_score(y, p), sensitivity=tp/(tp+fn),
        specificity=tn/(tn+fp), fpr=fp/(tn+fp), fnr=fn/(tp+fn), tn=tn, fp=fp, fn=fn, tp=tp)


def preparar():
    for d in ["config", "folds", "oof", "figures", "reports", *MODELS]:
        (OUT / d).mkdir(exist_ok=True)
    # Filtrar antes de consultar etiquetas o rutas: Test no participa en auditorías.
    with open(pose.COHORTE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columnas = reader.fieldnames
        filas = [r for r in reader if r["common_split"] in ("train", "validation")
                 and r["excluido"] == "False"]
    df = pd.DataFrame(filas).sort_values("video_id").reset_index(drop=True)
    df["label"] = df.label.astype(int)
    assert len(df) == 561 and df.label.value_counts().to_dict() == {1: 339, 0: 222}
    assert df.common_split.value_counts().to_dict() == {"train": 462, "validation": 99}
    assert df.video_id.is_unique and not set(df.video_id) & {"bp_349", "bp_352"}
    assert "group_id" not in columnas, "Revisar acreditación de grupos antes de continuar"
    df.to_csv(OUT / "config/development_manifest.csv", index=False)
    fuentes = [pose.COHORTE, seg.ESQUEMA, Path(__file__), Path(pose.__file__), Path(seg.__file__),
        ROOT / "training/bp/bp_front_posecnn.py"]
    for folder in ["bp_front_posecnn_common_v1", "bp_front_segcnn_v1", "bp_front_fusion_v1"]:
        base = ROOT / "training" / folder
        fuentes += list(base.glob("*.py"))
        fuentes += list(base.glob("**/*.pt"))
        fuentes += list(base.glob("**/*scaler*.npz"))
        fuentes += list(base.glob("**/channels_and_train_preprocessing.json"))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in set(fuentes)}
    nombres = [f["nombre"] for f in json.loads(seg.ESQUEMA.read_text(encoding="utf-8"))["features"]]
    canales = {}
    for variante in ("full", "sin_relacion"):
        archivo = ROOT / f"training/bp_front_segcnn_v1/{variante}/channels_and_train_preprocessing.json"
        original = json.loads(archivo.read_text(encoding="utf-8"))
        canales[variante] = {k: original[k] for k in ("indices", "nombres", "nombres_procedencia", "canales_total")}
        assert [nombres[i] for i in original["indices"]] == original["nombres"]
    guardar(OUT / "config/frozen_channels.json", canales)
    xp, mp, vs, ms, ps, auditoria, duplicados = [], [], [], [], [], [], {}
    for r in df.to_dict("records"):
        vid = r["video_id"]
        pp = pose.POSE_DIR / f"{vid}.npy"
        sp = Path(r["features_path"])
        assert pp.resolve() == Path(r["pose_path"]).resolve()
        assert sp.resolve().parent == (seg.FUENTE / "sequences").resolve()
        for p in (pp, sp):
            hashes[str(p.relative_to(ROOT))] = sha(p)
        raw = np.load(pp, allow_pickle=False)
        digest = hashlib.sha256(str((raw.shape, str(raw.dtype))).encode() + raw.tobytes()).hexdigest()
        duplicados.setdefault(digest, []).append(vid)
        x, m, centro, escala = pose.secuencia_a_170(raw)
        xp.append(x); mp.append(m)
        with np.load(sp, allow_pickle=False) as sec:
            assert list(sec["feature_names"]) == nombres
            assert len(sec["features"]) == len(raw) == int(r["T"])
            v, a, p = seg.remuestrear(sec)
        vs.append(v); ms.append(a); ps.append(p)
        auditoria.append(dict(video_id=vid, T_original=len(raw), pose_content_sha256=digest,
            missing_pose_values=int(np.isnan(raw).sum()), spatial_scale=escala,
            spatial_center_x=float(centro[0]), spatial_center_y=float(centro[1])))
    repetidos = [v for v in duplicados.values() if len(v) > 1]
    guardar(OUT / "config/duplicate_audit.json", {"development_exact_pose_duplicates": repetidos,
        "known_excluded_duplicate": "bp_352 de bp_351", "excluded": ["bp_349", "bp_352"],
        "scope": "contenido exacto de keypoints development; no acredita identidad física"})
    assert not repetidos, "Duplicados requieren revisión antes de distribuir folds"
    pd.DataFrame(auditoria).to_csv(OUT / "config/preprocessing_manifest.csv", index=False)
    arrays = [np.stack(a) for a in (xp, mp, vs, ms, ps)]
    np.savez_compressed(OUT / "config/development_resampled.npz", video_ids=df.video_id.to_numpy(dtype=str),
        pose=arrays[0], pose_availability=arrays[1], features=arrays[2], availability=arrays[3], provenance=arrays[4],
        original_lengths=df["T"].astype(int).to_numpy())
    folds, asignaciones, distribucion = [], [], []
    y = df.label.to_numpy()
    for fold, (outer, ev) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y), 1):
        ti, vi = next(StratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=42).split(outer, y[outer]))
        tr, va = outer[ti], outer[vi]
        assert not (set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev))
        assert set(tr)|set(va)|set(ev) == set(range(561))
        folds.append((tr, va, ev))
        roles = np.full(561, "outer_validation", dtype=object)
        roles[tr] = "inner_train"; roles[va] = "inner_validation"
        manifest = df[["video_id", "label", "common_split"]].copy()
        manifest["fold"] = fold; manifest["role"] = roles; manifest["group_id"] = ""
        manifest.to_csv(OUT / f"folds/fold_{fold}_split.csv", index=False)
        for role in ("inner_train", "inner_validation", "outer_validation"):
            sub = manifest[manifest.role == role]
            distribucion.append(dict(fold=fold, role=role, n=len(sub), valid=int(sub.label.sum()), invalid=int((sub.label==0).sum())))
        asignaciones.extend(dict(video_id=df.iloc[i].video_id, label=int(y[i]), fold=fold, group_id="") for i in ev)
    pd.DataFrame(asignaciones).sort_values("video_id").to_csv(OUT / "fold_assignments.csv", index=False)
    pd.DataFrame(distribucion).to_csv(OUT / "folds/class_distribution.csv", index=False)
    for p in [OUT / "fold_assignments.csv", * (OUT / "folds").glob("*.csv"), OUT / "config/development_manifest.csv"]:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    guardar(OUT / "config/source_hashes_before.json", hashes)
    guardar(OUT / "config/protocol.json", dict(seeds=SEEDS, outer_seed=42, inner_seed=42,
        inner_fraction=.15, groups="No acreditados; estratificación por video", threshold=.5,
        fusion_weights=[.5,.5], max_epochs=200, patience=15, lr=.001, weight_decay=.0001,
        scheduler={"factor":.5,"patience":5}, batch_pose=16, batch_seg=32,
        pose_optimizer="Adam", seg_optimizer="AdamW", seg_amp=torch.cuda.is_available(),
        std_ddof=1, selection="mínima loss interna; pose mejora estricta, seg delta 1e-7",
        historical_candidate="late_fusion_sin_relacion; selección original preservada",
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, torch=str(torch.__version__), cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    print("DEVELOPMENT VERIFICADO: 561, válidos=339, inválidos=222", flush=True)
    print(pd.DataFrame(distribucion).to_string(index=False), flush=True)
    return df, arrays, canales, folds, hashes


def preparar_seg(arrays, indices, canal, variante):
    v, m, p = arrays[2:]
    ix = canal["indices"]
    med, mean, std = [], [], []
    for j in ix:
        obs = v[indices, :, j][m[indices, :, j]].astype(np.float64)
        assert len(obs), f"Canal sin observaciones internas: {j}"
        med.append(np.median(obs)); mean.append(np.mean(obs)); std.append(max(np.std(obs), 1e-6))
    med, mean, std = [np.asarray(a, dtype=np.float32) for a in (med, mean, std)]
    mask = m[:, :, ix]
    x = (np.where(mask, v[:, :, ix], med) - mean) / std
    prov = p if variante == "full" else p[:, :, :3]
    x = np.ascontiguousarray(np.concatenate([x, mask.astype(np.float32), prov], axis=2).transpose(0,2,1))
    assert x.shape[1] == canal["canales_total"] and np.isfinite(x).all()
    return x, dict(median=med, mean=mean, scale=std)


def entrenar(nombre, fold, arrays, canales, y, tr, va, ev):
    carpeta = OUT / nombre / f"fold_{fold}"
    carpeta.mkdir(exist_ok=True)
    seed = SEEDS[nombre]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    es_pose = nombre == "posecnn"
    amp = not es_pose and device.type == "cuda"
    if es_pose:
        scaler = StandardScaler().fit(arrays[0][tr].reshape(len(tr), -1))
        np.savez_compressed(carpeta / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_, n_samples=scaler.n_samples_seen_, fit_indices=tr)
        aug, ya = pose.augment_train(arrays[0][tr], arrays[1][tr], y[tr], seed)
        xa = pose.transformar(aug, scaler).transpose(0,3,1,2).copy()
        x = pose.transformar(arrays[0], scaler).transpose(0,3,1,2).copy()
        modelo = BPFrontPoseCNN().to(device)
        optim = torch.optim.Adam(modelo.parameters(), lr=.001, weight_decay=.0001)
        loss_fn = torch.nn.BCELoss()
        batch = 16
    else:
        variante = nombre.removeprefix("segcnn_")
        x, stats = preparar_seg(arrays, tr, canales[variante], variante)
        np.savez_compressed(carpeta / "scaler.npz", **stats, fit_indices=tr)
        xa, ya = x[tr], y[tr]
        modelo = SegCNNTemporal(x.shape[1]).to(device)
        optim = torch.optim.AdamW(modelo.parameters(), lr=.001, weight_decay=.0001)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        batch = 32
    params = sum(p.numel() for p in modelo.parameters())
    assert params == {"posecnn":118529,"segcnn_full":56449,"segcnn_sin_relacion":54209}[nombre]
    dl = DataLoader(TensorDataset(torch.from_numpy(xa), torch.tensor(ya, dtype=torch.float32)),
        batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=.5, patience=5)
    grad = torch.amp.GradScaler("cuda", enabled=amp)

    def evaluar(indices):
        modelo.eval(); probs=[]; total=0.
        with torch.no_grad():
            for a in range(0, len(indices), 16 if es_pose else 128):
                idx = indices[a:a+(16 if es_pose else 128)]
                xb = torch.as_tensor(x[idx], device=device)
                yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    out = modelo(xb).reshape(-1)
                    loss = loss_fn(out.float(), yb)
                total += loss.item()*len(idx)
                probs.extend((out if es_pose else torch.sigmoid(out.float())).cpu().tolist())
        return total/len(indices), np.asarray(probs)

    best=float("inf"); waiting=0; history=[]; best_ep=0
    for ep in range(1,201):
        modelo.train()
        for xb, yb in dl:
            xb, yb=xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                out=modelo(xb).reshape(-1)
                loss=loss_fn(out.float(),yb)
            grad.scale(loss).backward(); grad.step(optim); grad.update()
        lt, pt=evaluar(tr); lv, pv=evaluar(va)
        assert np.isfinite([lt,lv]).all()
        scheduler.step(lv)
        history.append(dict(epoch=ep,train_loss=lt,validation_loss=lv,
            train_f1=f1_score(y[tr],pt>=.5),validation_f1=f1_score(y[va],pv>=.5),lr=optim.param_groups[0]["lr"]))
        if lv < best-(0 if es_pose else 1e-7):
            best=lv; best_ep=ep; waiting=0
            torch.save(dict(state_dict=modelo.state_dict(),fold=fold,seed=seed,epoch=ep,inner_loss=lv),carpeta/"best.pt")
        else:
            waiting+=1
        if ep==1 or ep%10==0:
            print(f"fold={fold} {nombre} epoch={ep} inner_loss={lv:.5f}",flush=True)
        if waiting>=15:
            break
    modelo.load_state_dict(torch.load(carpeta/"best.pt",weights_only=True,map_location=device)["state_dict"])
    _, probs=evaluar(ev)
    pd.DataFrame(history).to_csv(carpeta/"learning_curves.csv",index=False)
    fig,ax=plt.subplots()
    ax.plot([h["train_loss"] for h in history],label="Inner train")
    ax.plot([h["validation_loss"] for h in history],label="Inner validation")
    ax.set(xlabel="Epoch (índice 0)",ylabel="Loss",title=f"{nombre} fold {fold}"); ax.legend()
    fig.savefig(carpeta/"learning_curves.png",dpi=160); plt.close(fig)
    guardar(carpeta/"training_metadata.json",dict(seed=seed,fold=fold,parameters=params,best_epoch=best_ep,epochs_run=ep,
        train_n=len(tr),inner_val_n=len(va),outer_n=len(ev),checkpoint_sha256=sha(carpeta/"best.pt"),fresh_initialization=True))
    print(f"COMPLETADO fold={fold} {nombre} best_epoch={best_ep}",flush=True)
    return probs


def main():
    torch.set_num_threads(4)
    if (OUT / "config/source_hashes_before.json").exists():
        if "--resume" not in sys.argv:
            raise RuntimeError("Salida ya inicializada: usar --resume")
        hashes=json.loads((OUT/"config/source_hashes_before.json").read_text(encoding="utf-8"))
        script_key=str(Path(__file__).relative_to(ROOT))
        cambios=[p for p,h in hashes.items() if p != script_key and sha(ROOT/p)!=h]
        assert not cambios, cambios
        guardar(OUT/"config/resume_revision.json",dict(reason="timeout externo 120 segundos; reinicio determinista del entrenamiento incompleto",
            original_script_sha256=hashes[script_key], resume_script_sha256=sha(Path(__file__))))
        hashes[script_key]=sha(Path(__file__))
        df=pd.read_csv(OUT/"config/development_manifest.csv")
        # Cache local producido por esta ejecución; primera revisión guardó IDs object.
        with np.load(OUT/"config/development_resampled.npz",allow_pickle=True) as z:
            arrays=[z[k] for k in ("pose","pose_availability","features","availability","provenance")]
            assert list(z["video_ids"])==list(df.video_id)
        canales=json.loads((OUT/"config/frozen_channels.json").read_text(encoding="utf-8"))
        folds=[]
        for fold in range(1,6):
            manifest=pd.read_csv(OUT/f"folds/fold_{fold}_split.csv")
            assert list(manifest.video_id)==list(df.video_id)
            # Recuperar orden original de StratifiedShuffleSplit (afecta minibatches).
            outer,ev=list(StratifiedKFold(5,shuffle=True,random_state=42).split(df,df.label))[fold-1]
            ti,vi=next(StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(outer,df.label.to_numpy()[outer]))
            tr,va=outer[ti],outer[vi]
            for idx,role in ((tr,"inner_train"),(va,"inner_validation"),(ev,"outer_validation")):
                assert set(idx)==set(np.flatnonzero(manifest.role==role))
            folds.append((tr,va,ev))
    else:
        df, arrays, canales, folds, hashes = preparar()
    y=df.label.to_numpy(); predicciones=[]; metricas_fold=[]
    for fold,(tr,va,ev) in enumerate(folds,1):
        probs={}
        for n in MODELS[:3]:
            saved=OUT/n/f"outer_predictions_fold_{fold}.csv"
            if saved.exists():
                anterior=pd.read_csv(saved)
                assert list(anterior.video_id)==list(df.iloc[ev].video_id)
                assert np.array_equal(anterior.label,y[ev])
                probs[n]=anterior.probability.to_numpy()
            else:
                folder=OUT/n/f"fold_{fold}"
                if folder.exists():
                    assert not (folder/"training_metadata.json").exists(), "Modelo completo sin predicciones: requiere revisión"
                    folder.rename(OUT/n/f"fold_{fold}_interrupted_timeout")
                probs[n]=entrenar(n,fold,arrays,canales,y,tr,va,ev)
        probs["late_fusion_full"]=(probs["posecnn"]+probs["segcnn_full"])*.5
        probs["late_fusion_sin_relacion"]=(probs["posecnn"]+probs["segcnn_sin_relacion"])*.5
        for nombre,p in probs.items():
            filas=[dict(model=nombre,fold=fold,video_id=df.iloc[i].video_id,label=int(y[i]),probability=float(q),predicted=int(q>=.5)) for i,q in zip(ev,p)]
            predicciones.extend(filas)
            metricas_fold.append(dict(model=nombre,fold=fold,n=len(ev),**metricas(y[ev],p)))
            pd.DataFrame(filas).to_csv(OUT/nombre/f"outer_predictions_fold_{fold}.csv",index=False)
        pd.DataFrame(predicciones).to_csv(OUT/"oof/partial_predictions.csv",index=False)
    oof=pd.DataFrame(predicciones)
    assert len(oof)==2805 and not oof.duplicated(["model","video_id"]).any()
    for n in MODELS:
        assert set(oof[oof.model==n].video_id)==set(df.video_id)
    oof.to_csv(OUT/"cv_oof_predictions.csv",index=False)
    mf=pd.DataFrame(metricas_fold); mf.to_csv(OUT/"cv_metrics_by_fold.csv",index=False)
    summary=mf.groupby("model")[[k for k in metricas(y,np.full(len(y),.6))]].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns]
    summary.to_csv(OUT/"cv_metrics_summary.csv")
    pooled=[]
    for n in MODELS:
        sub=oof[oof.model==n]
        pooled.append(dict(model=n,n=len(sub),**metricas(sub.label.to_numpy(),sub.probability.to_numpy())))
    pd.DataFrame(pooled).to_csv(OUT/"cv_pooled_oof_metrics.csv",index=False)
    changed=[p for p,h in hashes.items() if sha(ROOT/p)!=h]
    guardar(OUT/"cv_integrity_verification.json",dict(status="TRAINING_COMPLETE_REPORT_PENDING",sources_unchanged=not changed,
        changed=changed,trained_models=15,outer_folds=5,development_videos=561,oof_rows=2805,
        historical_checkpoints_loaded=0,test_sequences_opened=0,test_predictions_read=0,
        groups_verified=False,partition_overlap=False,scaler_fit="inner_train_only",
        checkpoint_hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.glob("*/fold_*/best.pt")}))
    assert not changed
    print("15 ENTRENAMIENTOS COMPLETADOS; OOF VERIFICADO",flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with open(OUT/"execution.log","a",encoding="utf-8",buffering=1) as log:
        class Tee:
            def write(self,text):
                sys.__stdout__.write(text); log.write(text)
            def flush(self):
                sys.__stdout__.flush()
                if not log.closed:
                    log.flush()
        sys.stdout=Tee()
        try:
            main()
        except Exception:
            detalle=traceback.format_exc(); print(detalle)
            guardar(OUT/"blocked.json",dict(status="FRONT_5FOLD_CV_BLOCKED",error=detalle))
            raise
