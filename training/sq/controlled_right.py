"""SQ right: dos scalers con mismos datos, folds y PoseCNN independientes.

python -m training.sq.controlled_right --stage audit
python -m training.sq.controlled_right --stage prepare
python -m training.sq.controlled_right --stage cv --variant temporal
python -m training.sq.controlled_right --stage cv --variant point_xy
python -m training.sq.controlled_right --stage select
python -m training.sq.controlled_right --stage fit-final
python -m training.sq.controlled_right --stage evaluate-test
python -m training.sq.controlled_right --stage report
"""
import argparse
import csv
import json
import pickle
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from training.common.train_utils import PoseCNN, augment_dataset
from training.sq import train_posecnn as sq
from training.sq.controlled_front import PointXYScaler, transform
from training.sq.run_analysis import digest, metrics, read_predictions


DATA = sq.paths("right")["data"]
BASE = sq.ROOT / "training"
VARIANTS = {"temporal": "baseline_t170_v2", "point_xy": "point_xy_v1"}
CHOICE = BASE / "results" / "sq" / "right" / "posecnn" / "selection_development_only.json"
VIDEO = sq.ROOT / "dataset" / "sq" / "right"


def run(variant):
    return sq.paths("right", VARIANTS[variant])


def guard(path):
    if path.exists():
        raise FileExistsError(f"Ejecución existente: {path}")


def audit():
    summary, manifest, labels = sq.audit("right")
    video_ids = {p.stem for p in VIDEO.glob("*.mp4")}
    pose_ids = {r["video_id"] for r in manifest}
    label_ids = {r["video_id"] for r in labels}
    if summary["blocking_issues"] or video_ids != pose_ids or summary["exact_pose_duplicates"]:
        raise RuntimeError("Revisar IDs o duplicados antes del split")
    lengths = {}
    for path in sorted(VIDEO.glob("*.mp4")):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(path)
        lengths[path.stem] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    mismatches = {r["video_id"]: [r["frames"], lengths[r["video_id"]]] for r in manifest
                  if abs(r["frames"] - lengths[r["video_id"]]) > 1}
    if mismatches:
        raise RuntimeError(f"Duración vídeo/pose discordante: {mismatches}")
    coords, confidences, drops = [], [], []
    for r in manifest:
        raw = np.load(run("temporal")["source"] / (r["video_id"] + ".npy"))
        xy = raw[..., :2]
        coords.extend(xy[np.isfinite(xy).all(axis=2)].tolist())
        confidences.extend(raw[..., 2][np.isfinite(raw[..., 2])].tolist())
        if r["status"] == "usable":
            normalized, _, scale = sq.preprocess_left(raw)
            before = (raw[:, 1, 1] - raw[:, 0, 1]) / scale
            after = normalized[:, 1, 1] - normalized[:, 0, 1]
            drops.append(float(abs(before.max() - after.max())))
    coords = np.asarray(coords)
    summary.update({"videos": len(video_ids), "video_without_pose": sorted(video_ids - pose_ids),
                    "label_without_video": sorted(label_ids - video_ids),
                    "frame_count_mismatches": mismatches, "xy_min": coords.min(axis=0).tolist(),
                    "xy_max": coords.max(axis=0).tolist(),
                    "xy_outside_unit": int(np.any((coords < 0) | (coords > 1), axis=1).sum()),
                    "confidence_quantiles": np.percentile(confidences, [0, 5, 50, 95, 100]).tolist(),
                    "max_hip_knee_proxy_resample_difference": max(drops)})
    sq.save_json(DATA / "audit_summary.json", summary)
    # Hoja de contacto: vídeo y pose de la MISMA vista/ID, cinco instantes.
    ids = ("sq_001", "sq_190", "sq_300", "sq_375")
    fig, axes = plt.subplots(len(ids), 5, figsize=(16, 10))
    colors = ("red", "lime", "cyan", "yellow", "magenta")
    for row, ident in enumerate(ids):
        cap = cv2.VideoCapture(str(VIDEO / f"{ident}.mp4"))
        raw = np.load(run("temporal")["source"] / f"{ident}.npy")
        for col, frame in enumerate(np.linspace(0, len(raw)-1, 5).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Frame {ident}/{frame} ilegible")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            axes[row, col].imshow(image)
            for k, color in enumerate(colors):
                x, y = raw[frame, k, :2]
                if np.isfinite(x) and np.isfinite(y):
                    axes[row, col].scatter(x*width, y*height, c=color, s=18)
                    axes[row, col].text(x*width, y*height, str(k), color="white", fontsize=7)
            axes[row, col].set(title=f"{ident} frame {frame}", xticks=[], yticks=[])
        cap.release()
    fig.tight_layout()
    fig.savefig(DATA / "right_pose_contact_sheet.jpg", dpi=110)
    plt.close(fig)
    print("Auditoría right:", summary)


def prepare():
    guard(DATA / "split.npz")
    summary = json.loads((DATA / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("frame_count_mismatches") is None or summary["frame_count_mismatches"]:
        raise RuntimeError("Primero ejecutar auditoría completa de vídeos")
    sq.prepare("right")
    sq.save_json(DATA / "audit_summary.json", summary)
    with np.load(DATA / "split.npz") as split:
        roles = {role: split[f"ids_{role}"].astype(str).tolist() for role in ("train", "validation", "test")}
    if len(set(sum(roles.values(), []))) != sum(map(len, roles.values())):
        raise RuntimeError("ID duplicado entre particiones")
    sq.save_json(DATA / "membership.json", {"roles": roles, "split_sha256": digest(DATA / "split.npz"),
                                           "policy": "estratificado por video_id; sin mapeo entre vistas; no se ha demostrado equivalencia física de números entre cámaras"})
    for variant in VARIANTS:
        p = run(variant)
        guard(p["models"] / "config.json")
        p["models"].mkdir(parents=True, exist_ok=True)
        conf = {"exercise":"sq", "view":"right", "run_id":VARIANTS[variant],
                "source_split_sha256":digest(DATA / "split.npz"), "positive_class":"1=valido",
                "input_shape":[170,5,3], "schema":list(enumerate(sq.VIEW_CONFIGS["right"]["points"])),
                "scaler": "StandardScaler flattened (170*5*3)" if variant == "temporal" else "PointXYScaler per point XY across frames; confidence unchanged",
                "normalization":"fixed median hip center; median knee-ankle shank scale",
                "architecture":"training.common.train_utils.PoseCNN", "parameters":102017,
                "seed":42, "fold_seeds":[43,44,45,46,47], "threshold":0.5,
                "training":{"epochs":200,"batch":16,"patience":8,"adam_lr":0.001,"weight_decay":1e-4,
                            "scheduler_patience":4,"scheduler_factor":0.5,"augmentation_factor":3,
                            "transforms":["translation","scaling","noise","temporal_shift"]}}
        sq.save_json(p["models"] / "config.json", conf)


def development():
    with np.load(DATA / "split.npz") as split:
        X = np.concatenate([split["X_train"], split["X_validation"]])
        y = np.concatenate([split["y_train"], split["y_validation"]])
        ids = np.concatenate([split["ids_train"], split["ids_validation"]]).astype(str)
        n = len(split["ids_train"])
    return X, y, ids, n


def folds(ids, y):
    for fold, (outer_train, outer) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids,y), 1):
        fit, early = train_test_split(outer_train, test_size=.15, stratify=y[outer_train], random_state=42+fold)
        yield fold, fit, early, outer


def scale(variant, X, y, early, seed, path):
    scaler = StandardScaler().fit(X.reshape(len(X), -1)) if variant == "temporal" else PointXYScaler().fit(X)
    assert scaler.n_samples_seen_ == len(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(scaler, handle)
    aug, labels = augment_dataset(X,y,np.random.RandomState(seed),3,
                                  transforms=("translation","scaling","noise","temporal_shift"))
    assert np.array_equal(aug[::4,:,:,2],X[:,:,:,2])
    if variant == "point_xy":
        assert np.array_equal(transform(scaler,aug)[...,2],aug[...,2])
    return transform(scaler,aug),labels,transform(scaler,early),scaler


def cv(variant):
    X,y,ids,_ = development()
    p = run(variant)
    guard(p["results"] / "cv_predictions.csv")
    p["results"].mkdir(parents=True,exist_ok=True)
    rows, scores = [], []
    for fold, fit, early, outer in folds(ids,y):
        folder = p["models"] / "cv" / f"fold_{fold}"
        folder.mkdir(parents=True,exist_ok=True)
        members = [{"video_id":str(ids[i]),"label":int(y[i]),"role":role} for role,indices in
                   (("fit",fit),("early",early),("outer_validation",outer)) for i in indices]
        sq.save_csv(folder / "membership.csv",members,list(members[0]))
        X_fit,labels,X_early,scaler = scale(variant,X[fit],y[fit],X[early],42+fold,folder / "scaler.pkl")
        model,epoch,_ = sq.fit(X_fit,labels,X_early,y[early],42+fold,folder)
        _,pred,prob = sq.evaluate(model,transform(scaler,X[outer]),y[outer],42+fold)
        score = {"fold":fold,"best_epoch":epoch,"n_fit":len(fit),"n_early":len(early),**metrics(y[outer],prob)}
        scores.append(score)
        rows.extend({"fold":fold,"video_id":str(ids[i]),"true_label":int(y[i]),
                     "predicted_label":int(pred[j]),"probability_valid":float(prob[j])} for j,i in enumerate(outer))
        print(variant,"fold",fold,score,flush=True)
    assert len(rows)==len(ids) and {r["video_id"] for r in rows}==set(ids)
    sq.save_csv(p["results"] / "cv_predictions.csv",rows,list(rows[0]))
    sq.save_csv(p["results"] / "cv_metrics.csv",scores,list(scores[0]))
    keys = ("accuracy","balanced_accuracy","f1_macro","recall_invalid","precision_invalid",
            "f1_invalid","precision_valid","recall_valid","f1_valid","roc_auc","specificity","npv")
    sq.save_json(p["results"] / "cv_summary.json",{key:{"mean":float(np.mean([s[key] for s in scores])),
            "std":float(np.std([s[key] for s in scores]))} for key in keys})


def select():
    guard(CHOICE)
    a,b = [read_predictions(run(v)["results"] / "cv_predictions.csv") for v in VARIANTS]
    assert {(r["video_id"],r["fold"],r["true_label"]) for r in a} == {(r["video_id"],r["fold"],r["true_label"]) for r in b}
    old,new = [json.loads((run(v)["results"] / "cv_summary.json").read_text(encoding="utf-8")) for v in VARIANTS]
    # Regla fijada sin test: priorizar suma balanced acc + F1 macro; recall inválidos como desempate.
    ranking = lambda item:(item["balanced_accuracy"]["mean"]+item["f1_macro"]["mean"],item["recall_invalid"]["mean"])
    chosen = "point_xy" if ranking(new)>ranking(old) else "temporal"
    sq.save_json(CHOICE,{"selected":chosen,"run_id":VARIANTS[chosen],"rule":"max CV mean(balanced_accuracy)+mean(f1_macro); recall_invalid desempata",
                         "threshold":0.5,"reference_cv":old,"variant_cv":new,
                         "split_sha256":digest(DATA / "split.npz"),
                         "cv_sha256":{v:digest(run(v)["results"] / "cv_predictions.csv") for v in VARIANTS}})
    print("Selección en desarrollo:",chosen,ranking(old),ranking(new))


def selected():
    decision=json.loads(CHOICE.read_text(encoding="utf-8"))
    assert decision["split_sha256"]==digest(DATA / "split.npz")
    return decision["selected"]


def fit_final():
    variant=selected()
    p=run(variant)
    guard(p["models"] / "best_model.pth")
    X,y,ids,n=development()
    X_fit,labels,X_val,scaler=scale(variant,X[:n],y[:n],X[n:],42,p["models"] / "scaler.pkl")
    model,epoch,_=sq.fit(X_fit,labels,X_val,y[n:],42,p["models"])
    _,pred,prob=sq.evaluate(model,X_val,y[n:],42)
    rows=[{"video_id":str(i),"true_label":int(t),"predicted_label":int(q),"probability_valid":float(z)}
          for i,t,q,z in zip(ids[n:],y[n:],pred,prob)]
    sq.save_csv(p["results"] / "validation_predictions.csv",rows,list(rows[0]))
    sq.save_json(p["results"] / "validation_metrics.json",{"best_epoch":epoch,**metrics(y[n:],prob)})
    print("Final validation",epoch,metrics(y[n:],prob))


def evaluate_test():
    variant=selected()
    p=run(variant)
    guard(p["results"] / "test_metrics.json")
    with np.load(DATA / "split.npz") as split:
        X,y,ids=split["X_test"],split["y_test"],split["ids_test"].astype(str)
    with (p["models"] / "scaler.pkl").open("rb") as handle:
        scaler=pickle.load(handle)
    model=PoseCNN(T=170,K=5,C=3).to(sq.DEVICE)
    model.load_state_dict(torch.load(p["models"] / "best_model.pth",map_location=sq.DEVICE,weights_only=True))
    _,pred,prob=sq.evaluate(model,transform(scaler,X),y,42)
    rows=[{"video_id":str(i),"true_label":int(t),"predicted_label":int(q),"probability_valid":float(z)}
          for i,t,q,z in zip(ids,y,pred,prob)]
    sq.save_csv(p["results"] / "test_predictions.csv",rows,list(rows[0]))
    sq.save_json(p["results"] / "test_metrics.json",metrics(y,prob))
    print("Test reservado right",metrics(y,prob))


def report():
    variant=selected(); p=run(variant)
    audit=json.loads((DATA / "audit_summary.json").read_text(encoding="utf-8"))
    test=json.loads((p["results"] / "test_metrics.json").read_text(encoding="utf-8"))
    val=json.loads((p["results"] / "validation_metrics.json").read_text(encoding="utf-8"))
    cvdata={v:json.loads((run(v)["results"] / "cv_summary.json").read_text(encoding="utf-8")) for v in VARIANTS}
    rows=read_predictions(p["results"] / "test_predictions.csv")
    labels={r["video_id"]:r for r in sq.load_labels(p["labels"])}
    codes={}
    for code in sorted({code for r in rows for code in labels[r["video_id"]]["codigos"].split("|") if code}):
        subset=[r for r in rows if code in labels[r["video_id"]]["codigos"].split("|")]
        codes[code]={"appearances":len(subset),"errors":[r["video_id"] for r in subset if r["true_label"]!=r["predicted_label"]]}
    sq.save_json(p["results"] / "fault_codes_nonexclusive.json",codes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    history = read_predictions(p["models"] / "history.csv")
    for key in ("train_loss", "val_loss"):
        axes[0].plot([int(r["epoch"]) for r in history], [float(r[key]) for r in history], label=key)
    axes[0].set(xlabel="Época", ylabel="BCELoss", title="Fit final (validation interna)")
    axes[0].legend()
    for f in range(1, 6):
        curves = read_predictions(p["models"] / "cv" / f"fold_{f}" / "history.csv")
        axes[1].plot([int(r["epoch"]) for r in curves], [float(r["val_loss"]) for r in curves], label=f"fold {f}")
    axes[1].set(xlabel="Época", ylabel="BCELoss", title="Early validation por fold")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(p["results"] / "loss_curves.png")
    plt.close(fig)
    with np.load(DATA / "split.npz") as split:
        distribution={role:dict(Counter(split[f"y_{role}"].tolist())) for role in ("train","validation","test")}
    folds=read_predictions(p["results"] / "cv_metrics.csv")
    lines=["# SQ right — PoseCNN independiente", "",f"Run elegido solo en desarrollo: `{VARIANTS[variant]}`; referencia `baseline_t170_v2`, candidato `point_xy_v1`.",
           "0=inválido, 1=válido; umbral 0.5. Matriz [[TN,FP],[FN,TP]].",f"Distribución (0/1): {distribution}.",
           f"Auditoría: {audit['csv_rows']} etiquetas, {audit['videos']} vídeos y poses, {audit['quality_counts']}; faltan {audit['label_without_video']}.",
           f"Split: `training/data/sq/right/posecnn/membership.json`; 5 folds con fit/early/outer_validation en `training/models/sq/right/posecnn/runs/{VARIANTS[variant]}/cv/`.",
           "Arquitectura PoseCNN K5 C3 T170 (102017 parámetros); Adam 0.001, weight decay 1e-4, batch 16, 200 épocas máximo, patience 8; seeds 42, folds 43–47; train original x4 tras scaler.fit.",
           f"Mejor época final: {val['best_epoch']}.","", "| Métrica | CV temporal media ± std | CV XY media ± std | Validation elegido | Test elegido |", "|---|---:|---:|---:|---:|"]
    for key in ("accuracy","balanced_accuracy","precision_valid","recall_valid","f1_valid","precision_invalid","recall_invalid","f1_invalid","f1_macro","specificity","npv","roc_auc"):
        a,b=(cvdata[v][key] for v in VARIANTS)
        lines.append(f"| {key} | {a['mean']:.4f} ± {a['std']:.4f} | {b['mean']:.4f} ± {b['std']:.4f} | {val[key]:.4f} | {test[key]:.4f} |")
    lines.extend(["",f"Validation matrix: {val['matrix']}; test matrix: {test['matrix']}.","",
                  "| Fold | N (0/1) | Época | Balanced acc | F1 macro | Recall inválido | AUC | Matriz |","|---|---|---:|---:|---:|---:|---:|---|"])
    for f in folds:
        lines.append(f"| {f['fold']} | {f['n']} ({f['n_invalid']}/{f['n_valid']}) | {f['best_epoch']} | {float(f['balanced_accuracy']):.4f} | {float(f['f1_macro']):.4f} | {float(f['recall_invalid']):.4f} | {float(f['roc_auc']):.4f} | {f['matrix']} |")
    lines.extend(["","Códigos (apariciones NO exclusivas): `fault_codes_nonexclusive.json`; no sumar como vídeos distintos.",
                  "Predicciones y probabilidades por vídeo: `cv_predictions.csv`, `validation_predictions.csv`, `test_predictions.csv`.",
                  "Curvas: `loss_curves.png`; valores originales en `training/models/sq/right/posecnn/runs/<run>/history.csv` y `cv/fold_*/history.csv`.",
                  "", "Errores test (ID, verdad, predicción, p válido):"])
    lines.extend(f"- {r['video_id']}: {r['true_label']} → {r['predicted_label']}, {float(r['probability_valid']):.4f}" for r in rows if r["true_label"]!=r["predicted_label"])
    diagnostic=BASE / "results" / "sq" / "right" / "posecnn" / "development_diagnostics" / "report.md"
    if diagnostic.is_file():
        detail=["#"+line if line.startswith("#") else line for line in diagnostic.read_text(encoding="utf-8").splitlines()[2:]]
        lines.extend(["", "## Auditoría de errores CV de desarrollo (sin test)", "",
                      *detail])
    (p["results"] / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def verify():
    X,y,ids,n=development()
    membership=json.loads((DATA / "membership.json").read_text(encoding="utf-8"))
    assert membership["split_sha256"]==digest(DATA / "split.npz")
    assert ids.tolist()==membership["roles"]["train"]+membership["roles"]["validation"]
    assert len(set(sum(membership["roles"].values(),[])))==len(X)+len(membership["roles"]["test"])
    for variant in VARIANTS:
        p=run(variant)
        conf=json.loads((p["models"] / "config.json").read_text(encoding="utf-8"))
        assert conf["source_split_sha256"]==membership["split_sha256"]
        preds=read_predictions(p["results"] / "cv_predictions.csv")
        assert len(preds)==len(ids) and {r["video_id"] for r in preds}==set(ids)
        for fold,fit,early,outer in folds(ids,y):
            folder=p["models"] / "cv" / f"fold_{fold}"
            members=read_predictions(folder / "membership.csv")
            actual={(r["video_id"],r["role"],int(r["label"])) for r in members}
            expected={(str(ids[i]),role,int(y[i])) for role,indices in
                      (("fit",fit),("early",early),("outer_validation",outer)) for i in indices}
            assert actual==expected
            assert {r["video_id"] for r in preds if int(r["fold"])==fold}=={str(ids[i]) for i in outer}
            with (folder / "scaler.pkl").open("rb") as file:
                scaler=pickle.load(file)
            assert scaler.n_samples_seen_==len(fit)
            assert isinstance(scaler,StandardScaler if variant=="temporal" else PointXYScaler)
    choice=json.loads(CHOICE.read_text(encoding="utf-8"))
    assert choice["selected"]=="point_xy" and selected()=="point_xy"
    with (run("point_xy")["models"] / "scaler.pkl").open("rb") as file:
        assert pickle.load(file).n_samples_seen_==n
    with np.load(DATA / "split.npz") as split:
        test=read_predictions(run("point_xy")["results"] / "test_predictions.csv")
        assert [r["video_id"] for r in test]==split["ids_test"].astype(str).tolist()
        assert [int(r["true_label"]) for r in test]==split["y_test"].tolist()
    print("Verificado: split/test, ambos CV en los mismos folds, scaler solo train y selección congelada")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("audit","prepare","cv","select","fit-final","evaluate-test","report","verify"))
    parser.add_argument("--variant",choices=tuple(VARIANTS))
    args=parser.parse_args()
    if args.stage=="cv" and args.variant is None:
        parser.error("cv necesita --variant")
    {"audit":audit,"prepare":prepare,"cv":lambda:cv(args.variant),"select":select,
     "fit-final":fit_final,"evaluate-test":evaluate_test,"report":report,"verify":verify}[args.stage]()
