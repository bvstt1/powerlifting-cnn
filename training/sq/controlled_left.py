"""SQ left: comparación controlada del scaler XY por punto con left v2.

python -m training.sq.controlled_left --stage prepare
python -m training.sq.controlled_left --stage cv
python -m training.sq.controlled_left --stage select
python -m training.sq.controlled_left --stage fit-final
python -m training.sq.controlled_left --stage evaluate-test  # solo si seleccionado
"""
import argparse
import csv
import json
import pickle
import platform

import numpy as np
import sklearn
import torch

from training.common.train_utils import PoseCNN, augment_dataset
from training.sq import train_posecnn as sq
from training.sq.controlled_front import PointXYScaler, fold_indices, transform
from training.sq.run_analysis import digest, metrics, read_predictions


BASE = sq.paths("left", "baseline_t170_v2")
RUN_ID = "point_xy_v1"
RUN = sq.paths("left", RUN_ID)
RUN["data"] = sq.ROOT / "training" / "data" / "sq" / "left" / "posecnn" / "runs" / RUN_ID
PROVENANCE = BASE["models"] / "provenance"
DECISION = RUN["results"] / "selection_development_only.json"
FIELDS = ("balanced_accuracy", "f1_macro", "recall_invalid", "precision_invalid", "f1_invalid",
          "precision_valid", "recall_valid", "f1_valid", "roc_auc", "accuracy")


def frozen():
    manifest = json.loads((PROVENANCE / "manifest.json").read_text(encoding="utf-8"))
    if digest(PROVENANCE / "split.npz") != manifest["sha256"]["split.npz"]:
        raise RuntimeError("Split congelado de left v2 alterado")
    return manifest


def prepare():
    manifest = frozen()
    RUN["data"].mkdir(parents=True, exist_ok=True)
    target = RUN["data"] / "development.npz"
    if target.exists():
        raise FileExistsError(target)
    with np.load(PROVENANCE / "split.npz") as ref:
        # Se accede solamente a train/validation. Test permanece sellado.
        X = np.concatenate([ref["X_train"], ref["X_validation"]]).astype(np.float32)
        y = np.concatenate([ref["y_train"], ref["y_validation"]]).astype(int)
        ids = np.concatenate([ref["ids_train"], ref["ids_validation"]]).astype(str)
        n = len(ref["ids_train"])
    assert ids.tolist() == manifest["split_ids"]["train"] + manifest["split_ids"]["validation"]
    assert X.shape == (314, 170, 5, 3) and np.isfinite(X).all()
    assert not set(ids) & set(manifest["split_ids"]["test"])
    assert len(list(fold_indices(ids, y, PROVENANCE))) == 5
    np.savez_compressed(target, X=X, y=y, ids=ids, n_train=n)
    model = PoseCNN(T=170, K=5, C=3)
    config = {"run_id": RUN_ID, "view": "sq_left", "reference": "baseline_t170_v2",
              "only_change": "PointXYScaler: XY per landmark shared across 170 frames; confidence unchanged",
              "input_shape": [170,5,3], "architecture": "training.common.train_utils.PoseCNN",
              "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
              "seed":42, "fold_seeds":list(range(43,48)), "threshold":0.5,
              "hyperparameters": {"epochs_max":200,"early_patience":8,"scheduler_patience":4,
                                  "learning_rate":0.001,"weight_decay":0.0001,"batch":16,
                                  "augmentation_factor":3,"augmentation": ["translation","scaling","noise","temporal_shift"]},
              "source_split_sha256":manifest["sha256"]["split.npz"],
              "reference_fold_sha256":{key:item["sha256"] for key,item in manifest["folds"].items()},
              "test_ids_only":manifest["split_ids"]["test"],
              "versions":{"python":platform.python_version(),"torch":torch.__version__,
                          "numpy":np.__version__,"sklearn":sklearn.__version__}}
    RUN["models"].mkdir(parents=True, exist_ok=True)
    sq.save_json(RUN["data"] / "config.json", config)
    sq.save_json(RUN["models"] / "config.json", config)
    print("Desarrollo SQ left aislado",X.shape,"test no leído",target)


def scale_fit(X_fit, y_fit, X_early, seed, path):
    scaler = PointXYScaler().fit(X_fit)
    assert scaler.n_samples_seen_ == len(X_fit)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(scaler, stream)
    augmented, labels = augment_dataset(X_fit, y_fit, np.random.RandomState(seed), 3,
                                        transforms=("translation","scaling","noise","temporal_shift"))
    assert np.array_equal(augmented[::4,:,:,2],X_fit[:,:,:,2])
    return transform(scaler,augmented),labels,transform(scaler,X_early),scaler


def summarize(rows):
    folds = []
    for fold in range(1,6):
        subset = [r for r in rows if int(r["fold"])==fold]
        folds.append({"fold":fold,**metrics([r["true_label"] for r in subset],
                                             [r["probability_valid"] for r in subset])})
    summary = {k:{"mean":float(np.mean([r[k] for r in folds])),
                  "std":float(np.std([r[k] for r in folds]))} for k in FIELDS}
    return folds, summary


def cv():
    target = RUN["results"] / "cv_predictions.csv"
    if target.exists():
        raise FileExistsError(target)
    with np.load(RUN["data"] / "development.npz") as dev:
        X,y,ids = dev["X"],dev["y"],dev["ids"].astype(str)
    RUN["results"].mkdir(parents=True,exist_ok=True)
    rows = []
    for fold, fit, early, outer in fold_indices(ids,y,PROVENANCE):
        folder = RUN["models"] / "cv" / f"fold_{fold}"
        X_fit,y_fit,X_early,scaler = scale_fit(X[fit],y[fit],X[early],42+fold,folder / "scaler.pkl")
        model,best_epoch,_ = sq.fit(X_fit,y_fit,X_early,y[early],42+fold,folder,patience=8)
        _,pred,probs = sq.evaluate(model,transform(scaler,X[outer]),y[outer],42+fold)
        rows.extend({"fold":fold,"best_epoch":best_epoch,"video_id":ids[i],"true_label":int(y[i]),
                     "predicted_label":int(pred[j]),"probability_valid":float(probs[j])}
                    for j,i in enumerate(outer))
        fold_metrics = metrics(y[outer],probs)
        print(f"fold {fold}: recall inválidos {fold_metrics['recall_invalid']:.4f}, "
              f"F1 macro {fold_metrics['f1_macro']:.4f}, válidos {fold_metrics['recall_valid']:.4f}")
    assert len(rows)==len(ids) and {r["video_id"] for r in rows}==set(ids)
    sq.save_csv(target,rows,list(rows[0]))
    folds,summary = summarize(rows)
    sq.save_csv(RUN["results"] / "cv_metrics.csv",folds,list(folds[0]))
    sq.save_json(RUN["results"] / "cv_summary.json",summary)
    print("CV",summary)


def select():
    if DECISION.exists():
        raise FileExistsError(DECISION)
    frozen()
    base_rows = read_predictions(BASE["results"] / "cv_predictions.csv")
    new_rows = read_predictions(RUN["results"] / "cv_predictions.csv")
    assert {r["video_id"] for r in base_rows}=={r["video_id"] for r in new_rows}
    a = {r["video_id"]:r for r in base_rows}; b = {r["video_id"]:r for r in new_rows}
    assert all(a[i]["true_label"]==b[i]["true_label"] and a[i]["fold"]==b[i]["fold"] for i in a)
    old_folds,old = summarize(base_rows)
    new_folds,new = summarize(new_rows)
    # Prioridad simultánea: menos inválidos aceptados sin sacrificar
    # balanced accuracy/F1 macro ni ocultar un desplome de F1 válidos.
    chosen = (new["recall_invalid"]["mean"] > old["recall_invalid"]["mean"] and
              new["balanced_accuracy"]["mean"] >= old["balanced_accuracy"]["mean"] and
              new["f1_macro"]["mean"] >= old["f1_macro"]["mean"] and
              new["f1_valid"]["mean"] >= old["f1_valid"]["mean"]-0.03)
    old_fp = {i for i in a if a[i]["true_label"]=="0" and a[i]["predicted_label"]=="1"}
    new_fp = {i for i in b if b[i]["true_label"]=="0" and b[i]["predicted_label"]=="1"}
    old_fn = {i for i in a if a[i]["true_label"]=="1" and a[i]["predicted_label"]=="0"}
    new_fn = {i for i in b if b[i]["true_label"]=="1" and b[i]["predicted_label"]=="0"}
    result = {"selected":"point_xy_v1" if chosen else "baseline_t170_v2",
              "rule":"CV: recall inválido sube; balanced accuracy y F1 macro no bajan; F1 válido no baja >0.03",
              "threshold":0.5,"reference_cv":old,"variant_cv":new,
              "reference_folds":old_folds,"variant_folds":new_folds,
              "corrected_false_valid":sorted(old_fp-new_fp),
              "persistent_false_valid":sorted(old_fp & new_fp),
              "new_false_valid":sorted(new_fp-old_fp),
              "corrected_false_invalid":sorted(old_fn-new_fn),
              "persistent_false_invalid":sorted(old_fn & new_fn),
              "new_false_invalid":sorted(new_fn-old_fn),
              "cv_probabilities_sha256":{"baseline":digest(BASE["results"]/"cv_predictions.csv"),
                                         "variant":digest(RUN["results"]/"cv_predictions.csv")},
              "source_split_sha256":frozen()["sha256"]["split.npz"],
              "test_policy":"left v2 test consultado antes: si se evalúa, exploratorio"}
    sq.save_json(DECISION,result)
    print("Selección solo desarrollo:",result["selected"],"recall invalid",old["recall_invalid"],new["recall_invalid"],
          "corregidos/persistentes/nuevos",len(old_fp-new_fp),len(old_fp & new_fp),len(new_fp-old_fp))


def fit_final():
    if not DECISION.exists():
        raise RuntimeError("Primero congelar selección basada en CV")
    if (RUN["models"] / "best_model.pth").exists():
        raise FileExistsError("Checkpoint final ya existe")
    with np.load(RUN["data"] / "development.npz") as dev:
        X,y,ids,n = dev["X"],dev["y"],dev["ids"],int(dev["n_train"])
    X_fit,y_fit,X_val,scaler = scale_fit(X[:n],y[:n],X[n:],42,RUN["models"] / "scaler.pkl")
    model,epoch,_ = sq.fit(X_fit,y_fit,X_val,y[n:],42,RUN["models"],patience=8)
    _,pred,probs = sq.evaluate(model,X_val,y[n:],42)
    RUN["results"].mkdir(parents=True,exist_ok=True)
    sq.save_csv(RUN["results"] / "validation_predictions.csv",
                [{"video_id":str(i),"true_label":int(label),"predicted_label":int(pr),"probability_valid":float(prob)}
                 for i,label,pr,prob in zip(ids[n:],y[n:],pred,probs)],
                ["video_id","true_label","predicted_label","probability_valid"])
    sq.save_json(RUN["results"] / "validation_metrics.json",{"best_epoch":epoch,**metrics(y[n:],probs)})
    print("Entrenamiento final en train+validation internos, sin test",epoch,metrics(y[n:],probs))


def evaluate_test():
    decision=json.loads(DECISION.read_text(encoding="utf-8"))
    if decision["selected"]!=RUN_ID:
        raise RuntimeError("La variante no fue seleccionada: test reservado; baseline v2 sigue vigente")
    if (RUN["results"] / "test_metrics.json").exists():
        raise FileExistsError("Test ya evaluado")
    assert decision["source_split_sha256"]==frozen()["sha256"]["split.npz"]
    with np.load(PROVENANCE / "split.npz") as split:
        ids=split["ids_test"].astype(str)
        y=split["y_test"].astype(int)
        X=split["X_test"].astype(np.float32)
    with (RUN["models"] / "scaler.pkl").open("rb") as file:
        scaler=pickle.load(file)
    assert isinstance(scaler,PointXYScaler)
    model=PoseCNN(T=170,K=5,C=3).to(sq.DEVICE)
    model.load_state_dict(torch.load(RUN["models"] / "best_model.pth",map_location=sq.DEVICE,weights_only=True))
    _,pred,probs=sq.evaluate(model,transform(scaler,X),y,42)
    sq.save_csv(RUN["results"] / "test_predictions.csv",
                [{"video_id":str(i),"true_label":int(label),"predicted_label":int(pr),"probability_valid":float(prob)}
                 for i,label,pr,prob in zip(ids,y,pred,probs)],
                ["video_id","true_label","predicted_label","probability_valid"])
    sq.save_json(RUN["results"] / "test_metrics.json",metrics(y,probs))
    print("Test exploratorio left (ya consultado en v2):",metrics(y,probs))


def verify():
    manifest=frozen()
    with np.load(RUN["data"] / "development.npz") as dev:
        X,y,ids=dev["X"],dev["y"],dev["ids"].astype(str)
        assert X.shape==(314,170,5,3) and np.isfinite(X).all()
        assert ids.tolist()==manifest["split_ids"]["train"]+manifest["split_ids"]["validation"]
        assert not set(ids) & set(manifest["split_ids"]["test"])
        folds=list(fold_indices(ids,y,PROVENANCE))
    for fold,fit,early,outer in folds:
        with (RUN["models"] / "cv" / f"fold_{fold}" / "scaler.pkl").open("rb") as file:
            scaler=pickle.load(file)
        assert isinstance(scaler,PointXYScaler) and scaler.n_samples_seen_==len(fit)
        assert scaler.mean_.shape==(5,2) and scaler.scale_.shape==(5,2)
    config=json.loads((RUN["models"] / "config.json").read_text(encoding="utf-8"))
    assert config["trainable_parameters"]==102017
    assert config["source_split_sha256"]==manifest["sha256"]["split.npz"]
    print("Verificado: 314 desarrollo, K5, mismos IDs/folds, scaler por punto fit train original, 102017 parámetros")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("prepare","cv","select","fit-final","evaluate-test","verify"))
    args=parser.parse_args()
    {"prepare":prepare,"cv":cv,"select":select,"fit-final":fit_final,"evaluate-test":evaluate_test,"verify":verify}[args.stage]()
