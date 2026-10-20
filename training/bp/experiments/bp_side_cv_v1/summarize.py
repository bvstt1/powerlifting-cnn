"""Integridad de folds, pooled OOF y análisis pareado; nunca usa Test."""
import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from training.bp.experiments.bp_posecnn_common_v1.core import ROOT, paths, rows_csv, save_csv, save_json
from training.bp.experiments.bp_side_segcnn_v1.data import sha
from .protocol import VARIANT, N_DEV, output, snapshot
from .train import CONFIG, measure


def paired(model_a,model_b,rows):
    a={r["video_id"]:r for r in rows if r["model"]==model_a}
    b={r["video_id"]:r for r in rows if r["model"]==model_b}
    if not a or set(a)!=set(b):raise RuntimeError("Pareo no es OOF completo")
    shared=[];only_a=[];only_b=[];disagreements=[]
    for vid in sorted(a):
        r,s=a[vid],b[vid]
        if int(r["label"])!=int(s["label"]) or int(r["fold"])!=int(s["fold"]):
            raise RuntimeError("Fold/label distintos en par")
        ea,eb=int(r["error"]),int(s["error"])
        if ea and eb:shared.append(vid)
        elif ea:only_a.append(vid)
        elif eb:only_b.append(vid)
        if r["predicted"]!=s["predicted"]:disagreements.append(vid)
    return {"a":model_a,"b":model_b,"n":len(a),"both_error":shared,"only_a_error":only_a,
            "only_b_error":only_b,"prediction_disagreements":disagreements,
            "both_error_n":len(shared),"only_a_n":len(only_a),"only_b_n":len(only_b),
            "disagreement_n":len(disagreements),
            "a_fp_ids":sorted(vid for vid,r in a.items() if int(r["false_positive"])),
            "b_fp_ids":sorted(vid for vid,r in b.items() if int(r["false_positive"]))}


def verify_fold(view,k,assignments,manifest):
    out=output(view);d=out/f"fold_{k:02d}"
    summary=json.loads((d/"fold_metrics.json").read_text(encoding="utf-8"))
    roles=json.loads((d/"roles_and_classes.json").read_text(encoding="utf-8"))
    if (summary["fold"]!=k or summary["view"]!=view or summary["test_npz_opened"]!=0 or
        not summary["backbones_initialized_from_scratch"] or not summary["outer_never_used_for_selection"] or
        summary["outer_predictions_sha256"]!=sha(d/"outer_predictions.csv")):
        raise RuntimeError("Fold mal documentado")
    byrole={role:{r["video_id"] for r in assignments if int(r["fold"])==k and r["role"]==role}
            for role in ("inner_train","inner_validation","outer_evaluation")}
    if (byrole["inner_train"]!=set(roles["inner_train_ids"]) or
        byrole["inner_validation"]!=set(roles["inner_validation_ids"]) or
        byrole["outer_evaluation"]!=set(roles["outer_eval_ids"]) or
        any(byrole[a]&byrole[b] for a,b in (("inner_train","outer_evaluation"),
                                             ("inner_validation","outer_evaluation"),
                                             ("inner_train","inner_validation")))):
        raise RuntimeError("Roles inner/outer mezclados")
    pose=summary["pose_training"];seg=summary["seg_training"]
    with open(d/"pose/scaler_inner_train.pkl","rb") as file:scaler=pickle.load(file)
    if (pose["scaler_n_samples_seen"]!=len(byrole["inner_train"]) or
        scaler.n_samples_seen_!=len(byrole["inner_train"]) or
        pose["scaler_sha256"]!=sha(d/"pose/scaler_inner_train.pkl") or
        pose["checkpoint_sha256"]!=sha(d/"pose/best_internal_validation.pt") or
        seg["preprocessing_sha256"]!=sha(d/"seg/inner_train_preprocessing.json") or
        seg["checkpoint_sha256"]!=sha(d/"seg/best_internal_validation.pt")):
        raise RuntimeError("Fuente/fit fold inconsistente")
    preprocessing=json.loads((d/"seg/inner_train_preprocessing.json").read_text(encoding="utf-8"))
    if (preprocessing["fitted_on_inner_train_only"]!=len(byrole["inner_train"]) or
        not preprocessing["outer_evaluation_not_used_in_fit"] or
        preprocessing["input_C_frozen"]!=(138 if view=="left" else 90)):
        raise RuntimeError("Scaler Seg incluye datos externos o C distinto")
    if view=="left":
        head=summary["fusion_training"]
        with np.load(d/"head/embedding_scaler_inner_train.npz",allow_pickle=False) as em:
            n=int(em["n_fit"]);size=len(em["mean"])
        if (head["embedding_scaler_fit_count"]!=n or n!=len(byrole["inner_train"]) or size!=192 or
            head["head_params"]!=6209 or head["checkpoint_sha256"]!=sha(d/"head/best_internal_validation.pt") or
            head["scaler_sha256"]!=sha(d/"head/embedding_scaler_inner_train.npz")):
            raise RuntimeError("Cabeza fold usa datos externos")
    return summary,byrole


def analyze(view):
    out=output(view);p=paths(view)
    manifest=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    if (manifest["view"]!=view or manifest["test_npy_opened"]!=0 or
        manifest["development_array_sha256"]!=sha(out/"development_unscaled.npz") or
        manifest["fold_assignments_sha256"]!=sha(out/"fold_assignments.csv") or
        manifest["source_sha256"]!=snapshot(view) or
        json.loads((out/"cv_config.json").read_text(encoding="utf-8"))!=CONFIG):
        raise RuntimeError("Fuentes/folds finales mutados")
    roles=rows_csv(out/"fold_assignments.csv")
    with np.load(out/"development_unscaled.npz",allow_pickle=False) as data:
        ids=data["ids"].tolist();labels=dict(zip(ids,map(int,data["y"])))
    expected_models=(("pose","seg_full","embedding_full") if view=="left" else
                     ("pose","seg_sin_relacion","late_sin_relacion"))
    if len(ids)!=N_DEV[view] or "bp_354" in ids and view=="right":
        raise RuntimeError("Development no corresponde a cohorte")
    data=[];fold_rows=[];outer_assignments=defaultdict(list)
    for k in range(1,6):
        info,members=verify_fold(view,k,roles,manifest)
        outer_assignments[k]=members["outer_evaluation"]
        fold_prediction=rows_csv(out/f"fold_{k:02d}"/"outer_predictions.csv")
        for model in expected_models:
            subset=[r for r in fold_prediction if r["model"]==model]
            if len(subset)!=len(members["outer_evaluation"]) or {r["video_id"] for r in subset}!=members["outer_evaluation"]:
                raise RuntimeError("Video outer no evaluado exactamente una vez")
            y=np.array([int(r["label"]) for r in subset]);prob=np.array([float(r["probability_valid"]) for r in subset])
            if any(int(r["label"])!=labels[r["video_id"]] for r in subset):
                raise RuntimeError("Label outer equivocado")
            calc=measure(y,prob)
            old=info["outer_metrics"][model]
            for name in ("accuracy","f1","macro_f1","balanced_accuracy","roc_auc","pr_auc","specificity_invalid","TN","FP","FN","TP"):
                if not np.isclose(calc[name],old[name],atol=1e-10,rtol=0):
                    raise RuntimeError("Fold métrica difiere: "+name)
            if not np.array_equal(np.load(out/f"fold_{k:02d}"/f"confusion_{model}.npy",allow_pickle=False),calc["confusion_matrix_0_1"]):
                raise RuntimeError("Matriz de confusión fold incorrecta")
            fold_rows.append({"fold":k,"model":model,"n":len(subset),"best_pose_epoch":info["pose_training"]["best_epoch"],
                              "best_seg_epoch":info["seg_training"]["best_epoch"],
                              "best_head_epoch":info["fusion_training"]["best_epoch"] if view=="left" else "fixed",
                              **{key:calc[key] for key in ("accuracy","precision","recall","f1","macro_f1",
                                   "balanced_accuracy","roc_auc","pr_auc","sensitivity_valid","specificity_invalid",
                                   "FPR","FNR","TN","FP","FN","TP")}})
        data.extend(fold_prediction)
    save_csv(out/"fold_metrics.csv",fold_rows)
    if len(data)!=N_DEV[view]*3:raise RuntimeError("OOF incompleto")
    pooled=[];oof_rows=[];per_model={}
    for model in expected_models:
        subset=sorted((r for r in data if r["model"]==model),key=lambda r:r["video_id"])
        if [r["video_id"] for r in subset]!=sorted(ids):raise RuntimeError("OOF no cubre todos los IDs")
        y=np.array([int(r["label"]) for r in subset]);p=np.array([float(r["probability_valid"]) for r in subset])
        metric=measure(y,p)
        per_model[model]=metric
        pooled.append({"model":model,"n":len(subset),**{key:metric[key] for key in ("accuracy","precision","recall",
                        "f1","macro_f1","balanced_accuracy","roc_auc","pr_auc","sensitivity_valid",
                        "specificity_invalid","FPR","FNR","TN","FP","FN","TP")}})
        oof_rows.extend(subset)
    save_csv(out/"oof_predictions.csv",oof_rows)
    save_csv(out/"oof_pooled_metrics.csv",pooled)
    results=[]
    for model in expected_models:
        subset=[r for r in fold_rows if r["model"]==model]
        line={"model":model,"n_folds":5}
        for key in ("accuracy","f1","macro_f1","balanced_accuracy","roc_auc","specificity_invalid",
                    "FPR","pr_auc"):
            vals=[r[key] for r in subset if r[key] is not None]
            line[f"{key}_mean"]=float(np.mean(vals)) if vals else None
            line[f"{key}_std"]=float(np.std(vals)) if vals else None
        line["false_positives_by_fold"]="/".join(str(r["FP"]) for r in subset)
        line["false_negatives_by_fold"]="/".join(str(r["FN"]) for r in subset)
        results.append(line)
    save_csv(out/"five_fold_stability.csv",results)
    comparable={name:{r["video_id"]:r for r in oof_rows if r["model"]==name} for name in expected_models}
    pairs=((expected_models[0],expected_models[1]),(expected_models[1],expected_models[2]),
           (expected_models[0],expected_models[2]))
    paired=[]
    for first,second in pairs:
        ea={vid for vid,r in comparable[first].items() if int(r["error"])}
        eb={vid for vid,r in comparable[second].items() if int(r["error"])}
        disag={vid for vid in ids if comparable[first][vid]["predicted"]!=comparable[second][vid]["predicted"]}
        item={"a":first,"b":second,"shared_errors":sorted(ea&eb),"only_a_errors":sorted(ea-eb),
              "only_b_errors":sorted(eb-ea),"disagreements":sorted(disag),
              "shared_n":len(ea&eb),"only_a_n":len(ea-eb),"only_b_n":len(eb-ea),
              "disagreements_n":len(disag),
              "fp_a":sorted(vid for vid,r in comparable[first].items() if int(r["false_positive"])),
              "fp_b":sorted(vid for vid,r in comparable[second].items() if int(r["false_positive"])),
              "per_fold":[{"fold":k,"shared_errors":len((ea&eb)&outer_assignments[k]),
                            "only_a":len((ea-eb)&outer_assignments[k]),
                            "only_b":len((eb-ea)&outer_assignments[k]),
                            "disagreements":len(disag&outer_assignments[k])} for k in range(1,6)]}
        paired.append(item)
    save_json(out/"oof_paired_comparisons.json",{"view":view,"pairs":paired,"no_oracle_performance":True})
    fold_by_id={r["video_id"]:int(r["fold"]) for r in oof_rows if r["model"]==expected_models[0]}
    save_csv(out/"oof_paired_by_video.csv",[{"video_id":vid,"label":labels[vid],"fold":fold_by_id[vid],
        **{f'{model}_{field}':comparable[model][vid][field] for model in expected_models
           for field in ("probability_valid","predicted","error","false_positive")}}
        for vid in sorted(ids)])
    summary={"view":view,"development_n":N_DEV[view],"folds":5,"group_policy":manifest["grouping"],
             "pooled_oof":per_model,"fold_stability":results,
             "class_counts":dict(Counter(map(int,labels.values()))),
             "test_npy_opened":0,"test_seg_npz_opened":0,"source_sha256_after":snapshot(view),
             "source_hashes_unchanged":snapshot(view)==manifest["source_sha256"],
             "preselected_final_candidate_unchanged":True,"folds_only_development":True,
             "conditional_cv_not_fully_nested":True}
    if not summary["source_hashes_unchanged"]:raise RuntimeError("Fuente protegida mutada")
    save_json(out/"cv_integrity.json",summary)
    print(view,"5-fold OOF",{k:round(v["f1"],4) for k,v in per_model.items()},
          "FP",{k:v["FP"] for k,v in per_model.items()},flush=True)
    return summary


def main():
    analyze("left");analyze("right")


if __name__=="__main__":main()
