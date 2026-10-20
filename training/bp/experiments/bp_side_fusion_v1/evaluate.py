"""Métricas y comparaciones por video de 7 candidatos por vista, solo Validation."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.bp.experiments.bp_posecnn_common_v1.core import rows_csv, save_csv, save_json
from training.bp.experiments.bp_side_segcnn_v1.data import sha
from training.bp.experiments.bp_side_segcnn_v1.train import metrics
from .data import fusion_out, snapshot
from .train import CONFIG

NAMES=("pose","seg_full","seg_sin_relacion","late_full","late_sin_relacion",
       "embedding_full","embedding_sin_relacion")


def get_rows(file):
    rows=rows_csv(file)
    if not rows or len(rows)!=len({r["video_id"] for r in rows}):raise RuntimeError("Predicciones duplicadas o vacías")
    return {r["video_id"]:r for r in rows}


def paired_error(rows,variant,kind,baseline):
    pose="pose";seg="seg_"+variant;fus=kind+"_"+variant
    groups={"shared_error":[],"pose_only_error":[],"seg_only_error":[],"both_correct":[]}
    detail={"corrected_shared":[],"retained_shared":[],"corrected_pose_only":[],
            "corrected_seg_only":[],"new_error_both_correct":[],"fusion_errors":[],"fusion_fp":[]}
    for r in rows:
        vid=r["video_id"]
        pe=int(r[pose+"_error"]);se=int(r[seg+"_error"]);fe=int(r[fus+"_error"])
        cat="shared_error" if pe and se else ("pose_only_error" if pe else ("seg_only_error" if se else "both_correct"))
        groups[cat].append(vid)
        if cat=="shared_error":detail["retained_shared" if fe else "corrected_shared"].append(vid)
        if cat=="pose_only_error" and not fe:detail["corrected_pose_only"].append(vid)
        if cat=="seg_only_error" and not fe:detail["corrected_seg_only"].append(vid)
        if cat=="both_correct" and fe:detail["new_error_both_correct"].append(vid)
        if fe:detail["fusion_errors"].append(vid)
        if int(r["label"])==0 and int(r[fus+"_pred"])==1:detail["fusion_fp"].append(vid)
    # Consistencia de error compartido contra auditoría Validation preexistente.
    expected=set(baseline["paired"]["seg_"+variant]["shared_errors"])
    if set(groups["shared_error"])!=expected:raise RuntimeError("Auditoría de errores base cambió")
    return {"variant":variant,"kind":kind,"baseline_groups":groups,"fusion_effect":detail,
            "shared_count":len(groups["shared_error"]),"corrected_exclusive":
            len(detail["corrected_pose_only"])+len(detail["corrected_seg_only"]),
            "new_errors_count":len(detail["new_error_both_correct"]),
            "false_positive_count":len(detail["fusion_fp"])}


def evaluate(view):
    out=fusion_out(view)
    source=json.loads((out/"fusion_source_manifest.json").read_text(encoding="utf-8"))
    if source["source_sha256"]!=snapshot(view) or not (out/"heads_training_complete.json").exists():
        raise RuntimeError("Fuentes protegidas/training cambiaron")
    baseline=json.loads((out.parent/f"bp_{view}_02_posecnn_compatibility_v1" if view=="left" else
                         out.parent/"bp_right_02_posecnn_common_v1").joinpath("validation_pose_seg_comparison.json").read_text(encoding="utf-8"))
    base_path=out.parent/("bp_left_02_posecnn_compatibility_v1" if view=="left" else "bp_right_02_posecnn_common_v1")
    pose_path=base_path/("validation_predictions.csv" if view=="left" else "seed_1337/validation_predictions.csv")
    base=get_rows(pose_path)
    original_ids=sorted(base)
    expected_n=91 if view=="left" else 93
    if len(original_ids)!=expected_n:raise RuntimeError("Validation original distinto")
    predictions={"pose":base};variants_info={}
    for variant in ("full","sin_relacion"):
        seg_root=out.parent/f"bp_{view}_01_segcnn_v1"/variant
        selected_seg=json.loads((seg_root/"selected_checkpoint.json").read_text(encoding="utf-8"))
        predictions["seg_"+variant]=get_rows(seg_root/f'seed_{selected_seg["seed"]}'/"validation_predictions.csv")
        predictions["late_"+variant]=get_rows(out/f"late_fusion_{variant}"/"validation_predictions.csv")
        root=out/f"embedding_fusion_{variant}"
        selected=json.loads((root/"selected_checkpoint.json").read_text(encoding="utf-8"))
        per_seed=[]
        for seed in CONFIG["seeds"]:
            result=json.loads((root/f"seed_{seed}"/"metrics.json").read_text(encoding="utf-8"))
            per_seed.append(result)
        if selected["seed"]!=min(per_seed,key=lambda r:(r["validation_loss"],r["seed"]))["seed"]:
            raise RuntimeError("Selección de seed no utiliza Validation loss")
        predictions["embedding_"+variant]=get_rows(root/f'seed_{selected["seed"]}'/"validation_predictions.csv")
        variants_info[variant]={"selected_embedding_seed":selected["seed"],
                                "seed_metrics":per_seed,"seg_seed":selected_seg["seed"]}
    for name in NAMES:
        if set(predictions[name])!=set(original_ids):raise RuntimeError("Predicciones no pareadas: "+name)
    y=np.array([int(base[v]["label"]) for v in original_ids])
    rows=[];results={}
    for name in NAMES:
        pred=predictions[name]
        prob=np.array([float(pred[v]["probability_valid"]) for v in original_ids])
        if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)) or any(int(pred[v]["label"])!=int(y[i]) for i,v in enumerate(original_ids)):
            raise RuntimeError("Probabilidades/labels inválidos "+name)
        result=metrics(y,prob);result["FPR"]=result["FP"]/(result["FP"]+result["TN"])
        results[name]=result
        if name in ("pose","seg_full","seg_sin_relacion"):
            expected=baseline["metrics"][name]
        elif name.startswith("late_"):
            v=name.removeprefix("late_")
            expected=json.loads((out/f"late_fusion_{v}"/"validation_metrics.json").read_text(encoding="utf-8"))["metrics"]
        else:
            v=name.removeprefix("embedding_")
            sd=variants_info[v]["selected_embedding_seed"]
            expected=json.loads((out/f"embedding_fusion_{v}"/f"seed_{sd}"/"metrics.json").read_text(encoding="utf-8"))["validation_metrics"]
        if any(not np.isclose(result[k],expected[k],atol=1e-9,rtol=0) for k in
               ("accuracy","precision","recall","f1","macro_f1","balanced_accuracy","roc_auc","pr_auc","TN","FP","FN","TP")):
            raise RuntimeError("Métrica original distinta: "+name)
        if name.startswith("late_"):
            location=out/f'late_fusion_{name.removeprefix("late_")}'
            cm=np.asarray(result["confusion_matrix_0_1"],dtype=np.int64)
            np.save(location/"validation_confusion.npy",cm)
            fig,ax=plt.subplots(figsize=(4,4));ax.imshow(cm,cmap="Blues")
            for i in range(2):
                for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
            ax.set(xlabel="Predicción",ylabel="Real",xticks=[0,1],yticks=[0,1])
            fig.tight_layout();fig.savefig(location/"validation_confusion.png",dpi=130);plt.close(fig)
    for vid,label in zip(original_ids,y):
        item={"video_id":vid,"label":int(label)}
        for name in NAMES:
            p=float(predictions[name][vid]["probability_valid"])
            item.update({name+"_prob":p,name+"_pred":int(p>=.5),
                         name+"_error":int((p>=.5)!=label)})
        rows.append(item)
    save_csv(out/"fusion_validation_predictions.csv",rows)
    paired=[]
    for variant in ("full","sin_relacion"):
        for kind in ("late","embedding"):
            paired.append(paired_error(rows,variant,kind,baseline))
    save_json(out/"fusion_pairwise_analysis.json",{"view":view,"comparisons":paired,
              "oracle_accuracy_not_reported":True,"test_npy_opened":0})
    table=[]
    for name in NAMES:
        s=results[name]
        table.append({"candidate":name,"selected_seed":variants_info[name.split("_",1)[1]]["selected_embedding_seed"]
                      if name.startswith("embedding_") else "",
                      "accuracy":s["accuracy"],"precision":s["precision"],"recall":s["recall"],"f1":s["f1"],
                      "macro_f1":s["macro_f1"],"balanced_accuracy":s["balanced_accuracy"],
                      "roc_auc":s["roc_auc"],"pr_auc":s["pr_auc"],"sensitivity_valid":s["sensitivity_valid"],
                      "specificity_invalid":s["specificity_invalid"],"FPR":s["FPR"],
                      "TN":s["TN"],"FP":s["FP"],"FN":s["FN"],"TP":s["TP"]})
    save_csv(out/"fusion_experiments_comparison.csv",table)
    # Regla fijada en fusion_config.json ANTES de abrir Validation de fusión.
    complexity=lambda name:(1 if name=="pose" or name.startswith("seg_") else
                            (2 if name.startswith("late_") else 3))
    prior={"seg_full":0,"seg_sin_relacion":1,"pose":2,"late_full":3,
           "late_sin_relacion":4,"embedding_full":5,"embedding_sin_relacion":6}
    candidates=sorted(NAMES,key=lambda name:(-round(results[name]["macro_f1"],12),
                  -round(results[name]["specificity_invalid"],12),
                  -round(results[name]["balanced_accuracy"],12),complexity(name),prior[name]))
    winner=candidates[0];w=results[winner]
    chosen_head=None
    chosen_scaler=None
    if winner.startswith("embedding_"):
        variant=winner.removeprefix("embedding_")
        parent=out/f"embedding_fusion_{variant}"
        entry=json.loads((parent/"selected_checkpoint.json").read_text(encoding="utf-8"))
        file=parent/entry["checkpoint"]
        chosen_head={"path":str(file.resolve()),"sha256":sha(file),"seed":entry["seed"]}
        scale=parent/"embedding_scaler_train.npz"
        chosen_scaler={"path":str(scale.resolve()),"sha256":sha(scale),
                       "fit":"only Train embeddings, in-sample encoders"}
    elif winner.startswith("seg_"):
        variant=winner.removeprefix("seg_")
        meta=source["source_sha256"][f"seg_{variant}_checkpoint"]
        chosen_head={"path":meta["path"],"sha256":meta["sha256"],"seed":variants_info[variant]["seg_seed"]}
    else:
        pose_meta=source["source_sha256"]["pose_checkpoint"]
        chosen_head={"path":pose_meta["path"],"sha256":pose_meta["sha256"],"seed":42 if view=="left" else 1337}
    selection={"view":view,"winner":winner,"ranking":candidates,"criteria":CONFIG["candidate_selection"],
               "winner_validation":{"macro_f1":w["macro_f1"],"specificity_invalid":w["specificity_invalid"],
                                    "balanced_accuracy":w["balanced_accuracy"],"FP":w["FP"],"FPR":w["FPR"]},
               "winner_head_seed":variants_info[winner.split("_",1)[1]]["selected_embedding_seed"]
                   if winner.startswith("embedding_") else None,
               "frozen_checkpoint":chosen_head,"frozen_embedding_scaler":chosen_scaler,
               "backbone_checkpoint_sha256":{
                   "pose":source["source_sha256"]["pose_checkpoint"]["sha256"],
                   "seg_full":source["source_sha256"]["seg_full_checkpoint"]["sha256"],
                   "seg_sin_relacion":source["source_sha256"]["seg_sin_relacion_checkpoint"]["sha256"]},
               "test_npy_opened":0,"warning":"Validation ya utilizado para elegir bases/variantes/cabezas; no prueba externa"}
    save_json(out/"fusion_selected_candidate.json",selection)
    save_json(out/"fusion_validation_summary.json",{"view":view,"validation_n":expected_n,"metrics":results,
              "variants":variants_info,"test_npy_opened":0,"candidate":selection})
    print(view,"candidate",winner,"macro",round(w["macro_f1"],4),"FP",w["FP"],
          "late F1",round(results["late_full"]["f1"],4),round(results["late_sin_relacion"]["f1"],4),flush=True)
    return results


def main():
    evaluate("left");evaluate("right")


if __name__=="__main__":main()
