"""Análisis de predicciones congeladas, sin inferencia ni selección de modelos."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from .run import ROOT, LIMITACIONES, outdir, readj, savej, sha, verify, metrics, now, protect_inputs


def tabla(df):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v,(float,np.floating)) else str(v)
    return "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"]*len(df.columns)) + "|\n" + "\n".join("| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False,name=None))


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    centro=(p+z*z/(2*n))/d;radio=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(centro-radio),float(centro+radio)]


def bootstrap(y,prob,protocol):
    cfg=protocol["confidence_intervals"];rng=np.random.default_rng(cfg["seed"])
    indices=rng.integers(0,len(y),(cfg["replicates"],len(y)))
    yy=y[indices];keep=(yy.sum(1)>0)&(yy.sum(1)<len(y));indices=indices[keep];yy=yy[keep]
    result={};draws={}
    for model,p in prob.items():
        pred=p[indices]>=.5;pos=yy==1;neg=~pos
        tp=(pred&pos).sum(1);tn=(~pred&neg).sum(1);fp=(pred&neg).sum(1);fn=(~pred&pos).sum(1)
        f1=2*tp/(2*tp+fp+fn)
        samples=dict(accuracy=(tp+tn)/len(y),f1=f1,macro_f1=(f1+2*tn/(2*tn+fp+fn))/2,
            balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,specificity=tn/(tn+fp),
            roc_auc=np.array([roc_auc_score(y[ix],p[ix]) for ix in indices]))
        draws[model]=samples
        result[model]={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
        m=metrics(y,p);result[model]["fpr_wilson"]=wilson(m["fp"],m["tn"]+m["fp"])
        result[model]["specificity_wilson"]=wilson(m["tn"],m["tn"]+m["fp"])
    differences={}
    for a,b in protocol["paired_comparisons"]:
        differences[f"{b} minus {a}"]={k:dict(estimate=metrics(y,prob[b])[k]-metrics(y,prob[a])[k],
            ci95=np.quantile(draws[b][k]-draws[a][k],[.025,.975]).tolist()) for k in cfg["differences"]}
    return dict(policy=cfg,accepted=len(indices),discarded_single_class=int((~keep).sum()),models=result,paired_differences=differences)


def analyze(view):
    protocol,source=verify(view);out=outdir(view)
    state=readj(out/"execution_state.json");assert state["status"]=="TEST_INFERENCE_COMPLETE"
    assert sha(out/"test_predictions.csv")==state["prediction_sha256"]
    df=pd.read_csv(out/"test_predictions.csv");ids=protocol["test_ids"]
    prob={};y=None
    for model in protocol["models"]:
        sub=df[df.model==model];assert list(sub.video_id)==ids
        if y is None:y=sub.label.to_numpy()
        assert np.array_equal(y,sub.label)
        prob[model]=sub.probability_valid.to_numpy()
        assert np.array_equal(sub.predicted,prob[model]>=.5)
    scores=pd.DataFrame([dict(model=m,**metrics(y,p)) for m,p in prob.items()])
    old=pd.read_csv(out/"validation_original_metrics.csv")
    deltas=[]
    for m in prob:
        a=old.set_index("model").loc[m];b=scores.set_index("model").loc[m]
        deltas.append(dict(model=m,**{k+"_test_minus_validation":float(b[k]-a[k]) for k in ("accuracy","f1","macro_f1","roc_auc","specificity","fpr")}))
    pd.DataFrame(deltas).to_csv(out/"validation_test_differences.csv",index=False)
    cv=ROOT/"training/bp/experiments"/f"bp_{view}_04_cv_v1"
    cvmean=pd.read_csv(cv/"five_fold_stability.csv");cvfold=pd.read_csv(cv/"fold_metrics.csv");cvpool=pd.read_csv(cv/"oof_pooled_metrics.csv")
    for name,data in (("cv_mean_std",cvmean),("cv_folds",cvfold),("cv_pooled_oof",cvpool)):
        data.to_csv(out/f"{name}.csv",index=False)
    comparisons=[];idarray=np.array(ids)
    for a,b in protocol["paired_comparisons"]:
        pa=prob[a]>=.5;pb=prob[b]>=.5;ea=pa!=y;eb=pb!=y
        masks=dict(shared_errors=ea&eb,exclusive_a=ea&~eb,exclusive_b=~ea&eb,
            fp_corrected=(y==0)&ea&~eb,fp_introduced=(y==0)&~ea&eb,
            fn_corrected=(y==1)&ea&~eb,fn_introduced=(y==1)&~ea&eb,disagreements=pa!=pb)
        comparisons.append(dict(a=a,b=b,probability_pearson=float(np.corrcoef(prob[a],prob[b])[0,1]),
            counts={k:int(v.sum()) for k,v in masks.items()},ids={k:idarray[v].tolist() for k,v in masks.items()}))
    savej(out/"paired_analysis.json",comparisons)
    pairtable=pd.DataFrame([dict(a=r["a"],b=r["b"],pearson=r["probability_pearson"],**r["counts"]) for r in comparisons])
    pairtable.to_csv(out/"paired_analysis.csv",index=False)
    df[np.abs(df.probability_valid-.5)<=.05].to_csv(out/"near_threshold.csv",index=False)
    df[df.label!=df.predicted].to_csv(out/"errors_by_video.csv",index=False)
    intervals=bootstrap(y,prob,protocol);savej(out/"confidence_intervals.json",intervals)
    icrows=[dict(model=m,metric=k,lower=v[0],upper=v[1]) for m,vals in intervals["models"].items() for k,v in vals.items()]
    pd.DataFrame(icrows).to_csv(out/"confidence_intervals.csv",index=False)
    for m,p in prob.items():
        s=metrics(y,p);cm=np.array([[s["tn"],s["fp"]],[s["fn"],s["tp"]]])
        fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=18)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Inválido","Válido"],yticklabels=["Inválido","Válido"],xlabel="Predicción",ylabel="Etiqueta",title=f"BP_{view.upper()} — {m}")
        fig.tight_layout();fig.savefig(out/f"confusion_{m}.png",dpi=160);plt.close(fig)
    for kind in ("roc","precision_recall"):
        fig,ax=plt.subplots(figsize=(7,5))
        for m,p in prob.items():
            if kind=="roc":x,z,_=roc_curve(y,p)
            else:z,x,_=precision_recall_curve(y,p)
            ax.plot(x,z,label=m)
            pd.DataFrame(dict(x=x,y=z)).to_csv(out/f"{kind}_{m}.csv",index=False)
        if kind=="roc":ax.plot([0,1],[0,1],"k--")
        else:ax.axhline(y.mean(),color="gray",linestyle="--")
        ax.set(xlabel="FPR" if kind=="roc" else "Recall",ylabel="Sensibilidad" if kind=="roc" else "Precision",title=f"BP_{view.upper()} Test")
        ax.legend();fig.tight_layout();fig.savefig(out/f"{kind}.png",dpi=160);plt.close(fig)
    input_hashes={}
    for role in ("validation","test"):input_hashes.update(readj(out/f"{role}_input_hashes.json"))
    changed=[p for p,h in input_hashes.items() if sha(ROOT/p)!=h];assert not changed,changed
    verify(view)
    protect_inputs(view,after=True)
    integrity=dict(status="INTEGRITY_OK",at=now(),protected_files=len(source["protected_sha256"]),
        evaluated_inputs=len(input_hashes),changed=[],protocol_sha256=sha(out/f"bp_{view}_final_test_protocol_frozen.json"),
        checkpoints_unchanged=True,test_inference_runs=state["test_inference_runs"])
    savej(out/"integrity_verification.json",integrity)
    principal=protocol["principal"];s=scores.set_index("model").loc[principal];d=next(r for r in deltas if r["model"]==principal)
    text=[f"# Evaluación final BP_{view.upper()} Test V1",f"**{view.upper()}_FINAL_TEST_COMPLETE**. Protocolo congelado {protocol['frozen_utc']}; SHA-256 `{integrity['protocol_sha256']}`.",
        f"Evaluación única: {len(y)} videos, {int(y.sum())} válidos y {int((y==0).sum())} inválidos. Una predicción por levantamiento. Principal invariable: **{principal}**. Umbral 0.5; promedio tardío 50/50. Sin entrenamiento ni ajustes.",
        "## Fuentes y reproducción", "Ver `source_manifest.json`, `duplicate_audit.json`, `validation_reproducibility.json` e `integrity_verification.json`. Coincidencia de IDs, etiquetas, decisiones y métricas Validation comprobada, probabilidades dentro de tolerancias históricas. Los pesos originales no fueron reemplazados por los de CV.",
        "## Tabla A — Validation original",tabla(old),"## Tabla B — CV 5-Fold (media ± desviación poblacional)",tabla(cvmean),"### Pooled OOF",tabla(cvpool),"### Por fold",tabla(cvfold),
        "LEFT late_full no fue evaluada en la CV original; no se agrega retrospectivamente." if view=="left" else "La fusión tardía permanece como comparador secundario.",
        "## Tabla C — Test final",tabla(scores),"AP corresponde a Average Precision, no al área trapezoidal de PR. Positivo=1 válido; **FP=inválido aprobado**, FN=válido rechazado.",
        "## Diferencias Test menos Validation",tabla(pd.DataFrame(deltas)),
        "CV usa nuevos pesos por fold y menos datos en cada inner Train; Test usa checkpoints originales seleccionados. El promedio CV no es rendimiento del mismo checkpoint ni debe coincidir necesariamente con Test.",
        "## Errores pareados (B respecto a A)",tabla(pairtable),
        "IDs completos y correlaciones: `paired_analysis.json`; todos los errores: `errors_by_video.csv`; cercanía al umbral |p−0.5|≤0.05: `near_threshold.csv`.",
        "## Incertidumbre exploratoria 95%",tabla(pd.DataFrame(icrows)),
        f"Bootstrap pareado ordinario por video, {intervals['accepted']} réplicas aceptadas de 10000, seed 20261015; {intervals['discarded_single_class']} descartadas por una sola clase. Percentiles 2.5/97.5; mismos índices para todos los modelos. Wilson para FPR/especificidad. Sin agrupamiento acreditado ni corrección por multiplicidad; no se utiliza para seleccionar modelos."]
    for pair,vals in intervals["paired_differences"].items():
        text.append(f"### {pair}\n"+tabla(pd.DataFrame([dict(metric=k,delta=v['estimate'],lower=v['ci95'][0],upper=v['ci95'][1]) for k,v in vals.items()])))
    text.extend(["## Interpretación experimental",
        f"El principal obtiene Accuracy={s.accuracy:.4f}, F1={s.f1:.4f}, Macro-F1={s.macro_f1:.4f}, AUC={s.roc_auc:.4f}. Aprueba incorrectamente {int(s.fp)} de {int(s.tn+s.fp)} inválidos (FPR={s.fpr:.4f}); esta tasa debe figurar explícitamente como limitación arbitral.",
        f"Respecto a Validation, ΔAccuracy={d['accuracy_test_minus_validation']:+.4f}, ΔF1={d['f1_test_minus_validation']:+.4f}, ΔMacro-F1={d['macro_f1_test_minus_validation']:+.4f}, ΔAUC={d['roc_auc_test_minus_validation']:+.4f}. Una caída es compatible con optimismo de selección/sobreajuste y variabilidad de muestra; no identifica por sí sola la causa."])
    for base in (("pose","seg_full") if view=="left" else ("pose","late_sin_relacion")):
        b=scores.set_index("model").loc[base]
        text.append(f"Principal frente a {base}: ΔAccuracy={s.accuracy-b.accuracy:+.4f}, ΔMacro-F1={s.macro_f1-b.macro_f1:+.4f}, ΔFP={int(s.fp-b.fp):+d}. Son diferencias descriptivas; revisar sus IC pareados antes de afirmar superioridad.")
    text.extend(["No se cambia el candidato según Test. La falta de ventaja de una fusión no acredita equivalencia ni uso autónomo en competición.",
        "## Limitaciones metodológicas",*['- '+v for v in LIMITACIONES],
        "## Integridad y reproducibilidad",f"{integrity['protected_files']} artefactos protegidos y {integrity['evaluated_inputs']} entradas verificadas antes/después, sin cambios. Ver también `protected_inputs_before.json` y `protected_inputs_after.json` para MP4/RAW/keypoints/features completos.",
        "Código congelado: `training/bp/experiments/bp_lateral_final_test_v1/run.py` y `analyze.py`. Comandos: `python -B -m training.bp.experiments.bp_lateral_final_test_v1.run --view "+view+" --stage freeze`, luego `validation`, luego `test`; análisis `python -B -m training.bp.experiments.bp_lateral_final_test_v1.analyze --view "+view+"`. El evaluador rechaza sobrescribir protocolos o repetir inferencias ya guardadas. Los artefactos existentes permiten recalcular métricas sin volver a ejecutar modelos.",
        "![ROC](roc.png)\n\n![Precision Recall](precision_recall.png)"])
    for m in prob:text.append(f"![{m}](confusion_{m}.png)")
    (out/f"Reporte_Final_BP_{view.upper()}_Test_V1.md").write_text("\n\n".join(text),encoding="utf-8")
    savej(out/"final_status.json",dict(status=f"{view.upper()}_FINAL_TEST_COMPLETE",at=now(),**{k:state[k] for k in ('n','valid','invalid','test_inference_runs')}))
    print(view,"ANALYSIS_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--view",choices=["left","right"],required=True)
    analyze(p.parse_args().view)
