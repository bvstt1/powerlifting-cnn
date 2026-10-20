"""Verificación independiente de artefactos CV y reporte; no inferencia Test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve
from .run import OUT, ROOT, MODELS, guardar, sha, metricas, preparar_seg


def tabla(df):
    def valor(x):
        if isinstance(x,(float,np.floating)):
            return "N/D" if np.isnan(x) else f"{x:.4f}"
        return str(x).replace("|","/")
    filas=["| " + " | ".join(map(str,df.columns)) + " |",
           "| " + " | ".join(["---"]*len(df.columns)) + " |"]
    filas.extend("| " + " | ".join(valor(x) for x in row) + " |" for row in df.itertuples(index=False,name=None))
    return "\n".join(filas)


def main():
    df=pd.read_csv(OUT/"config/development_manifest.csv")
    oof=pd.read_csv(OUT/"cv_oof_predictions.csv")
    mf=pd.read_csv(OUT/"cv_metrics_by_fold.csv")
    summary=pd.read_csv(OUT/"cv_metrics_summary.csv").set_index("model")
    pooled=pd.read_csv(OUT/"cv_pooled_oof_metrics.csv")
    assignment=pd.read_csv(OUT/"fold_assignments.csv").set_index("video_id")
    with np.load(OUT/"config/development_resampled.npz",allow_pickle=True) as z:
        arrays=[z[k] for k in ("pose","pose_availability","features","availability","provenance")]
    canales=json.loads((OUT/"config/frozen_channels.json").read_text(encoding="utf-8"))
    hashes=json.loads((OUT/"config/source_hashes_before.json").read_text(encoding="utf-8"))
    revision=json.loads((OUT/"config/resume_revision.json").read_text(encoding="utf-8"))
    hashes[str((OUT/"run.py").relative_to(ROOT))]=revision["resume_script_sha256"]
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    assert len(oof)==2805 and not oof.duplicated(["model","video_id"]).any()
    assert np.isfinite(oof.probability).all() and oof.probability.between(0,1).all()
    assert np.array_equal(oof.predicted, (oof.probability>=.5).astype(int))
    assert set(oof.model)==set(MODELS)
    wide=oof.pivot(index="video_id",columns="model",values="probability").sort_index()
    labels=df.set_index("video_id").loc[wide.index,"label"].to_numpy()
    for nombre in MODELS:
        sub=oof[oof.model==nombre]
        assert set(sub.video_id)==set(df.video_id)
        assert np.array_equal(sub.label,df.set_index("video_id").loc[sub.video_id,"label"])
        assert np.array_equal(sub.fold,assignment.loc[sub.video_id,"fold"])
        calc=metricas(sub.label.to_numpy(),sub.probability.to_numpy())
        for k,v in calc.items():
            assert np.isclose(v,pooled.set_index("model").loc[nombre,k],atol=1e-12)
        for fold in range(1,6):
            s=sub[sub.fold==fold]
            calc=metricas(s.label.to_numpy(),s.probability.to_numpy())
            row=mf[(mf.model==nombre)&(mf.fold==fold)].iloc[0]
            for k,v in calc.items():
                assert np.isclose(v,row[k],atol=1e-12)
        for k in calc:
            assert np.isclose(summary.loc[nombre,k+"_mean"],mf[mf.model==nombre][k].mean())
            assert np.isclose(summary.loc[nombre,k+"_std"],mf[mf.model==nombre][k].std(ddof=1))
    for variante in ("full","sin_relacion"):
        assert np.allclose(wide["late_fusion_"+variante],.5*(wide.posecnn+wide["segcnn_"+variante]),atol=1e-15)
    trained=[]
    for fold in range(1,6):
        split=pd.read_csv(OUT/f"folds/fold_{fold}_split.csv")
        tr=np.flatnonzero(split.role=="inner_train")
        va=np.flatnonzero(split.role=="inner_validation")
        ev=np.flatnonzero(split.role=="outer_validation")
        assert not(set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev))
        assert len(tr)+len(va)+len(ev)==561
        assert set(split.iloc[ev].video_id)==set(assignment[assignment.fold==fold].index)
        for nombre in MODELS[:3]:
            folder=OUT/nombre/f"fold_{fold}"
            meta=json.loads((folder/"training_metadata.json").read_text(encoding="utf-8"))
            assert sha(folder/"best.pt")==meta["checkpoint_sha256"]
            hist=pd.read_csv(folder/"learning_curves.csv")
            assert meta["fresh_initialization"] and meta["epochs_run"]==len(hist)
            assert meta["best_epoch"]==int(hist.loc[hist.validation_loss.idxmin(),"epoch"])
            with np.load(folder/"scaler.npz") as stats:
                assert set(stats["fit_indices"])==set(tr)
                # Recalcular con el orden de ajuste guardado, exclusivamente inner train.
                idx=stats["fit_indices"]
                if nombre=="posecnn":
                    scaler=StandardScaler().fit(arrays[0][idx].reshape(len(idx),-1))
                    assert np.array_equal(stats["mean"],scaler.mean_)
                    assert np.array_equal(stats["scale"],scaler.scale_)
                    assert int(stats["n_samples"])==len(tr)
                else:
                    variante=nombre.removeprefix("segcnn_")
                    _,nuevo=preparar_seg(arrays,idx,canales[variante],variante)
                    assert all(np.array_equal(stats[k],v) for k,v in nuevo.items())
            trained.append(dict(model=nombre,fold=fold,**{k:meta[k] for k in ("best_epoch","epochs_run","parameters","seed")}))
    pd.DataFrame(trained).to_csv(OUT/"reports/training_summary.csv",index=False)
    pares=[("posecnn","segcnn_full"),("posecnn","segcnn_sin_relacion"),
        ("posecnn","late_fusion_full"),("posecnn","late_fusion_sin_relacion"),
        ("segcnn_full","late_fusion_full"),("segcnn_sin_relacion","late_fusion_sin_relacion")]
    analisis=[]; casos=[]
    for a,b in pares:
        pa,pb=wide[a].to_numpy()>=.5,wide[b].to_numpy()>=.5
        ea,eb=pa!=labels,pb!=labels
        masks=dict(shared_errors=ea&eb,exclusive_a=ea&~eb,exclusive_b=~ea&eb,
            fp_corrected=(labels==0)&ea&~eb,fp_introduced=(labels==0)&~ea&eb,
            fn_corrected=(labels==1)&ea&~eb,fn_introduced=(labels==1)&~ea&eb)
        analisis.append(dict(model_a=a,model_b=b,**{k:int(v.sum()) for k,v in masks.items()},
            probability_pearson=wide[a].corr(wide[b])))
        for categoria,mask in masks.items():
            for vid in wide.index[mask]:
                casos.append(dict(model_a=a,model_b=b,category=categoria,video_id=vid,
                    label=int(df.set_index("video_id").loc[vid,"label"]),p_a=wide.loc[vid,a],p_b=wide.loc[vid,b]))
    paired=pd.DataFrame(analisis)
    paired.to_csv(OUT/"cv_paired_error_analysis.csv",index=False)
    pd.DataFrame(casos).to_csv(OUT/"oof/paired_error_ids.csv",index=False)
    wide.assign(label=labels).to_csv(OUT/"oof/predictions_wide.csv")
    oof[oof.label!=oof.predicted].to_csv(OUT/"oof/errors_by_video.csv",index=False)
    for met in ("accuracy","f1","macro_f1","roc_auc","specificity"):
        fig,ax=plt.subplots(figsize=(10,5))
        for n in MODELS:
            s=mf[mf.model==n]
            ax.plot(s.fold,s[met],marker="o",label=n)
        ax.set(xlabel="Outer fold",ylabel=met,xticks=range(1,6)); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT/f"figures/folds_{met}.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for n in MODELS:
        fpr,tpr,_=roc_curve(labels,wide[n]); precision,recall,_=precision_recall_curve(labels,wide[n])
        axs[0].plot(fpr,tpr,label=n); axs[1].plot(recall,precision,label=n)
    axs[0].set(xlabel="FPR",ylabel="Sensibilidad",title="ROC pooled OOF")
    axs[1].set(xlabel="Recall",ylabel="Precision",title="PR pooled OOF (área reportada: AP)")
    for ax in axs: ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/"figures/pooled_roc_pr.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(18,4))
    for ax,n in zip(axs,MODELS):
        row=pooled.set_index("model").loc[n]
        cm=np.asarray([[row.tn,row.fp],[row.fn,row.tp]],dtype=int)
        ax.imshow(cm,cmap="Blues"); ax.set(title=n.replace("_","\n"),xlabel="Predicción",ylabel="Real",xticks=[0,1],yticks=[0,1])
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(OUT/"figures/pooled_confusion.png",dpi=180); plt.close(fig)
    histpath=ROOT/"training/bp/experiments/bp_front_03_fusion_v1/fusion_experiments_comparison.csv"
    historical=pd.read_csv(histpath).set_index("modelo")
    comparison=[]
    for n,old in zip(MODELS,["pose","seg_full","seg_sin","late_full","late_sin"]):
        row=historical.loc[old]
        comparison.append(dict(model=n,evaluation="A: Validation original",n=99,**{k:row[k] for k in ("accuracy","f1","macro_f1","roc_auc")}))
        comparison.append(dict(model=n,evaluation="B: CV media externa",n=561,**{k:summary.loc[n,k+"_mean"] for k in ("accuracy","f1","macro_f1","roc_auc")}))
        comparison.append(dict(model=n,evaluation="C: Test histórico (fuente ausente)",n=100,accuracy=np.nan,f1=np.nan,macro_f1=np.nan,roc_auc=np.nan))
    pd.DataFrame(comparison).to_csv(OUT/"reports/historical_comparison.csv",index=False)
    guardar(OUT/"reports/documentary_sources.json",dict(validation_source=str(histpath.relative_to(ROOT)),validation_sha256=sha(histpath),
        test_report_requested="training/bp/experiments/bp_front_05_final_test_v1/Reporte_Final_BP_FRONT_PoseCNN_SegCNN_Fusion_Test.md",
        test_report_exists=False,test_metrics_transcribed=False))
    integrity=json.loads((OUT/"cv_integrity_verification.json").read_text(encoding="utf-8"))
    integrity.update(status="FRONT_5FOLD_CV_COMPLETE",independent_verification_passed=True,
        scalers_recomputed_inner_train=True,all_fold_and_pooled_metrics_recomputed=True,
        fusion_formula_verified=True,completed_training_runs=15,interrupted_attempts=1,
        selected_cv_checkpoints={str((OUT/r["model"]/f'fold_{r["fold"]}'/"best.pt").relative_to(OUT)):
            sha(OUT/r["model"]/f'fold_{r["fold"]}'/"best.pt") for r in trained},
        documentation_limitation="Reporte Test autorizado ausente; métricas históricas Test no transcritas",
        source_code_revision="Solo runner CV: soporte de reanudación y lectura de cache. Fuentes históricas inalteradas.")
    blocked=OUT/"blocked.json"
    if blocked.exists():
        blocked.rename(OUT/"config/resolved_resume_cache_error.json")
    cols=["accuracy","f1","macro_f1","roc_auc","specificity"]
    compact=pd.DataFrame([dict(model=n,**{k:f'{summary.loc[n,k+"_mean"]:.4f} ± {summary.loc[n,k+"_std"]:.4f}' for k in cols}) for n in MODELS])
    fp=mf.pivot(index="model",columns="fold",values="fp").reindex(MODELS)
    fp["total"]=fp.sum(axis=1)
    lines=["# BP_FRONT — Validación cruzada estratificada 5-Fold V1", "",
        "**Estado: FRONT_5FOLD_CV_COMPLETE.** 15 entrenamientos nuevos completos; 561 videos development; 2.805 predicciones OOF (561 por configuración).",
        "## Protocolo y alcance",
        "Cohorte obligatoria filtrada a Train (462) + Validation (99): 339 válidos y 222 inválidos. Exclusiones bp_349 y bp_352 preservadas. No se abrieron secuencias ni predicciones Test. No se ejecutó YOLO ni se reutilizaron pesos históricos.",
        "StratifiedKFold(5, shuffle=True, random_state=42), orden por video_id. Sin group_id acreditado: no se inventaron correspondencias físicas. Auditoría de duplicados exactos del contenido de keypoints development: ninguno. Esta comprobación no equivale a acreditar independencia de atletas/sesiones.",
        "Dentro de cada outer train: StratifiedShuffleSplit(test_size=0.15, random_state=42). Escalado/imputación ajustados solo con inner train; early stopping y scheduler solo con inner validation. Outer validation se usa exclusivamente tras seleccionar el checkpoint. Unidad: levantamiento completo.",
        "Semillas fijas: Pose=42, FULL=1337, SIN=42. Pose: Adam, BCELoss, FP32, batch 16; Seg: AdamW, BCEWithLogitsLoss, AMP CUDA, batch 32. Todas: lr=0.001, weight_decay=1e-4, máximo 200 épocas, patience 15; scheduler factor 0.5/patience 5. Determinismo estricto activado. Sin búsqueda de semillas ni hiperparámetros.",
        "PoseCNN: (3,170,6), 118.529 parámetros; normalización por hombros, ausencia neutral/confianza cero, remuestreo entre vecinos observados y augmentation histórico con máscara (original + 3). Scaler por timestep×keypoint×canal, fit original antes del aumento. Seg FULL: (79,170), 56.449 parámetros; SIN: (65,170), 54.209 parámetros. Se congelaron los índices históricos (incluida exclusión de profile_ambiguous); no se recalcularon canales constantes para seleccionar features. Remuestreo V1 sin cruzar ausencias/cambios de procedencia o identidad. Mediana/media/std solo sobre observaciones de inner train. Máscaras/procedencias conservadas; no se aumentó SegCNN.",
        "Fusiones: 0.5×Pose + 0.5×Seg del mismo fold. Umbral fijo 0.5. Clase positiva válido; FP = inválido aprobado incorrectamente.",
        "## Distribución de particiones",tabla(pd.read_csv(OUT/"folds/class_distribution.csv")),
        "## Media ± desviación entre folds", "Desviación muestral (ddof=1), no intervalo de confianza. Folds con conjuntos de entrenamiento solapados; no son cinco experimentos independientes.",tabla(compact),
        "Todas las métricas, incluida Precision, Recall, Balanced Accuracy, AP, sensibilidad, FPR/FNR y conteos, están en cv_metrics_by_fold.csv y cv_metrics_summary.csv.",
        "## Falsos positivos",tabla(fp.reset_index()),
        "## Pooled OOF", "Una única predicción externa por video y modelo. AP = average_precision_score (integración escalonada precision–recall), no área trapezoidal. El pooled AUC compara scores de cinco redes diferentes; no debe confundirse con la media de AUC por fold.",tabla(pooled),
        "![Confusiones](figures/pooled_confusion.png)","![ROC y PR](figures/pooled_roc_pr.png)",
        "## Estabilidad y complementariedad",
        "Fusión SIN presenta la menor desviación de Accuracy, F1, Macro-F1 y ROC-AUC. No es universal: otras métricas pueden ordenar la estabilidad de otra manera. Fusión FULL logra mayor Accuracy/F1/Macro-F1 promedio; SIN logra mayor AUC promedio. Estas observaciones no alteran el candidato preseleccionado: **Pose + SIN**.",
        "FULL mejora descriptivamente frente a Pose y Seg FULL. SIN mejora frente a Pose; frente a Seg SIN, su ganancia media en F1 es pequeña, con menor variabilidad y menos FP pooled (41 frente a 51), a costa de más FN (28 frente a 21). No se declara superioridad estadística ni se ajusta ningún umbral.",
        "En la tabla, corregido/introducido se refiere a pasar de A a B. Errores exclusivos A son fallos de A acertados por B; exclusivos B, lo contrario. No se presenta rendimiento de oráculo.",tabla(paired),
        "IDs completos por categoría: oof/paired_error_ids.csv. Errores individuales: oof/errors_by_video.csv.",
        "![Variabilidad F1](figures/folds_f1.png)","![Variabilidad Macro-F1](figures/folds_macro_f1.png)","![Variabilidad AUC](figures/folds_roc_auc.png)",
        "## Comparación documental A/B/C",tabla(pd.DataFrame(comparison)),
        "**Limitación documental:** no existe el reporte autorizado training/bp/experiments/bp_front_05_final_test_v1/Reporte_Final_BP_FRONT_PoseCNN_SegCNN_Fusion_Test.md. Se deja Test como N/D; no se sustituyó por otros archivos Test, por predicciones individuales ni por números del prompt. N=100 es contexto del protocolo solicitado, no un conteo nuevo de Test.",
        "Validation original usa checkpoints seleccionados previamente y 99 videos; CV usa 15 redes nuevas, entrenamiento interno de 380–381 videos y evaluación externa de 112–113. Test histórico corresponde al protocolo anterior. Las diferencias pueden deberse al tamaño de entrenamiento, particiones, inicializaciones independientes, variabilidad de muestras y validación interna/externa. No son el mismo checkpoint ni la misma evaluación.",
        "## Integridad y reproducibilidad",
        "Se recalcularon independientemente todas las métricas por fold/pooled, fórmulas de fusión, pertenencia de OOF, ausencia de solapamientos y los 15 scalers exclusivamente con inner train. Checkpoints CV seleccionados: 15. Hashes de fuentes históricas, secuencias development y manifiestos sin cambios. Ningún checkpoint histórico se cargó para inferencia CV. Los archivos Test no se abrieron; por esa misma restricción no se calculó un nuevo hash de sus secuencias. No se acredita mediante esta ejecución una auditoría universal de archivos fuera de las fuentes registradas.",
        "Incidencia operativa: timeout del primer comando tras completar folds 1–3 y comenzar Pose fold 4. Se preservó el intento incompleto en posecnn/fold_4_interrupted_timeout; se reinició determinísticamente con idéntica semilla/partición (las pérdidas registradas de épocas 1/10/20/30 coinciden). Hubo también un error de lectura de IDs object del cache local al reanudar, corregido sin cambiar datos ni entrenamiento. 15 ejecuciones completas + 1 intento interrumpido; el intento interrumpido no produjo OOF. Cambios de runner documentados en config/resume_revision.json; logs en execution.log.",
        "Artefactos: config/source_hashes_before.json, config/protocol.json, config/frozen_channels.json, fold_assignments.csv, folds/fold_*_split.csv, scalers/checkpoints/curvas por fold, cv_*csv, cv_integrity_verification.json. El cache conserva identidad, T original, valores, máscaras y procedencias remuestreadas; los NPZ originales permanecen intactos.",
        "## Limitaciones metodológicas y conclusiones para la tesis",
        "1. CV retrospectiva: arquitecturas definidas usando development; Test ya conocido. Es una comprobación complementaria de robustez, no selección anidada completa anterior a Test. No restaura el carácter ciego ni elimina la exposición previa a development.",
        "2. Independencia física no acreditada por atleta/sesión/evento: estratificación por video y deduplicación exacta no resuelven esa limitación.",
        "3. SegCNN está parcialmente asistida por Pose durante la extracción geométrica; las modalidades no son totalmente independientes.",
        "4. Geometría 2D: relaciones barra–silueta no demuestran contacto barra-pecho.",
        "5. Independencia de entrenamiento YOLO frente a evaluación posterior no plenamente acreditada.",
        "6. Resultados descriptivos de cinco folds con semillas fijadas por arquitectura; no separan exhaustivamente variabilidad de partición e inicialización. Tampoco permiten atribuir toda diferencia FULL/SIN exclusivamente a los canales, pues las semillas congeladas difieren.",
        "Conclusión: la fusión tardía reduce la variabilidad principal y muestra complementariedad útil, con mejoras descriptivas sobre Pose. El control de FP sigue siendo relevante: ambas fusiones aprueban erróneamente 41/222 inválidos OOF. Pose + SIN conserva su condición de candidato histórico; no se seleccionan nuevos modelos ni se reentrenan modelos finales."]
    (OUT/"Reporte_BP_FRONT_CrossValidation_5Fold_V1.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    guardar(OUT/"cv_integrity_verification.json",integrity)
    guardar(OUT/"reports/output_hashes.json",{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name not in ("output_hashes.json",) and "__pycache__" not in str(p)})
    print("FRONT_5FOLD_CV_COMPLETE — verificación independiente aprobada")
    print(compact.to_string(index=False)); print(fp.to_string()); print(paired.to_string(index=False))


if __name__=="__main__":
    main()
