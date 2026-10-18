"""Comparación por video exclusivamente en Validation; nunca carga Test NPZ."""
import csv
import json
from collections import Counter

import numpy as np

from .data import ROOT, RELATION_EXCLUDED, cohort, paths, sha
from .train import CONFIG, metrics, save_csv, save_json


def predictions(path):
    with open(path,encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len({r["video_id"] for r in rows}):raise ValueError("Predicciones duplicadas")
    return {r["video_id"]:r for r in rows}


def source_quality(view,ids,source):
    result={}
    for vid in ids:
        # Solo video IDs Validation; jamás se abren secuencias Test.
        with np.load(source/"sequences"/f"{vid}_temporal.npz",allow_pickle=False) as z:
            t=len(z["frame_index"]);bar=z["bar_state"];rel=z["relation_state"]
            mask=z["availability"].astype(bool);names=z["feature_names"].tolist()
            if str(z["view"])!=view:raise ValueError("Vista cruzada")
            rate=lambda a:round(float(np.mean(a)),6)
            result[vid]={"bar_corrob_fraction":rate(bar==1),"bar_unverified_fraction":rate(np.isin(bar,[2,4])),
                         "bar_ambiguous_fraction":rate(bar==3),
                         "projection_fraction":rate(np.isin(rel,[2,3])),
                         "far_projection_fraction":rate(rel==3),
                         "bench_missing_fraction":rate(z["raw_class_counts"][:,1]==0),
                         "person_missing_fraction":rate(z["raw_class_counts"][:,2]==0),
                         "velocity_available_fraction":rate(mask[:,names.index("bar_vy_h_s")]),
                         "T":t}
    return result


def group_analysis(pred,quality):
    # Subgrupos descriptivos prefijados; no se usan para ajustar threshold/modelo.
    groups={"bar_corrob_ge50pct":lambda q:q["bar_corrob_fraction"]>=.5,
            "bar_corrob_lt50pct":lambda q:q["bar_corrob_fraction"]<.5,
            "bar_ambiguous_any":lambda q:q["bar_ambiguous_fraction"]>0,
            "bar_unverified_any":lambda q:q["bar_unverified_fraction"]>0,
            "projection_ge80pct":lambda q:q["projection_fraction"]>=.8,
            "projection_lt80pct":lambda q:q["projection_fraction"]<.8,
            "far_projection_any":lambda q:q["far_projection_fraction"]>0,
            "bench_missing_any":lambda q:q["bench_missing_fraction"]>0,
            "person_missing_any":lambda q:q["person_missing_fraction"]>0,
            "velocity_available_lt20pct":lambda q:q["velocity_available_fraction"]<.2}
    rows=[]
    for label,fn in groups.items():
        rs=[r for vid,r in pred.items() if fn(quality[vid])]
        e=sum(int(x["error"]) for x in rs)
        rows.append({"subset":label,"n":len(rs),"errors":e,"error_rate":e/len(rs) if rs else None,
                     "FP":sum(int(x["false_positive"]) for x in rs),
                     "FN":e-sum(int(x["false_positive"]) for x in rs)})
    return rows


def pair_prediction(full,abl,quality,view,seed_label):
    if set(full)!=set(abl):raise RuntimeError("FULL/ablación no pareados")
    paired=[]
    for vid in sorted(full):
        a,b=full[vid],abl[vid]
        if a["label"]!=b["label"]:raise RuntimeError("Labels distintos")
        paired.append({"video_id":vid,"label":int(a["label"]),"full_prob":float(a["probability_valid"]),
                       "full_pred":int(a["predicted"]),"full_error":int(a["error"]),
                       "sin_relacion_prob":float(b["probability_valid"]),"sin_relacion_pred":int(b["predicted"]),
                       "sin_relacion_error":int(b["error"]),
                       **quality[vid]})
    summary={"view":view,"pair":seed_label,"videos":len(paired),
             "both_correct":sum(not r["full_error"] and not r["sin_relacion_error"] for r in paired),
             "both_wrong":sum(r["full_error"] and r["sin_relacion_error"] for r in paired),
             "only_full_wrong":sum(r["full_error"] and not r["sin_relacion_error"] for r in paired),
             "only_sin_relacion_wrong":sum(r["sin_relacion_error"] and not r["full_error"] for r in paired),
             "disagreements":sum(r["full_pred"]!=r["sin_relacion_pred"] for r in paired),
             "full_fp_ids":[r["video_id"] for r in paired if r["label"]==0 and r["full_pred"]==1],
             "sin_relacion_fp_ids":[r["video_id"] for r in paired if r["label"]==0 and r["sin_relacion_pred"]==1]}
    return paired,summary


def report_view(view):
    groups,names,p,info=cohort(view)
    out=p["out"];source=p["source"]
    manifest=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    if not (out/"training_complete.json").exists() or manifest["test_npz_opened"]!=0:
        raise RuntimeError("Entrenamiento incompleto")
    expected={r["video_id"] for r in groups["validation"]}
    quality=source_quality(view,expected,source)
    save_csv(out/"validation_source_quality.csv",[{"video_id":vid,**q} for vid,q in sorted(quality.items())])
    models={}; all_predictions={}
    for variant in CONFIG["variants"]:
        contract=json.loads((out/variant/"channels_and_train_preprocessing.json").read_text(encoding="utf-8"))
        scores=[]
        for seed in CONFIG["seeds"]:
            d=out/variant/f"seed_{seed}"
            result=json.loads((d/"metrics.json").read_text(encoding="utf-8"))
            pred=predictions(d/"validation_predictions.csv")
            if (set(pred)!=expected or result["test_npz_opened"]!=0 or
                result["C"]!=contract["C"] or result["checkpoint_sha256"]!=sha(d/"best_validation_loss.pt")):
                raise RuntimeError("Artefactos Validation incompatibles")
            y=np.asarray([int(pred[vid]["label"]) for vid in sorted(pred)])
            prob=np.asarray([float(pred[vid]["probability_valid"]) for vid in sorted(pred)])
            calc=metrics(y,prob)
            for key in ("accuracy","precision","recall","f1","macro_f1","balanced_accuracy","roc_auc","pr_auc","TN","FP","FN","TP"):
                if not np.isclose(calc[key],result["validation_metrics"][key],rtol=0,atol=1e-9):
                    raise RuntimeError("Métricas/predicciones difieren: "+key)
            if not np.array_equal(np.load(d/"validation_confusion.npy",allow_pickle=False),calc["confusion_matrix_0_1"]):
                raise RuntimeError("Matriz de confusión inconsistente")
            save_csv(d/"validation_quality_subgroups.csv",group_analysis(pred,quality))
            save_json(d/"validation_fp_ids.json",[vid for vid,r in sorted(pred.items()) if int(r["false_positive"])])
            scores.append(result);all_predictions[(variant,seed)]=pred
        winner=min(scores,key=lambda x:(x["validation_loss"],x["seed"]))
        selected=json.loads((out/variant/"selected_checkpoint.json").read_text(encoding="utf-8"))
        if selected["seed"]!=winner["seed"]:raise RuntimeError("Selección de seed no coincide con Validation loss")
        models[variant]={"scores":scores,"selected":winner,"contract":contract}
    pairs=[]
    for seed in CONFIG["seeds"]:
        rows,summary=pair_prediction(all_predictions[("full",seed)],all_predictions[("sin_relacion",seed)],quality,view,f"matched_seed_{seed}")
        save_csv(out/f"validation_paired_seed_{seed}.csv",rows);pairs.append(summary)
    fa=models["full"]["selected"]["seed"];sa=models["sin_relacion"]["selected"]["seed"]
    rows,selected_pair=pair_prediction(all_predictions[("full",fa)],all_predictions[("sin_relacion",sa)],quality,view,"selected_checkpoints")
    save_csv(out/"validation_paired_selected.csv",rows)
    save_json(out/"paired_comparison_validation.json",{"matched_seeds":pairs,"selected":selected_pair})
    common={"view":view,"train":len(groups["train"]),"validation":len(groups["validation"]),
            "test_npz_opened":0,"variants":{v:{"C":m["contract"]["C"],"scores":m["scores"],"selected_seed":m["selected"]["seed"]}
                                 for v,m in models.items()},"matched_seed_pairs":pairs,"selected_pair":selected_pair,
            "source_sha256":manifest["source_sha256"]}
    save_json(out/"comparison_validation.json",common)
    fp_rows=[]
    table=[]
    for variant in CONFIG["variants"]:
        for score in models[variant]["scores"]:
            m=score["validation_metrics"]
            fp=sorted(vid for vid,r in all_predictions[(variant,score["seed"])].items() if int(r["false_positive"]))
            fp_rows.append((variant,score["seed"],fp))
            table.append(f'| {variant} | {score["seed"]} | {score["C"]} | {score["parameters"]:,} | {score["best_epoch"]} | '
                         f'{score["train_loss"]:.4f} / {score["validation_loss"]:.4f} | '
                         f'{m["accuracy"]:.4f} | {m["precision"]:.4f} | {m["recall"]:.4f} | '
                         f'{m["f1"]:.4f} | {m["macro_f1"]:.4f} | {m["balanced_accuracy"]:.4f} | '
                         f'{m["roc_auc"]:.4f} | {m["pr_auc"]:.4f} | {m["TN"]}/{m["FP"]}/{m["FN"]}/{m["TP"]} |')
    lossdelta=[(seed,models["full"]["scores"][i]["validation_loss"]-models["sin_relacion"]["scores"][i]["validation_loss"],
                models["full"]["scores"][i]["validation_metrics"]["f1"]-models["sin_relacion"]["scores"][i]["validation_metrics"]["f1"])
               for i,seed in enumerate(CONFIG["seeds"])]
    better="mejora" if all(x[1]<0 for x in lossdelta) else ("deterioro" if all(x[1]>0 for x in lossdelta) else "mixta")
    attention_quality=[]
    for variant in CONFIG["variants"]:
        seed=models[variant]["selected"]["seed"]
        rows=list(csv.DictReader(open(out/variant/f"seed_{seed}"/"validation_quality_subgroups.csv",encoding="utf-8")))
        attention_quality.extend(f'| {variant} seed {seed} | {r["subset"]} | {r["n"]} | {r["errors"]} | {r["FP"]} | {r["FN"]} |'
                                 for r in rows)
    overfit='\n'.join(f'- {v} seed {r["seed"]}: train F1 {r["train_metrics"]["f1"]:.4f} vs Validation F1 {r["validation_metrics"]["f1"]:.4f}; '
                      f'train loss {r["train_loss"]:.4f} vs Validation loss {r["validation_loss"]:.4f} '
                      f'(brecha {r["validation_minus_train_loss"]:+.4f}).'
                      for v in CONFIG["variants"] for r in models[v]["scores"])
    fp_text='\n'.join(f'- {variant} seed {seed}: **{len(fp)} FP**, IDs {fp}.' for variant,seed,fp in fp_rows)
    selected=min((m["selected"] for m in models.values()),key=lambda x:x["validation_loss"])
    report=f"""# Entrenamiento BP_{view.upper()} SegCNN V1 — Validation exclusivamente

**{view.upper()}_SEGCNN_TRAINED_VALIDATION_COMPLETE.** Ocho ejecuciones totales en el proyecto (cuatro en esta vista); ningún NPZ Test abierto ni evaluado. Clase positiva **1=válido**; FP=levantamiento inválido predicho válido. Threshold fijo 0,5.

## Cohorte y trazabilidad

- Train **{len(groups['train'])}**; Validation **{len(groups['validation'])}**; split Test registrado únicamente por CSV de cohortes, sin abrir sus secuencias. `source_manifest.json` guarda SHA-256 de split/labels/cohorte/esquema/config y hashes **solo de secuencias Train/Validation**, así como versión de arquitectura. Identidad física entre vistas no verificada; no se usó ninguna secuencia {('right' if view=='left' else 'left').upper()} en esta vista.
- `1=válido, 0=inválido`: Train {dict(Counter(manifest['labels_train_validation'][v] for v in manifest['train_ids']))}, Validation {dict(Counter(manifest['labels_train_validation'][v] for v in manifest['validation_ids']))}. Sin split fuera de Train/Validation. Histórico PoseCNN: {manifest['pose_compatibility']}.
- El Test lateral ya se evaluó históricamente para PoseCNN, por lo que no es inéditamente ciego para *todo* el proyecto. Sus resultados no se consultaron para entrenar ni seleccionar SegCNN. Si se comparará/fusionará con PoseCNN, RIGHT requiere un nuevo baseline PoseCNN en cohorte común sin `bp_354`; los splits restantes coinciden documentalmente. LEFT comparte las particiones actuales de la PoseCNN histórica.

## Protocolo congelado

- Grid idéntico a BP lateral PoseCNN: `linspace(0,T_original-1,170)`; QUALITY/conteos se toman del frame próximo. Variables continuas se interpolan solo entre frames contiguos observados con misma procedencia. BAR requiere asociación local corroborada; selección no verificada usa frame próximo sin crear velocidad. Relación mantiene estado directo/proyección ordinaria/lejana y no cruza cambios; ausencia/ambigüedad deja NaN y máscara 0. No hay tracking persistente.
- Imputación mediana y media/desviación de **Train observado remuestreado**; mismo transform para Validation. `channels_and_train_preprocessing.json` conserva índices, nombres, constantes detectadas únicamente en Train y estadísticas por canal. Cada valor retiene máscara y codificación categórica de procedencia; se descartan únicamente canales constantes en Train. Son señales parcialmente pose-assisted, incluso la variante sin relación.
- Ablación definida antes de entrenar: elimina **{len(RELATION_EXCLUDED)}** features: {', '.join(sorted(RELATION_EXCLUDED))}. Retira perfil de torso y extremo proximal porque dependen de la región de torso; conserva BAR, PERSON y BENCH generales, muñeca–BAR y dinámica general. Retira también sus máscaras/procedencia al generar entradas. FULL C={models['full']['contract']['C']} ({models['full']['selected']['parameters']:,} parámetros); sin relación C={models['sin_relacion']['contract']['C']} ({models['sin_relacion']['selected']['parameters']:,} parámetros). Arquitectura FRONT reproducida: Conv1D 5 (padding 2) ×32/64, MaxPool2, Dropout .20, Conv1D 3 (padding 1) ×128, GAP, MLP 128→64→1 Dropout .40, logits y BCEWithLogitsLoss.
- Dos inicializaciones independientes: seeds 42/1337; AdamW 1e-3, decay 1e-4, batch 32, ReduceLROnPlateau patience 5/factor .5, early stopping patience 15, hasta 200 epochs. Mejor época y semilla por **menor loss Validation**, sin optimizar threshold.

## Resultados por semilla (una predicción por levantamiento)

| Variante | Seed | C | Parámetros | Época | Loss Train / Val | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

Especificidad de inválidos = TN/(TN+FP); sensibilidad de válidos = TP/(TP+FN), en `metrics.json` por seed. FP por variante:
{fp_text}

Checkpoints elegidos por variante: FULL **seed {fa}** (`{models['full']['selected']['checkpoint']}`, SHA {models['full']['selected']['checkpoint_sha256']}); sin relación **seed {sa}** (`{models['sin_relacion']['selected']['checkpoint']}`, SHA {models['sin_relacion']['selected']['checkpoint_sha256']}). Candidato preliminar mínimo loss Validation en esta vista: **{selected['variant']} seed {selected['seed']}**, loss {selected['validation_loss']:.4f}, F1 {selected['validation_metrics']['f1']:.4f}. Es preliminar, no resultado Test.

## Comparación pareada y proyecciones

FULL menos sin relación en la misma semilla (loss menor favorece FULL): {[(seed,round(dl,4),round(df,4)) for seed,dl,df in lossdelta]} como `(seed, Δloss, ΔF1)`. Patrón de loss FULL: **{better}** frente a ablación en las dos semillas; tamaño de entrada y optimización también cambian, por lo que no se atribuye causalidad a las proyecciones. Pareo de checkpoints seleccionados (pueden ser seeds diferentes): **{selected_pair['both_correct']}** ambos correctos, **{selected_pair['both_wrong']}** ambos erróneos, **{selected_pair['only_full_wrong']}** solo FULL errado, **{selected_pair['only_sin_relacion_wrong']}** solo sin relación errado, **{selected_pair['disagreements']}** predicciones discordantes. `validation_paired_seed_42.csv`, `validation_paired_seed_1337.csv`, `validation_paired_selected.csv` y `paired_comparison_validation.json` contienen los casos y probabilidades.

La mayoría de las separaciones originales son proyecciones 2D, no observaciones físicas de contacto. Los subgrupos siguientes son **descriptivos en Validation** (umbrales fijos antes del análisis: ≥50% BAR corroborada, ≥80% proyecciones, <20% velocidad); no modifican ningún modelo:

| Variante | Subgrupo | N videos | Errores | FP | FN |
|---|---|---:|---:|---:|---:|
{chr(10).join(attention_quality)}

`validation_source_quality.csv` conserva tasas por video; por seed `validation_quality_subgroups.csv` y `validation_fp_ids.json`. Un N bajo o vacío no autoriza conclusiones sobre ese estrato.

## Generalización y siguiente paso

{overfit}

Las curvas por seed, matrices de confusión y probabilidades están en `full/seed_*/` y `sin_relacion/seed_*/`. Para una futura fusión, alinear los videos de la cohorte monovista, usar predicciones fuera de muestra en desarrollo y no usar Test para calibrarla. En RIGHT, reentrenar PoseCNN en la cohorte común sin duplicado antes de afirmar comparabilidad/fusión; no comparar métricas históricas de particiones distintas como equivalentes. No se entrenaron PoseCNN ni fusiones aquí.
"""
    (out/f"Reporte_Entrenamiento_BP_{view.upper()}_SegCNN_V1.md").write_text(report,encoding="utf-8")
    return common


def main():
    left=report_view("left");right=report_view("right")
    lines=[]
    for s in (left,right):
        for variant,r in s["variants"].items():
            best=min(r["scores"],key=lambda m:m["validation_loss"])
            m=best["validation_metrics"]
            lines.append(f'| {s["view"].upper()} | {variant} | {best["seed"]} | {r["C"]} | {best["validation_loss"]:.4f} | '
                         f'{m["accuracy"]:.4f} | {m["f1"]:.4f} | {m["macro_f1"]:.4f} | '
                         f'{m["roc_auc"]:.4f} | {m["pr_auc"]:.4f} | {m["FP"]} |')
    text=f"""# Comparativo documental SegCNN BP_LEFT / BP_RIGHT V1

**Dos sistemas monovista independientes.** Este cuadro no define ni evalúa una CNN SIDE. Únicamente Validation: Train LEFT/RIGHT {left['train']}/{right['train']}; Validation {left['validation']}/{right['validation']}. Ningún NPZ Test se abrió ni se evaluó.

| Vista | Variante | Seed mínimo loss | C | Val loss | Acc | F1 | Macro-F1 | ROC-AUC | AP | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

LEFT: FULL tiene menor Validation loss en las dos semillas; RIGHT: la ablación sin relación la tiene en las dos. Se trata de una **asociación descriptiva** en Validation; cambios de dimensionalidad, inicialización y optimización impiden atribuir causalidad a las proyecciones. FULL incorpora relaciones mayoritariamente inferidas por extrapolación 2D y sigue siendo pose-assisted. RIGHT excluye `bp_354` de la cohorte común como duplicado exacto Train de `bp_027`; su PoseCNN histórica requiere reentrenamiento sobre la cohorte correspondiente antes de comparaciones/fusión estrictas. Identidad física entre vistas no verificada. Los Test laterales ya fueron observados históricamente por PoseCNN, pero esta SegCNN no los ha usado.

Los detalles por seed, FP, matrices, casos pareados, análisis descriptivo por calidad y brechas Train–Validation están en los reportes monovista. La imputación y escalado se ajustaron solo a Train de cada vista y variante. Ningún modelo PoseCNN ni fusión se entrenó en esta fase.

**LEFT_SEGCNN_TRAINED_VALIDATION_COMPLETE**; **RIGHT_SEGCNN_TRAINED_VALIDATION_COMPLETE**.
"""
    (ROOT/"training/Reporte_Comparativo_SegCNN_BP_LEFT_RIGHT_V1.md").write_text(text,encoding="utf-8")
    print("Validation reportes",left["view"],right["view"],flush=True)


if __name__=="__main__":main()
