"""Reportes Validation de fusiones BP_LEFT y BP_RIGHT, sin red SIDE."""
import json

from training.bp.experiments.bp_posecnn_common_v1.core import ROOT, save_json
from training.bp.experiments.bp_side_segcnn_v1.data import sha
from .data import fusion_out, snapshot


def row(name,score):
    return (f'| {name} | {score["accuracy"]:.4f} | {score["precision"]:.4f} | '
            f'{score["recall"]:.4f} | {score["f1"]:.4f} | {score["macro_f1"]:.4f} | '
            f'{score["balanced_accuracy"]:.4f} | {score["roc_auc"]:.4f} | '
            f'{score["pr_auc"]:.4f} | {score["sensitivity_valid"]:.4f} | '
            f'{score["specificity_invalid"]:.4f} | {score["FPR"]:.4f} | '
            f'{score["TN"]}/{score["FP"]}/{score["FN"]}/{score["TP"]} |')


def view_report(view):
    out=fusion_out(view)
    summary=json.loads((out/"fusion_validation_summary.json").read_text(encoding="utf-8"))
    source=json.loads((out/"fusion_source_manifest.json").read_text(encoding="utf-8"))
    config=json.loads((out/"fusion_config.json").read_text(encoding="utf-8"))
    integrity=json.loads((out/"fusion_integrity_verification.json").read_text(encoding="utf-8"))
    paired=json.loads((out/"fusion_pairwise_analysis.json").read_text(encoding="utf-8"))
    selected=json.loads((out/"fusion_selected_candidate.json").read_text(encoding="utf-8"))
    if (source["source_sha256"]!=snapshot(view) or integrity["status"]!="INTEGRITY_OK" or
        integrity["test_npy_opened"]!=0 or selected["test_npy_opened"]!=0):
        raise RuntimeError("Fuentes o integridad no conformes")
    if len(integrity["all_eight_head_replay"])!=4 or len(integrity["parts"])!=4:
        raise RuntimeError("Falta replay de una combinación")
    base=["pose","seg_full","seg_sin_relacion","late_full","late_sin_relacion",
          "embedding_full","embedding_sin_relacion"]
    scores=summary["metrics"]
    table="\n".join(row(name,scores[name]) for name in base)
    maxacc=max(s["accuracy"] for s in scores.values());maxf1=max(s["f1"] for s in scores.values())
    maxmacro=max(s["macro_f1"] for s in scores.values());minfp=min(s["FP"] for s in scores.values())
    extrema={"max_accuracy":[name for name,s in scores.items() if abs(s["accuracy"]-maxacc)<1e-12],
             "max_f1":[name for name,s in scores.items() if abs(s["f1"]-maxf1)<1e-12],
             "max_macro_f1":[name for name,s in scores.items() if abs(s["macro_f1"]-maxmacro)<1e-12],
             "min_fp":[name for name,s in scores.items() if s["FP"]==minfp]}
    save_json(out/"fusion_extrema_validation.json",extrema)
    seeds=[]
    for variant in ("full","sin_relacion"):
        for result in summary["variants"][variant]["seed_metrics"]:
            m=result["validation_metrics"]
            seeds.append(f'| {variant} | {result["seed"]} | {result["best_epoch"]} / {result["epochs_run"]} | '
                         f'{result["train_loss"]:.4f} / {result["validation_loss"]:.4f} | '
                         f'{result["train_metrics"]["f1"]:.4f} / {m["f1"]:.4f} | '
                         f'{m["macro_f1"]:.4f} | {m["FP"]} |')
    errors=[]
    for pair in paired["comparisons"]:
        e=pair["fusion_effect"]
        errors.append(f'| {pair["kind"]} {pair["variant"]} | {pair["shared_count"]} | '
                      f'{len(e["corrected_shared"])} / {len(e["retained_shared"])} | '
                      f'{len(e["corrected_pose_only"])} / {len(e["corrected_seg_only"])} | '
                      f'{len(e["new_error_both_correct"])} | {pair["false_positive_count"]} |')
    fps=[]
    paired_rows={r["video_id"]:r for r in _csv(out/"fusion_validation_predictions.csv")}
    for name in base:
        videos=[vid for vid,r in sorted(paired_rows.items()) if int(r["label"])==0 and int(r[name+"_pred"])==1]
        fps.append(f'- {name}: {videos} ({len(videos)} FP).')
    replay=max(p["maximum_absolute_probability_error"] for p in integrity["parts"])
    winner=selected["winner"]
    champion=selected["frozen_checkpoint"]
    report=f"""# BP_{view.upper()} — Fusión PoseCNN + SegCNN temporal V1 (solo Validation)

**{view.upper()}_FUSION_VALIDATION_COMPLETE.** Cohorte {source['cohort']['train']} Train y {source['cohort']['validation']} Validation; Test se consultó **solo como cantidad de IDs** ({source['cohort']['test_ids_only']}). No se abrió su NPY/NPZ, no se entrenó ni evaluó en Test, no hay clasificador SIDE.

## Integridad previa y contrato congelado

- PoseCNN + ambas SegCNN reprodujeron **exactamente** las probabilidades base de Validation (`base_reproducibility_full.json`, `base_reproducibility_sin_relacion.json`: max diff 0). Sus checkpoints, scalers y contratos, junto con split, labels y cohortes, conservan SHA-256 en `fusion_source_manifest.json`; los hashes antes/después son idénticos (`fusion_integrity_verification.json`). IDs y etiquetas coinciden estrictamente dentro de esta vista. Ninguna muestra de la otra vista se incorporó.
- `PoseEmbeddingAdapter.extract_embeddings`: `(N,64)` desde entrada `(N,3,170,8)`. `SegEmbeddingAdapter.extract_embeddings`: `(N,128)` tras AdaptiveAvgPool1d desde `(N,C,170)`; los forwards de ambos originales se reconstruyeron sin alterar probabilidad/logits. Concatenación `(N,192)`; cabeza Linear(192,32)→ReLU→Dropout(.40)→Linear(32,1): **6209 parámetros entrenables**. Pose y Seg: **0 entrenables**, `eval()` permanente para BatchNorm. Embeddings por vista normalizados con media/desviación fit **solo Train**; `embedding_fusion_*/embedding_scaler_train.npz` y `embeddings_development.npz` no contienen Test.
- Backbones conservados; selección de SegCNN FULL y SIN relación por menor Validation loss previa; Pose {('histórica compatible seed42' if view=='left' else 'reentrenada en 431 Train seed1337')}. La cabeza se entrenó con embeddings **in-sample** de Train (no OOF): AdamW lr=.001, weight_decay=.001, batch32, 100 epochs máx, scheduler plateau patience5/factor.5 y early stopping patience15. Dos seeds 42/1337 por variante; mejor checkpoint/seed por menor Validation loss, umbral fijo .5. No fine-tuning de backbones.
- Fusión tardía: `p=.5*p_pose+.5*p_seg` en cada video, sin parámetros ni ajuste de pesos; `late_fusion_*/validation_predictions.csv`. Validez positiva 1. FPR=FP/(TN+FP). Fuente SegCNN parcialmente asistida por Pose; las separaciones barra–torso son mayormente proyecciones 2D, no contacto físico confirmado.

## Siete configuraciones por video (Validation {summary['validation_n']})

| Configuración | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Sens válido | Spec inválido | FPR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table}

Mayor Accuracy: **{extrema['max_accuracy']}**. Mayor F1: **{extrema['max_f1']}**. Mayor Macro-F1: **{extrema['max_macro_f1']}**. Menos falsos positivos: **{extrema['min_fp']}**. La clase positiva es válido, por lo que FP significa un intento inválido aceptado. Listas completas:
{chr(10).join(fps)}

## Ocho ejecuciones de cabeza entre ambas vistas (cuatro de esta vista)

| Seg variante | Seed cabeza | Mejor época / épocas corridas | Loss Train / Val | F1 Train / Val | Macro-F1 Val | FP Val |
|---|---:|---|---|---|---:|---:|
{chr(10).join(seeds)}

Mejores seeds por menor loss Validation: FULL {summary['variants']['full']['selected_embedding_seed']}, SIN relación {summary['variants']['sin_relacion']['selected_embedding_seed']}. Los checkpoints de ambas seeds, métricas, curvas, matrices, predicciones y scaler Train están en `embedding_fusion_*/`. La brecha entre pérdidas Train–Validation y la elección de épocas tempranas señalan **sobreajuste potencial de la cabeza**; ninguna búsqueda adicional de hiperparámetros se efectuó. Las representaciones de Train son in-sample respecto de ambos backbones.

## Corrección/introducción de errores (comparación pareada)

| Fusión | Errores base compartidos | Compartidos corregidos / retenidos | Exclusivos Pose / Seg corregidos | Errores nuevos (ambos base correctos) | FP fusión |
|---|---:|---|---|---:|---:|
{chr(10).join(errors)}

Los IDs se preservan en `fusion_validation_predictions.csv` y `fusion_pairwise_analysis.json`. No se presenta un oráculo de elección entre modelos como rendimiento de fusión ni se crean reglas ad hoc. La reproducción posterior desde NPY Pose + NPZ Seg sobre Validation confirmó **todas las decisiones iguales**: máximo error absoluto de probabilidad {replay:.2e} por variación numérica de lote; las ocho cabezas también reproducen sus probabilidades con su propia matriz de embeddings. Ningún Test abierto.

## Candidato preliminar y limitaciones

- Regla congelada **antes** del entrenamiento: mayor Macro-F1 Validation, luego especificidad de inválidos, luego Balanced Accuracy; empates exactos favorecen menor complejidad. Las cabezas eligen primero seed por menor Validation loss. `fusion_config.json` documenta pesos tardíos, umbral y criterio. **Preselección BP_{view.upper()}: {winner}**, checkpoint `{champion['path']}`, SHA `{champion['sha256']}`; `fusion_selected_candidate.json` incluye ranking, hashes de backbones y scaler si aplica. Esta elección se hizo **antes de abrir Test**.
- Validation ya se utilizó antes para checkpoints y variantes, y aquí se reutiliza: **no es un estimador externo independiente**. La PoseCNN lateral histórica ya expuso Test en un proyecto anterior, aunque aquí no se abrió. No está comprobada la independencia por levantamiento físico, atleta o sesión entre LEFT/RIGHT. La SegCNN lateral deriva parte de sus features desde pose, por lo que complementariedad no equivale a modalidades estadísticamente independientes. No interpretar extrapolaciones 2D como contacto físico.
"""
    (out/f"Reporte_Entrenamiento_BP_{view.upper()}_Fusion_V1.md").write_text(report,encoding="utf-8")
    return summary,selected,extrema


def _csv(path):
    from training.bp.experiments.bp_posecnn_common_v1.core import rows_csv
    return rows_csv(path)


def main():
    left,lselect,lex=view_report("left")
    right,rselect,rex=view_report("right")
    winners=[]
    for name,result in (("LEFT",left),("RIGHT",right)):
        for key in ("pose","seg_full","seg_sin_relacion","late_full","late_sin_relacion","embedding_full","embedding_sin_relacion"):
            s=result["metrics"][key]
            winners.append(f'| {name} | {key} | {s["f1"]:.4f} | {s["macro_f1"]:.4f} | {s["specificity_invalid"]:.4f} | {s["FP"]} | {s["FPR"]:.4f} |')
    text=f"""# Comparativo documental de fusiones BP_LEFT y BP_RIGHT V1

**Dos cohortes, dos experimentos y dos selecciones independientes.** No existe CNN SIDE. Train/Validation LEFT {left['validation_n']} Validation, RIGHT {right['validation_n']} Validation; los Test no se abrieron ni evaluaron. Criterio prefijado: Macro-F1 → especificidad de inválidos → Balanced Accuracy, umbral .5 y pesos tardíos .5/.5 sin ajuste.

| Vista | Candidato | F1 | Macro-F1 | Especificidad inválidos | FP | FPR |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(winners)}

Preselección LEFT: **{lselect['winner']}** (FP={lselect['winner_validation']['FP']}, FPR={lselect['winner_validation']['FPR']:.4f}), con checkpoint/hash en `training/bp/experiments/bp_left_03_fusion_v1/fusion_selected_candidate.json`. Preselección RIGHT: **{rselect['winner']}** (FP={rselect['winner_validation']['FP']}, FPR={rselect['winner_validation']['FPR']:.4f}), checkpoint/hash en `training/bp/experiments/bp_right_03_fusion_v1/fusion_selected_candidate.json`. RIGHT no obliga a elegir una fusión cuando la SegCNN individual supera a las cabezas. Se mantienen los candidatos secundarios congelados.

Los backbones se cargaron congelados, con BatchNorm en eval; la cabeza sola tiene 6.209 parámetros sobre embeddings 64+128. Su Train fue **in-sample** respecto de los encoders, no stacking OOF; se vigilaron brechas Train–Validation y épocas tempranas. BP_LEFT y BP_RIGHT son parcialmente pose-assisted en la rama Seg, no modalidades estadísticamente independientes. Validation ha sido reutilizado para selección de base/variante/cabeza y no es test externo; los Test laterales PoseCNN ya tuvieron exposición histórica. La identidad física entre vistas tampoco está acreditada. No se ejecutó Test, fusionamos únicamente scores/embeddings de videos de la misma vista y no se ajustó umbral ni pesos por errores individuales.

**LEFT_FUSION_VALIDATION_COMPLETE** y **RIGHT_FUSION_VALIDATION_COMPLETE**. Detalle de IDs, FP, especificidades, pareos y SHA-256 en reportes monovista.
"""
    (ROOT/"training/docs/bp/Reporte_Comparativo_Fusiones_BP_LEFT_RIGHT_V1.md").write_text(text,encoding="utf-8")
    print("Reportes fusión LEFT/RIGHT completados",flush=True)


if __name__=="__main__":main()
