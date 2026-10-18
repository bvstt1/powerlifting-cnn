"""Reporte retrospectivo 5-fold, excluye Test y no cambia preselecciones finales."""
import json

from training.bp_posecnn_common_v1.core import ROOT, rows_csv
from .protocol import VARIANT, output, snapshot


def detail(view):
    out=output(view)
    report=json.loads((out/"cv_integrity.json").read_text(encoding="utf-8"))
    source=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    config=json.loads((out/"cv_config.json").read_text(encoding="utf-8"))
    pairs=json.loads((out/"oof_paired_comparisons.json").read_text(encoding="utf-8"))["pairs"]
    folds=rows_csv(out/"fold_metrics.csv")
    pooled={row["model"]:row for row in rows_csv(out/"oof_pooled_metrics.csv")}
    stability={row["model"]:row for row in rows_csv(out/"five_fold_stability.csv")}
    if (not report["source_hashes_unchanged"] or source["source_sha256"]!=snapshot(view) or
        report["test_npy_opened"]!=0 or len(folds)!=15):
        raise RuntimeError("Reporte sin integridad confirmada")
    models=(("pose","seg_full","embedding_full") if view=="left" else
            ("pose","seg_sin_relacion","late_sin_relacion"))
    pooled_table="\n".join(f'| {name} | {float(pooled[name]["accuracy"]):.4f} | '
                f'{float(pooled[name]["f1"]):.4f} | {float(pooled[name]["macro_f1"]):.4f} | '
                f'{float(pooled[name]["balanced_accuracy"]):.4f} | {float(pooled[name]["roc_auc"]):.4f} | '
                f'{float(pooled[name]["pr_auc"]):.4f} | {float(pooled[name]["specificity_invalid"]):.4f} | '
                f'{float(pooled[name]["FPR"]):.4f} | {float(pooled[name]["FNR"]):.4f} | '
                f'{pooled[name]["TN"]}/{pooled[name]["FP"]}/{pooled[name]["FN"]}/{pooled[name]["TP"]} |'
                for name in models)
    variation="\n".join(f'| {name} | {float(stability[name]["accuracy_mean"]):.4f}±{float(stability[name]["accuracy_std"]):.4f} | '
               f'{float(stability[name]["f1_mean"]):.4f}±{float(stability[name]["f1_std"]):.4f} | '
               f'{float(stability[name]["macro_f1_mean"]):.4f}±{float(stability[name]["macro_f1_std"]):.4f} | '
               f'{float(stability[name]["balanced_accuracy_mean"]):.4f}±{float(stability[name]["balanced_accuracy_std"]):.4f} | '
               f'{float(stability[name]["roc_auc_mean"]):.4f}±{float(stability[name]["roc_auc_std"]):.4f} | '
               f'{float(stability[name]["specificity_invalid_mean"]):.4f}±{float(stability[name]["specificity_invalid_std"]):.4f} | '
               f'{stability[name]["false_positives_by_fold"]} |'
               for name in models)
    fold_rows="\n".join(f'| {r["fold"]} | {r["model"]} | {r["n"]} | '
                        f'{float(r["accuracy"]):.4f} | {float(r["f1"]):.4f} | '
                        f'{float(r["macro_f1"]):.4f} | {float(r["roc_auc"]):.4f} | '
                        f'{float(r["pr_auc"]):.4f} | {float(r["specificity_invalid"]):.4f} | '
                        f'{float(r["FPR"]):.4f} | {float(r["FNR"]):.4f} | '
                        f'{r["TN"]}/{r["FP"]}/{r["FN"]}/{r["TP"]} | '
                        f'{r["best_pose_epoch"]}/{r["best_seg_epoch"]}/{r["best_head_epoch"]} |'
                        for r in folds)
    paired_rows="\n".join(f'- {r["a"]} vs {r["b"]}: compartidos **{r["shared_n"]}**, '
                           f'exclusivos {r["a"]} **{r["only_a_n"]}** y {r["b"]} **{r["only_b_n"]}**, '
                           f'desacuerdos **{r["disagreements_n"]}**. Por fold: '
                           f'{[(t["fold"],t["shared_errors"],t["only_a"],t["only_b"]) for t in r["per_fold"]]}. '
                           f'FP {r["a"]}: {len(r["fp_a"])}; FP {r["b"]}: {len(r["fp_b"])}.'
                           for r in pairs)
    preselected="Fusión embeddings Pose+Seg FULL seed1337" if view=="left" else "SegCNN SIN relación seed1337"
    chosen=source["source_sha256"]["final_fusion_head" if view=="left" else "final_seg_checkpoint"]
    duplicate_note=("RIGHT excluye `bp_354` (MP4 duplicado de `bp_027`) antes de crear folds. "
                    if view=="right" else "Los duplicados conocidos LEFT quedan fuera de development. ")
    comparator_note=("Para RIGHT el promedio tardío 50/50 es comparador **secundario**, sin cabeza, pesos ni umbral ajustados; incluso si varía favorablemente entre folds, el candidato principal sigue siendo SegCNN SIN. "
                     if view=="right" else "Para LEFT la cabeza usa seed fijo 1337 y no se utilizaron pesos finales originales en ningún fold. ")
    report_text=f"""# Validación cruzada estratificada BP_{view.upper()} — cinco folds sobre development

**CV_RETROSPECTIVE_VALIDATION_COMPLETE_{view.upper()}**. Fuente: Train+Validation históricos = **{report['development_n']} videos**; Test reservado ({source['original_cohort']['test_ids_only']} IDs) **no se abrió ni evaluó**. Variante SegCNN fija: `{VARIANT[view]}`; input C={138 if view=='left' else 90}, según contrato histórico. Preselección previa para Test: **{preselected}**, checkpoint protegido SHA-256 `{chosen}`. Esta CV NO cambió la preselección.

## Identidad, folds y ausencia de leakage

- Desarrollo procedente exclusivamente de {source['original_cohort']['train']} Train + {source['original_cohort']['validation']} Validation originales; {duplicate_note}La auditoría de hashes MP4 en development no encontró ningún par idéntico restante. No existen grupos físicos acreditados en BP lateral: se utilizó **StratifiedKFold(n_splits=5, shuffle=True, random_state=42) por video**, no mapeos por numeración. No puede afirmarse independencia de atleta, sesión o levantamiento físico; `fold_assignments.csv` y `fold_class_distributions.csv` conservan IDs/etiquetas y roles.
- Outer evaluation exclusivamente para el cálculo externo. En cada Outer Train se separó inner Validation ≈15 % con estratificación y seed 42+fold, común a PoseCNN/SegCNN/cabeza. Los checkpoints y scheduler se seleccionaron **solo con inner Validation**; no se seleccionaron seeds según Outer. Scaler Pose/Seg/embeddings fit **solo inner Train original**. Pose Train aumentado ×4 exclusivamente allí, sin flip/rotación. Todos los pesos, BatchNorm y estados de optimizer se inicializaron desde cero por fold. Los backbones de fusión se congelaron en `eval()`; embeddings inner Train son **in-sample**, embeddings Outer externos.
- Canales SegCNN idénticos al input congelado histórico (C={138 if view=='left' else 90}); topología mantenida si alguna feature resulta constante/vacía en un inner Train, con política fija mediana=0 si vacía y std=1 si no hay dispersión. Ningún estadístico del Outer (ni de la unión de development) se ajustó. Insumos originales temporalmente remuestreados a 170 con las mismas reglas de ausencia/procedencia ya congeladas.
- `source_manifest.json` y `cv_integrity.json` contienen hashes de sources/checkpoints finales antes/después: intactos. Hashes individuales Train/Validation de NPY Pose y NPZ Seg, sin apertura de archivos Test. Archivos CV por fold (scalers, parámetros, curvas, checkpoints y probabilidades) están bajo `fold_01/` … `fold_05/`.

## Métricas por fold externo (una predicción por levantamiento)

| Fold | Modelo | N | Acc | F1 | Macro-F1 | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP | Mejor época Pose/Seg/Cabeza |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{fold_rows}

Los CSV por fold incluyen Precision, Recall, Balanced Accuracy y sensibilidad de válidos; FPR=FP/(FP+TN), FNR=FN/(FN+TP), positivo=1 válido, FP=inválido aprobado. Todos los folds tuvieron ambas clases; si faltara alguna, ROC-AUC/AP se declararía N/D. Las matrices de confusión están también guardadas como `fold_XX/confusion_<model>.npy`.

## Media ± desviación poblacional de cinco folds (NO pooled)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | Especificidad inválidos | FP por fold |
|---|---:|---:|---:|---:|---:|---:|---|
{variation}

`five_fold_stability.csv` también ofrece AP y FPR media±desviación. Estas medias de folds **no** se confunden con las métricas pooled siguientes.

## Pooled OOF (únicamente Outer Validation de cada fold, un resultado por video)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{pooled_table}

`oof_predictions.csv` y `oof_paired_by_video.csv` permiten reproducir estas métricas sin mezclar encoders de distintos folds. Los F1 y AUC pooled no son la media de métricas de fold. Pooled FP = {[ (name,report['pooled_oof'][name]['FP']) for name in models ]}.

## Errores pareados en OOF

{paired_rows}

Los IDs compartidos y exclusivos, FP/FN y discrepancias por fold constan en `oof_paired_comparisons.json`; no se aplicaron reglas por video ni se calculó un oráculo. {comparator_note}

## Interpretación para la tesis

La arquitectura, el esquema de features y los hiperparámetros fueron escogidos antes usando desarrollo histórico. Esta CV es un análisis **retrospectivo condicionado por esas decisiones**, no una selección anidada totalmente independiente. Los modelos Test finales preseleccionados siguen intactos, y Test se reserva para encargo posterior. Además de falta de grupos físicos verificados, la rama SegCNN lateral depende parcialmente de Pose YOLO y de proyecciones 2D barra–torso; el entrenamiento YOLO previo no demuestra separación por atletas/eventos respecto a este conjunto. Estas limitaciones impiden interpretar un promedio CV alto como prueba externa de juicio de competición.
"""
    (out/f"Reporte_CV_BP_{view.upper()}_V1.md").write_text(report_text,encoding="utf-8")
    return report


def main():
    a=detail("left");b=detail("right")
    text=f"""# Resumen documental de CV 5-fold BP_LEFT y BP_RIGHT

Dos sistemas monovista, ningún modelo SIDE. Ningún NPZ/NPY de Test abierto. LEFT development=513 y RIGHT development=524; RIGHT mantiene excluido `bp_354`. Los grupos físicos entre atletas/sesiones/vistas no están acreditados y no se emparejaron IDs por numeración. Los folds y inner Validation fueron estratificados por video, con checks de hashes duplicados de development.

| Vista | Modelo | F1 cinco folds media±std | ROC-AUC media±std | Spec media±std | F1 pooled OOF | Macro-F1 pooled | FP pooled |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(f'| {view.upper()} | {name} | {item["f1_mean"]:.4f}±{item["f1_std"]:.4f} | {item["roc_auc_mean"]:.4f}±{item["roc_auc_std"]:.4f} | {item["specificity_invalid_mean"]:.4f}±{item["specificity_invalid_std"]:.4f} | {source["pooled_oof"][name]["f1"]:.4f} | {source["pooled_oof"][name]["macro_f1"]:.4f} | {source["pooled_oof"][name]["FP"]} |' for view,source in (("left",a),("right",b)) for item in source["fold_stability"] for name in (item["model"],))}

Las estimaciones pooled usan una probabilidad Outer Validation por video y **no son** medias de folds. Esta CV es retrospectiva y condicionada por decisiones de arquitectura tomadas anteriormente con development; no reabre la selección final. Candidatos para el protocolo Test posterior, protegidos por SHA-256: LEFT fusión embeddings FULL seed1337; RIGHT SegCNN SIN relación seed1337. La fusión tardía RIGHT 50/50 permanece comparador secundario. Los detalles de estabilidad, matrices, FP/FN por fold, epochs internos y análisis pareado están en los reportes monovista. Sin evaluación Test.
"""
    (ROOT/"training/Reporte_CV_BP_LEFT_RIGHT_V1.md").write_text(text,encoding="utf-8")
    print("CV reportes LEFT y RIGHT completados",flush=True)


if __name__=="__main__":main()
