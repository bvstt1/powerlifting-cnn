"""Reportes de preparación para fusión; solo predicciones Validation monovista."""
import json

from training.bp.experiments.bp_side_segcnn_v1.data import sha
from .core import ROOT, paths, rows_csv, save_json


def fmt(x):
    return f"{x:.4f}"


def table(name, m):
    return (f'| {name} | {fmt(m["accuracy"])} | {fmt(m["precision"])} | {fmt(m["recall"])} | '
            f'{fmt(m["f1"])} | {fmt(m["macro_f1"])} | {fmt(m["balanced_accuracy"])} | '
            f'{fmt(m["roc_auc"])} | {fmt(m["pr_auc"])} | {m["TN"]}/{m["FP"]}/{m["FN"]}/{m["TP"]} |')


def compare_rows(result):
    return "\n".join(table(k,v) for k,v in result["metrics"].items())


def paired(result):
    lines=[]
    for name,part in result["paired"].items():
        lines.append(f'- {name}: {part["shared_count"]} errores compartidos {part["shared_errors"]}; '
                     f'{part["pose_only_count"]} solo Pose {part["pose_only_errors"]}; '
                     f'{part["seg_only_count"]} solo Seg {part["seg_only_errors"]}; '
                     f'{len(part["prediction_disagreements"])} desacuerdos; '
                     f'correlación Pearson de probabilidades {part["probability_pearson_r"]:.4f}.')
    return "\n".join(lines)


def left_report():
    p=paths("left");out=p["left_out"]
    aud=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
    stats=json.loads((out/"validation_metrics.json").read_text(encoding="utf-8"))
    scal=json.loads((out/"scaler_audit.json").read_text(encoding="utf-8"))
    weights=json.loads((out/"checkpoint_audit.json").read_text(encoding="utf-8"))
    embedding=json.loads((out/"embeddings_audit.json").read_text(encoding="utf-8"))
    comparison=json.loads((out/"validation_pose_seg_comparison.json").read_text(encoding="utf-8"))
    protected=json.loads((out/"protected_sources.json").read_text(encoding="utf-8"))
    if (aud["status"]!="LEFT_POSECNN_HISTORICAL_COMPATIBLE" or aud["issues"] or
        stats["validation_video_order"]!=[r["video_id"] for r in rows_csv(out/"validation_predictions.csv")] or
        not stats["loss_reproduced"] or not scal["fit_original_train_only_verified"] or
        not embedding["exact_probability_match"] or comparison["validation_videos"]!=91 or
        protected["checkpoint_sha256"]!=sha(p["historical_ckpt"])):
        raise RuntimeError("LEFT no está acreditada")
    fp_left=[row["video_id"] for row in rows_csv(out/"validation_predictions.csv") if row["false_positive"]=="1"]
    report=f"""# Compatibilidad BP_LEFT PoseCNN histórica / SegCNN V1

**LEFT_POSECNN_HISTORICAL_COMPATIBLE**. Se conserva `training/models/bp/left/posecnn/runs/baseline_t170_v1/best_model.pth`; no se entrenó PoseCNN LEFT. Test NPY/SegCNN Test no abiertos ni evaluados.

## Cohorte e integridad

- Coincidencia **por video_id, label y split**: Train=422, Validation=91, Test=91 (solo IDs del split), sin asignación=7; `historical_minus_current={{'train': [], 'validation': []}}`. Sin muestras adicionales en pesos según split y scaler (`n_samples_seen_=422`). Los siete no asignados no aparecen en Train/Validation histórico. `train_validation_source_audit.csv` registra hashes SHA-256 de cada NPY abierto, su digest histórico de contenido `(T,8,3)`, escala/centro y el grid; `integrity.json` registra hashes de fuentes protegidas.
- Duplicados conocidos en la auditoría de origen: solo los excluidos dentro/sin split, sin duplicados exactos documentados entre Train y Validation/Test; no se leyó ningún NPY Test para esta verificación. Identidad física entre vistas/atletas/sesiones no verificada. No emparejar LEFT/RIGHT por número.
- Orden de puntos en la arquitectura histórica: rodilla, tobillo, cadera, hombro, muñeca/mano, codo, talón, punta. Canales x,y normalizados más confianza YOLO; no MediaPipe. Faltantes XY se interpolan solo en huecos interiores ≤5 (sin extrapolar extremos), confianza ausente a 0, centro fijo mediano del hombro índice 3, escala mediana de distancia hombro3–cadera2. Resample lineal `linspace(0,T_original-1,170)` y entrada `(N,3,170,8)`. SegCNN usa el **mismo grid**, pero abstiene/mantiene máscaras donde PoseCNN interpola breves huecos: misma posición temporal no significa igual política de faltantes.
- Scaler histórico StandardScaler fit sobre Train original `(422,4080)`, antes de aumento x4; máximo |mean/var/scale reconstruidas − guardadas| = {scal['max_abs_fit_difference']}; hash scaler `{aud['scaler_sha256']}`. Aumentos Train: translación, escala, ruido XY y shift temporal, sin flips/rotación. Validación sin aumento; no se comparte scaler con SegCNN.

## Pesos e inferencia reproducible

- Arquitectura inspeccionada `training/common/train_utils.py:PoseCNN`: Conv2D 3×3 padding same 32/64/128, BatchNorm+ReLU, MaxPool(2,1), Dropout2D .40, GAP 128, Dense(128→64), Dropout .50, Dense(64→1), Sigmoid; BCELoss, Adam lr=.001/decay=1e-4, batch16, ReduceLROnPlateau factor .5/patience4, EarlyStopping patience8. {weights['parameters']:,} parámetros; checkpoint SHA `{aud['checkpoint_sha256']}`; llaves cargadas estrictamente.
- `history.csv`: mejor época **{stats['historical_best_epoch']}**, Validation loss registrada **{stats['historical_validation_loss']:.9f}**; reproducida **{stats['validation_loss']:.9f}**, diferencia **{abs(stats['validation_loss']-stats['historical_validation_loss']):.2e}** con tolerancia predeterminada {stats['numerical_tolerance']}. No existía CSV de probabilidades histórico de este holdout; se reconstruyeron 91 en `validation_predictions.csv` y quedaron ordenadas en `validation_metrics.json`. No se consultaron reportes de Test históricos.
- Embedding externo `training.bp.experiments.bp_posecnn_common_v1.core.PoseEmbeddingAdapter.extract_embeddings(x)`: **{embedding['embedding_dimension']} dimensiones**, entrada `(N,3,170,8)`, salida `(N,64)` tras Linear128→64 y ReLU. Reconstrucción exacta de probabilidad original (`max_abs_diff={embedding['max_abs_probability_difference']}`), sin modificar pesos ni código histórico. Sin embeddings Test exportados.

## Comparación Validation (mismos 91 levantamientos; clase positiva 1=válido)

| Modelo | Accuracy | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{compare_rows(comparison)}

FP=inválido predicho válido. PoseCNN histórica: **{stats['metrics']['FP']} FP**, IDs **{fp_left}**; ver probabilidades en `validation_predictions.csv` y tabla alineada `validation_pose_seg_paired.csv`.

{paired(comparison)}

Las coincidencias y divergencias son descriptivas, no un resultado de fusión. SegCNN FULL seed42 y SIN relación seed42 se conservaron tal cual; solo se leyeron predicciones Validation. La modalidad SegCNN lateral puede depender de pose en la selección BAR: no asumir independencia estadística completa de entradas. Una fusión futura debe validarse por video en desarrollo sin ajustar usando Test.
"""
    (out/"Reporte_Compatibilidad_BP_LEFT_PoseCNN_SegCNN.md").write_text(report,encoding="utf-8")
    return comparison


def right_report():
    p=paths("right");out=p["right_out"]
    audit=json.loads((out/"integrity.json").read_text(encoding="utf-8"))
    history=json.loads((out/"historical_right_audit.json").read_text(encoding="utf-8"))
    source=json.loads((out/"source_manifest.json").read_text(encoding="utf-8"))
    selected=json.loads((out/"selected_checkpoint.json").read_text(encoding="utf-8"))
    embedding=json.loads((out/f'seed_{selected["seed"]}'/"embeddings_audit.json").read_text(encoding="utf-8"))
    comparison=json.loads((out/"validation_pose_seg_comparison.json").read_text(encoding="utf-8"))
    scores=[json.loads((out/f"seed_{seed}"/"metrics.json").read_text(encoding="utf-8")) for seed in (42,1337)]
    if (audit["status"]!="RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE" or
        source["cohort"]!={"train":431,"validation":93,"test_ids_only":93,"unassigned":12} or
        history["historical_train_count_scaler"]!=432 or source["historical_minus_current"]!={"train":["bp_354"],"validation":[]} or
        not embedding["exact_probability_match"] or comparison["validation_videos"]!=93 or
        selected["checkpoint_sha256"]!=sha(out/selected["checkpoint"])):
        raise RuntimeError("RIGHT no verificada")
    table="\n".join(f'| {r["seed"]} | {r["best_epoch"]} / {r["epochs_run"]} | {r["train_original_loss"]:.4f} / {r["best_validation_loss"]:.4f} | '
                    f'{r["validation_metrics"]["accuracy"]:.4f} | {r["validation_metrics"]["precision"]:.4f} | '
                    f'{r["validation_metrics"]["recall"]:.4f} | {r["validation_metrics"]["f1"]:.4f} | '
                    f'{r["validation_metrics"]["macro_f1"]:.4f} | {r["validation_metrics"]["balanced_accuracy"]:.4f} | '
                    f'{r["validation_metrics"]["roc_auc"]:.4f} | {r["validation_metrics"]["pr_auc"]:.4f} | '
                    f'{r["validation_metrics"]["TN"]}/{r["validation_metrics"]["FP"]}/{r["validation_metrics"]["FN"]}/{r["validation_metrics"]["TP"]} |'
                    for r in scores)
    fps=[]
    for r in scores:
        entries=rows_csv(out/("seed_"+str(r["seed"]))/"validation_predictions.csv")
        ids=[row["video_id"] for row in entries if row["false_positive"]=="1"]
        fps.append(f'- Seed {r["seed"]}: {ids}.')
    fptext="\n".join(fps)
    report=f"""# Reentrenamiento BP_RIGHT PoseCNN cohorte común V1

**RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE**. Dos inicializaciones completamente nuevas; 431 Train, 93 Validation, Test=93 **solo IDs**, 12 sin split excluidos. `bp_354` (MP4 duplicado exacto de `bp_027`) sigue en el split histórico Train pero fue excluido *solo de la cohorte derivada*, sin modificar splits/etiquetas/NPY. Ningún NPY Test abierto o evaluado; no se entrenó ninguna fusión.

## Fuentes, protocolo y procedencia

- `source_manifest.json` documenta coincidencia Train/Validation por ID/etiqueta y hashes de los **524 NPY** usados, sus digest históricos, centro/escala y configuración. Se comprobó `(T,8,3)` contra frames auditados para cada archivo, sin Inf; secuencias normalizadas `(170,8,3)`, entrada `(N,3,170,8)`. Orden: rodilla/tobillo/cadera/hombro/mano/codo/talón/punta; canales `(xyn_x,xyn_y,confianza YOLO)`. Grid `linspace(0,T_original-1,170)` igual al histórico y SegCNN; ausencias YOLO pose: interpolación XY interior ≤5, confianza faltante 0. SegCNN puede abstenerse sobre los mismos instantes; no se fuerza imputación conjunta.
- `StandardScaler` nuevo fit **solo** sobre 431 Train originales × 4080 coordenadas aplanadas, antes del aumento; `scaler_train_original.pkl`, SHA `{audit['new_scaler_sha256']}`. Histórico tenía 432 Train según `historical_right_audit.json` y scaler SHA `{history['historical_scaler_sha256']}`. Ningún peso, estado BatchNorm, optimizador o scaler históricos se transfirió. `development_original.npz` almacena **solo Train/Validation**.
- Se reprodujo `training.common.train_utils.PoseCNN` (102.017 parámetros): Conv2D 3×3 same 32/64/128 + BN/ReLU, MaxPool(2,1), Dropout2D .40, GAP128, Linear128→64 ReLU/Dropout .50/Linear64→1/Sigmoid. BCELoss + Adam lr .001 decay 1e-4, batch16, scheduler plateau patience4/factor.5, early stopping patience8, máx 200. Train ×4 (original+3 augmentations) con translación/escala/ruido XY/shift temporal, sin flip/rotación; por seed 1.724 secuencias derivadas de 431 videos, no levantamientos físicos nuevos. Threshold 0,5 fijo. Selección mejor época y seed por **menor loss Validation**.
- Baseline histórico RIGHT conservado para referencia: checkpoint SHA `{history['historical_checkpoint_sha256']}`, estado compatible con arquitectura, scaler fit 432, mejor época histórica de `history.csv` {history['history_best_validation_epoch']}. No se leyeron métricas ni predicciones Test históricas. El Test lateral fue evaluado históricamente para PoseCNN y no es completamente nuevo para todo el proyecto.

## Dos ejecuciones desde cero (Validation por video, 1=válido)

| Seed | Mejor época / corridas | Loss Train original / Val | Acc | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | TN/FP/FN/TP |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table}

**Seed seleccionada: {selected['seed']}**; checkpoint `{selected['checkpoint']}`, SHA `{selected['checkpoint_sha256']}`, loss Validation {selected['validation_loss']:.4f}. El FP significa intento inválido declarado válido. Falsos positivos:
{fptext}

Las brechas loss Validation menos Train original fueron {[(r['seed'],round(r['best_validation_loss']-r['train_original_loss'],4)) for r in scores]}; hay señal de generalización imperfecta, sin ajustes adicionales. Curvas y matrices en `seed_*/`; métricas y predicciones en archivos por semilla. Las probabilidades de Validation se reprodujeron al recargar ambos checkpoints con máximo error **0**.
- `PoseEmbeddingAdapter.extract_embeddings(x)` obtiene **{embedding['embedding_dimension']}** dimensiones (post-Linear128→64/ReLU); la reconstrucción de probabilidades fue exacta, sin modificar la arquitectura ni los pesos. No se exportaron embeddings Test.

## Comparación preliminar con SegCNN RIGHT (idénticos 93 Validation)

| Modelo | Accuracy | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{compare_rows(comparison)}

{paired(comparison)}

`validation_pose_seg_paired.csv` contiene probabilidades por ID, flags de error/FP, desacuerdos y comparación; `validation_pose_seg_comparison.json` incluye errores compartidos y exclusivos. Es evidencia **descriptiva**, no resultado de una fusión ni prueba de independencia de modalidades. Esta PoseCNN está ya en la cohorte común necesaria para diseñar posteriormente la fusión RIGHT mediante un protocolo de desarrollo que no ajuste hiperparámetros con Test. La identidad física entre vistas y atletas/sesiones sigue sin verificarse.
"""
    (out/"Reporte_Reentrenamiento_BP_RIGHT_PoseCNN_Common_V1.md").write_text(report,encoding="utf-8")
    return comparison,selected


def main():
    left=left_report();right,sel=right_report()
    text=f"""# Preparación de fusión BP_LEFT y BP_RIGHT — sin entrenamiento de fusión

Sistemas independientes, sin red SIDE. **No se abrió Test NPY/features ni se ajustó ninguna regla con Test.** Los Test PoseCNN laterales ya habían sido evaluados históricamente; no son inéditos para todo el proyecto. La independencia física entre vistas, atletas y sesiones no está acreditada.

| Vista | PoseCNN | Train / Validation | Pose F1 | Seg FULL F1 | Seg SIN relación F1 | Pareo Pose/SIN relación: compartidos, exclusivos Pose, exclusivos Seg | Embedding |
|---|---|---:|---:|---:|---:|---|---:|
| LEFT | Histórico reproducido, seed42, mejor época 45 | 422 / 91 | {left['metrics']['pose']['f1']:.4f} | {left['metrics']['seg_full']['f1']:.4f} | {left['metrics']['seg_sin_relacion']['f1']:.4f} | {left['paired']['seg_sin_relacion']['shared_count']} / {left['paired']['seg_sin_relacion']['pose_only_count']} / {left['paired']['seg_sin_relacion']['seg_only_count']} | 64 |
| RIGHT | Reentrenado desde cero, seed{sel['seed']} | 431 / 93 | {right['metrics']['pose']['f1']:.4f} | {right['metrics']['seg_full']['f1']:.4f} | {right['metrics']['seg_sin_relacion']['f1']:.4f} | {right['paired']['seg_sin_relacion']['shared_count']} / {right['paired']['seg_sin_relacion']['pose_only_count']} / {right['paired']['seg_sin_relacion']['seg_only_count']} | 64 |

LEFT conserva su checkpoint/scaler históricos porque coinciden todos los IDs, etiquetas, exclusiones, preprocesamiento y Validation loss (diferencia ≈2×10⁻⁹), además de embeddings sin cambio de probabilidad. RIGHT excluye `bp_354` de Train para alinear la cohorte SegCNN y posee dos checkpoints nuevos independientes. Los 64 embeddings son posteriores a la capa oculta y anteriores a la salida final. `training.bp.experiments.bp_posecnn_common_v1.core.PoseEmbeddingAdapter` permite extraerlos sin alterar pesos; cualquier calibración/fusión posterior debe usar únicamente development y predicciones fuera de muestra en su protocolo, nunca scores Test para elegir modelos. La SegCNN lateral es parcialmente pose-assisted; la complementariedad observada es solo descriptiva.

**LEFT_POSECNN_HISTORICAL_COMPATIBLE**; **RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE**. Detalles, hashes y FP en reportes de cada vista. No se entrenó fusión ni se evaluó Test.
"""
    (ROOT/"training/docs/bp/Reporte_Preparacion_Fusion_BP_LEFT_RIGHT.md").write_text(text,encoding="utf-8")
    print("Reportes LEFT/RIGHT y preparación descriptiva emitidos",flush=True)


if __name__=="__main__":main()
