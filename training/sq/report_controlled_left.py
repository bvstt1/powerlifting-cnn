"""Reporte reproducible left v2 vs point_xy_v1 sin cambiar entrenamiento.

python -m training.sq.report_controlled_left
"""
import json

from training.sq import controlled_left as exp
from training.sq import train_posecnn as sq
from training.sq.run_analysis import read_predictions


def main():
    choice=json.loads(exp.DECISION.read_text(encoding="utf-8"))
    old=choice["reference_cv"]; new=choice["variant_cv"]
    base=json.loads((exp.BASE["results"]/"complete_metrics_recovered.json").read_text(encoding="utf-8"))
    val=json.loads((exp.RUN["results"]/"validation_metrics.json").read_text(encoding="utf-8"))
    test=json.loads((exp.RUN["results"]/"test_metrics.json").read_text(encoding="utf-8"))
    labels={r["video_id"]:r for r in sq.load_labels(exp.BASE["labels"])}
    a={r["video_id"]:r for r in read_predictions(exp.BASE["results"]/"cv_predictions.csv")}
    b={r["video_id"]:r for r in read_predictions(exp.RUN["results"]/"cv_predictions.csv")}
    assert set(a)==set(b)
    oldfp=set(choice["corrected_false_valid"]+choice["persistent_false_valid"])
    newfp=set(choice["persistent_false_valid"]+choice["new_false_valid"])
    report_rows=[]
    for ident in sorted(oldfp|newfp):
        kind=("corregido" if ident in oldfp-newfp else "persistente" if ident in oldfp&newfp else "nuevo")
        report_rows.append({"video_id":ident,"fold":int(a[ident]["fold"]),"estado":kind,
                            "prob_v2":float(a[ident]["probability_valid"]),
                            "prob_point_xy":float(b[ident]["probability_valid"]),
                            "codigos":labels[ident]["codigos"],
                            "criterios":labels[ident]["criterios"],
                            "video_path":f"dataset/sq/left/{ident}.mp4"})
    sq.save_csv(exp.RUN["results"]/"cv_invalid_before_after.csv",report_rows,list(report_rows[0]))
    keys=(("accuracy","Accuracy"),("balanced_accuracy","Balanced accuracy"),
          ("precision_valid","Precision válido"),("recall_valid","Recall válido"),
          ("f1_valid","F1 válido"),("precision_invalid","Precision inválido"),
          ("recall_invalid","Recall inválido"),("f1_invalid","F1 inválido"),
          ("f1_macro","F1 macro"),("roc_auc","ROC-AUC"))
    lines=["# SQ left — experimento controlado de scaler por punto",
           "", "**Referencias:** `baseline_t170_v2` intacto, candidato `point_xy_v1` entrenado en ejecución nueva. Dos redes SQ siguen independientes por vista; no se usa front/right como fuente de datos.",
           "", "## Auditoría previa: 34 falsos válidos de CV de left v2", "",
           "- Los **34** vídeos con etiqueta 0 y predicción externa 1 (cada uno evaluado por su fold correspondiente) están en `training/results/sq/left/posecnn/runs/baseline_t170_v2/invalid_as_valid_manual_review.csv`. Los 2 errores test (`sq_199`, `sq_331`) se revisan **aparte**; nunca se utilizaron para elegir scaler o threshold.",
           "- Confirmación en los 34 falsos válidos y 75 inválidos correctos de desarrollo: `video_id` coincide con CSV/video/pose, todos los vídeos presentes, `T_video=T_pose` (tolerancia ±1) y reconstrucción del preprocesamiento coincide con los arrays congelados (error <2e-5). No se alteraron etiquetas.",
           "- Los 34 falsos válidos tienen missing XY=0 y confianza mediana ~0.9995; los inválidos correctos muestran valores similares. Alta confianza no garantiza articulaciones correctas. Salto máximo de keypoints: mediana 0.097 tibias en falsos válidos frente a 0.105 en correctos; por sí solo no señala corrupción generalizada.",
           "- La normalización espacial fija y la interpolación temporal **conservan** la diferencia vertical cadera–rodilla dentro de tolerancia de remuestreo: mediana del máximo por secuencia -0.078 antes y -0.079 después (falsos válidos). Esta señal es proxy geométrica, no regla exacta para SQ-01. Algunos intentos fallan por SQ-02 (rebote temporal) o SQ-04 (pies); cinco puntos sin barra/torso no observan todas las faltas. Un error de clasificación con señal presente no obliga a cambiar la etiqueta.",
           "- Hoja de contacto visual de ocho vídeos **solo de desarrollo**, incluidos los casos de alta probabilidad `sq_001`, `sq_324`, `sq_247`, `sq_315`, `sq_197` y dos sin código (`sq_208`, `sq_211`), en `training/results/sq/left/posecnn/development_diagnostics/cv_false_valid_contact_sheet.jpg`. Los puntos siguen piernas/pies en los frames muestreados; una hoja de cinco instantes no verifica rebotes o pasos durante todo el vídeo. No se atribuye cada error de todos los 34 a una causa única.",
           "- Clasificación causal: **extracción** (sin evidencia de fallas estructurales generales, pero geometría puede fallar en frames puntuales); **preprocesamiento** (preserva proxy de profundidad y archivos, pero normalización/resampling y scaler heredado pueden atenuar señales temporales); **clasificación** (aprobación incorrecta pese a datos disponibles, o falta no observable con estos cinco puntos). No se corrigió ninguna etiqueta por desacuerdo con la red.",
           "", "## Experimento: un solo cambio respecto a left v2", "",
           "`PointXYScaler`: media/desviación de XY por keypoint (5×2) compartida entre 170 frames; canal confianza pasa sin estandarizar. Scaler ajustado solo con train original dentro de cada fold (n=213/214) y fit final (n=258); aumento ×4 solo de train antes de transformar. Sin cambiar PoseCNN (102017 parámetros), T=170, K=5, seed 42, hiperparámetros ni IDs/folds. Umbral 0.5 fijo.",
           "", "Preparación aislada `training/data/sq/left/posecnn/runs/point_xy_v1/development.npz` con **314** vídeos de desarrollo (sin arrays de test); scaler/checkpoints/curvas/config en `training/models/sq/left/posecnn/runs/point_xy_v1/`; métricas/predicciones en `training/results/sq/left/posecnn/runs/point_xy_v1/`. Splits y memberships v2 en `.../baseline_t170_v2/provenance/` con hashes SHA-256; variante comparada por vídeo y fold. Config efectiva registra versiones, semillas y arquitectura.",
           "", "## CV de 5 folds — desarrollo solamente", "",
           "| Métrica (media ± desviación estándar) | Left v2 | Point XY |", "|---|---:|---:|"]
    for key,name in keys:
        lines.append(f"| {name} | {old[key]['mean']:.4f} ± {old[key]['std']:.4f} | {new[key]['mean']:.4f} ± {new[key]['std']:.4f} |")
    lines += ["", "**Lectura:** recall inválidos 0.6870→0.7225 (+0.0355), pero dispersión **0.0728→0.1766**. Mejoran también F1 macro y balanced accuracy en promedio, y F1 válido no baja. **No es mejora uniforme:** folds 1 y 5 disminuyen en recall inválido. Regla y ranking congelados en `selection_development_only.json` antes de consultar test; elige la variante para estudio, no confirma seguridad de uso.",
              "", "| Fold | Inválidos | Left v2 matriz | Point XY matriz | Recall inválido v2 → nuevo | F1 macro v2 → nuevo | Recall válido v2 → nuevo |",
              "|---|---:|---|---|---:|---:|---:|"]
    for arow,brow in zip(choice["reference_folds"],choice["variant_folds"]):
        assert arow["fold"]==brow["fold"]
        lines.append(f"| {arow['fold']} | {arow['n_invalid']} | {arow['matrix']} | {brow['matrix']} | {arow['recall_invalid']:.4f}→{brow['recall_invalid']:.4f} | {arow['f1_macro']:.4f}→{brow['f1_macro']:.4f} | {arow['recall_valid']:.4f}→{brow['recall_valid']:.4f} |")
    lines += ["", "**Variante point XY: métricas adicionales por fold** (P/R/F1 V=válido, I=inválido):", "",
              "| Fold | P V | R V | F1 V | P I | R I | F1 I | Balanced acc | F1 macro | ROC-AUC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in choice["variant_folds"]:
        lines.append("| " + " | ".join([str(fold["fold"])] +
                                      [f"{fold[k]:.4f}" for k in ("precision_valid","recall_valid","f1_valid",
                                                                     "precision_invalid","recall_invalid","f1_invalid",
                                                                     "balanced_accuracy","f1_macro","roc_auc")]) + " |")
    lines += ["", "### ¿Qué errores se redujeron? (solo CV externo)", "",
              f"- **17** inválidos antes aprobados ahora rechazados; **17** permanecen aprobados; **13** falsos válidos nuevos. Resultado neto 34→30 falsos válidos de 109 inválidos en development; la media por fold se reporta arriba.",
              f"- Válidos mal rechazados: {len(choice['corrected_false_invalid'])} corregidos, {len(choice['persistent_false_invalid'])} persistentes, {len(choice['new_false_invalid'])} nuevos. Sin sumar categorías de distintas clases.",
              "- `cv_invalid_before_after.csv` registra ID, fold, probabilidad válida antes/después, código, criterio y vídeo para cada error antiguo o nuevo. Códigos pueden coexistir; no sumar apariciones como vídeos exclusivos.",
              "", "| Estado | Vídeos de desarrollo (ID) |", "|---|---|"]
    for state,key in (("Corregidos","corrected_false_valid"),("Persistentes","persistent_false_valid"),("Nuevos","new_false_valid")):
        lines.append(f"| {state} | {', '.join(choice[key]) or '—'} |")
    lines += ["", "## Fit final nuevo (sin seleccionar por test)", "",
              f"Checkpoint mejor época **{val['best_epoch']}** elegida por pérdida de validation. Validation (n=56): accuracy {val['accuracy']:.4f}, balanced accuracy {val['balanced_accuracy']:.4f}, F1 macro {val['f1_macro']:.4f}, recall inválidos {val['recall_invalid']:.4f}, recall válidos {val['recall_valid']:.4f}; matriz `{val['matrix']}`. Baseline v2 en su validation: matriz `{base['validation']['matrix']}`, recall inválidos {base['validation']['recall_invalid']:.4f}, F1 macro {base['validation']['f1_macro']:.4f}.",
              "", "## Test **exploratorio** (consultado históricamente en left v2)", "",
              "| Métrica | Left v2 | Point XY |", "|---|---:|---:|"]
    for key,name in keys:
        lines.append(f"| {name} | {base['test'][key]:.4f} | {test[key]:.4f} |")
    lines += [f"| Matriz [[TN,FP],[FN,TP]] | {base['test']['matrix']} | {test['matrix']} |", "",
              "**Advertencia:** test tiene 19 inválidos/37 válidos. En el nuevo modelo recall inválidos test **baja** 0.8947→0.8421 (2→3 FP), mientras recall válidos sube 0.8919→0.9189 (4→3 FN). No usar este test previamente consultado para cambiar la selección o el umbral; la mejora media de desarrollo no implica mejora confirmada en el test. Left v2 se conserva como referencia; la variante queda como candidato exploratorio hasta evaluación externa nueva.", ""]
    old_test={r["video_id"]:r for r in read_predictions(exp.BASE["results"]/"test_predictions.csv")}
    new_test={r["video_id"]:r for r in read_predictions(exp.RUN["results"]/"test_predictions.csv")}
    old_fp={i for i,r in old_test.items() if r["true_label"]=="0" and r["predicted_label"]=="1"}
    new_fp={i for i,r in new_test.items() if r["true_label"]=="0" and r["predicted_label"]=="1"}
    assert old_fp=={"sq_199","sq_331"} and len(new_fp)==3
    lines += ["**Errores test separados del análisis de desarrollo:**", "",
              f"- Falsos válidos v2: {', '.join(sorted(old_fp))}. Nuevo: {', '.join(sorted(new_fp))}. Corregidos: {', '.join(sorted(old_fp-new_fp)) or 'ninguno'}; persistentes: {', '.join(sorted(old_fp&new_fp)) or 'ninguno'}; nuevos: {', '.join(sorted(new_fp-old_fp)) or 'ninguno'}.",
              "- SQ-01, SQ-02 y SQ-04 pueden coexistir; el desglose por códigos de left v2 permanece en `complete_metrics_recovered.json`. El nuevo `test_predictions.csv` incluye probabilidades por vídeo; estas observaciones no guiaron CV ni threshold.",
              "", "## Verificación y comandos", "",
              "```powershell", "python -m training.sq.export_left_invalid_review", "python -m training.sq.diagnose_left_invalid",
              "python -m unittest training.tests.test_sq_controlled_left training.tests.test_sq_controlled_front training.tests.test_sq_posecnn",
              "python -m training.sq.controlled_left --stage prepare", "python -m training.sq.controlled_left --stage cv",
              "python -m training.sq.controlled_left --stage verify", "python -m training.sq.controlled_left --stage select",
              "python -m training.sq.controlled_left --stage fit-final", "python -m training.sq.controlled_left --stage evaluate-test",
              "python -m training.sq.report_controlled_left", "```", "",
              "Test y baseline nunca se reentrenan durante CV de la variante. Las etapas protegen archivos existentes contra reescrituras. Pendiente: revisión manual de los 17 persistentes, especialmente faltas SQ-01/SQ-02/SQ-04, y confirmación en intentos nuevos nunca consultados. La hoja de contacto no sustituye una auditoría completa de todos los frames."]
    target=sq.ROOT/"training"/"docs"/"SQ_LEFT_point_xy_v1.md"
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(target)


if __name__=="__main__":
    main()
