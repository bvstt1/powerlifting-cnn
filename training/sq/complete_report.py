"""Genera reporte conjunto desde artefactos verificados, sin entrenar.

python -m training.sq.complete_report
"""
import json

from training.sq import train_posecnn as sq
from training.sq.controlled_front import REFERENCE, SELECTED, paths
from training.sq.run_analysis import read_predictions, metrics


def table_metric(data, names):
    return "\n".join(f"| {display} | {data[key]:.4f} |" for key, display in names)


def main():
    select = json.loads(SELECTED.read_text(encoding="utf-8"))
    assert select["selected_variant"] == "D"
    d = paths("D")
    left = sq.paths("left", "baseline_t170_v2")
    left_data = json.loads((left["results"] / "complete_metrics_recovered.json").read_text(encoding="utf-8"))
    d_test = json.loads((d["results"] / "test_metrics.json").read_text(encoding="utf-8"))
    d_val = json.loads((d["results"] / "validation_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((sq.ROOT / "training" / "results" / "sq" / "front" / "posecnn" / "development_diagnostics" / "summary.json").read_text(encoding="utf-8"))
    left_split = json.loads((left["models"] / "provenance" / "manifest.json").read_text(encoding="utf-8"))["split_ids"]
    names = (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
             ("precision_valid", "Precision válido"), ("recall_valid", "Recall válido"),
             ("f1_valid", "F1 válido"), ("precision_invalid", "Precision inválido"),
             ("recall_invalid", "Recall inválido / especificidad"),
             ("f1_invalid", "F1 inválido"), ("f1_macro", "F1 macro"),
             ("npv", "VPN"), ("roc_auc", "ROC-AUC"))
    cv_fields = (("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "F1 macro"),
                 ("roc_auc", "ROC-AUC"), ("recall_invalid", "Recall inválido"),
                 ("f1_valid", "F1 válido"))
    cv = select["cv_comparison"]
    rows = []
    for variant, desc in (("K10_reference", "K10, T170, hombros, temporal, patience 8"),
                          ("A", "solo patience=20"), ("B", "solo T=96"),
                          ("C", "solo escala tibias"), ("D", "solo scaler XY por punto")):
        scores = cv[variant]
        cells = [f"{scores[key]['mean']:.4f} ± {scores[key]['std']:.4f}" for key, _ in cv_fields]
        rows.append("| " + " | ".join([variant, desc] + cells) + " |")
    fold_lines = []
    for f in left_data["cv_folds"]:
        fold_lines.append("| " + " | ".join([str(f["fold"]), str(f["n"]), str(f["n_invalid"]), str(f["n_valid"])] +
                                           [f"{f[k]:.4f}" for k, _ in names if k != "npv"] +
                                           [str(f["matrix"])]) + " |")
    aggregate = "\n".join(f"| {display} | {left_data['cv_summary'][key]['mean']:.4f} ± {left_data['cv_summary'][key]['std']:.4f} |"
                          for key, display in names)
    defects = "\n".join(f"| {code} | {item['n']} | {item['correct']} | {item['n']-item['correct']} | {', '.join(item['errors']) or '—'} |"
                        for code, item in left_data["fault_codes_nonexclusive"].items())
    errors = "\n".join(f"| {r['video_id']} | {r['true_label']} | {r['predicted_label']} | {float(r['probability_valid']):.4f} |"
                       for r in left_data["test_errors"])
    front_rows = read_predictions(d["results"] / "test_predictions.csv")
    front_labels = {r["video_id"]: r for r in sq.load_labels(REFERENCE["labels"])}
    front_codes = sorted({c for r in front_rows for c in front_labels[r["video_id"]]["codigos"].split("|") if c})
    front_defects = []
    for code in front_codes:
        selected = [r for r in front_rows if code in front_labels[r["video_id"]]["codigos"].split("|")]
        wrong = [r["video_id"] for r in selected if r["true_label"] != r["predicted_label"]]
        front_defects.append(f"| {code} | {len(selected)} | {len(selected)-len(wrong)} | {len(wrong)} | {', '.join(wrong) or '—'} |")
    thresholds = []
    validation_rows = read_predictions(d["results"] / "validation_predictions.csv")
    for threshold in (.4,.5,.6):
        m = metrics([r["true_label"] for r in validation_rows],
                    [r["probability_valid"] for r in validation_rows], threshold)
        thresholds.append(f"| {threshold:.1f} | {m['matrix']} | {m['recall_invalid']:.4f} | {m['recall_valid']:.4f} | {m['balanced_accuracy']:.4f} |")
    matrix = d_test["matrix"]
    text = f"""# SQ PoseCNN — tanda frontal controlada y métricas recuperadas de left v2

## Alcance y trazabilidad

- Dos modelos independientes: `sq_front` (YOLO 16 pts; subset K10) y `sq_left` (YOLO K5). Etiquetas: 0=inválido, 1=válido; matriz de confusión **filas reales [0,1], columnas predichas [0,1]**, `[[TN,FP],[FN,TP]]`. Umbral fijo 0.5.
- Referencias intactas: `baseline_t170_v1` (front K16), `baseline_t170_k10` (front), `baseline_t170_v2` (left). Snapshots verificables en `training/models/sq/<vista>/posecnn/runs/<run>/provenance/` (`split.npz`, `split.csv`, cinco `fold_*_membership.csv`, SHA-256 e IDs en `manifest.json`). No cambiaron particiones ni folds entre variantes front.
- `training/data/sq/front/posecnn/runs/controlled_<a|b|c|d>/development.npz` aísla X/y de **377** development para cada configuración. No contiene X_test/y_test. `training/models` y `training/results` repiten ejercicio/vista/run. Config efectiva también en `training/models/.../config.json` (semillas 42/43–47, versiones, scaler y 102017 parámetros entrenables de la misma `PoseCNN` global-average-pooling; T/K NO cambian este número).
- El test front había sido consultado antes en K16 y K10. Su lectura posterior en D es **exploratoria** y no se usó en la selección. Variantes A/B/C no tienen métricas test. No se seleccionaron seeds, longitudes adicionales ni umbral por test.

## Diagnóstico frontal basado en desarrollo

- `video_id`/CSV/npy concordantes en 444 muestras; `sq_148` y `sq_313` carecían de vídeo y fueron eliminados antes con backups, sin reasignar etiquetas ni eliminar casos por error del modelo. Las vistas **no son necesariamente el mismo intento**: no se utilizó acuerdo left-front como evidencia.
- La secuencia del extractor YOLO define cadera 2/3, rodillas 4/5, tobillos 9/6. La hoja de contacto `training/results/sq/front/posecnn/development_diagnostics/pose_video_contact_sheet.jpg` muestra cuatro vídeos **de desarrollo** en cinco frames con los puntos superpuestos; en esa muestra siguen las articulaciones y no se ven inversiones obvias de lados, pero no certifica las 444 trayectorias. Escala de tibias izq/der: razón mediana {summary['ratio_tibias']['median']:.3f}, P95 {summary['ratio_tibias']['p95']:.3f}, máximo {summary['ratio_tibias']['max']:.3f}; hay outliers a revisar. Salto máximo por video mediano {summary['largest_jump_in_tibias']['median']:.3f} tibias, P95 {summary['largest_jump_in_tibias']['p95']:.3f}, máximo {summary['largest_jump_in_tibias']['max']:.3f}. La confianza YOLO alta **no prueba** precisión de posición.
- T mediana {summary['T']['median']:.0f}, P95 {summary['T']['p95']:.1f} en desarrollo (T=170 interpola muchos vídeos). La hoja muestra espera/posición de salida al inicio y al final, variable entre vídeos. Fracción mediana del movimiento vertical total de cadera en primer/último sexto: {summary['hip_start_motion_fraction']['median']:.3f}/{summary['hip_end_motion_fraction']['median']:.3f} (proxy, no segmentación automática del intento). No se cambió el recorte en esta tanda.
- El `StandardScaler` anterior ajustaba por `(t,punto,canal)`; 984/1700 features de confianza de K10 tienen desviación <0.001 en train, y al estandarizar pueden amplificar fluctuaciones mínimas. D comparte estadísticas XY por punto entre frames y deja confianza en [0,1]. Ajuste **solo en train original** dentro de cada fold y en el fit final; aumento en XY no toca confianza y desplaza temporalmente todos los canales sin cambiar valores.
- `PoseCNN` real (`training/common/train_utils.py`) emplea *global average pooling*: 102017 parámetros entrenables para K10, K16, T96/T170. `T×K×C` es dimensión de entrada al scaler, no cantidad de parámetros. Un checkpoint temprano no permite afirmar que un fold sea «no aprendible» ni sabemos el techo de front.

## Primera tanda controlada: selección **exclusivamente CV de desarrollo**

| Variante | Único cambio sobre K10 | Balanced acc CV | F1 macro CV | ROC-AUC CV | Recall inválido CV | F1 válido CV |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Cada celda CV es media ± desviación estándar poblacional de 5 folds estratificados sobre desarrollo, con validación interna separada para early stopping; mismos IDs por fold que K10 (sin asumir evento compartido entre cámaras). **D** mejora en promedio balanced accuracy (+0.0296), F1 macro (+0.0306), AUC (+0.0223), recall inválido (+0.025) y reduce dispersión de las dos primeras. A y B no mejoran de forma consistente; C empeora. No se justificó combinar cambios. Regla congelada antes de evaluar test: maximizar media de balanced accuracy + F1 macro, AUC para desempate. Ranking y SHA-256 de predicciones CV: `training/results/sq/front/posecnn/controlled_selection.json`.

### Variante C: escala de tibias

Centro frontal original fijo (mediana de punto medio de hombros). Escala fija = **mediana de todas las distancias positivas y finitas** rodilla izquierda 4–tobillo izquierdo 9 y rodilla derecha 5–tobillo derecho 6, calculadas solo con ambos puntos observados en el frame original y confianza >0. Los huecos interpolados no contribuyen a la escala. Se mantuvo toda la trayectoria; C empeoró en CV, por lo que no pasó a test.

### D: detalles de CV por fold

| Fold | N | Balanced acc | F1 macro | AUC | Recall inválido | F1 válido | Matriz |
|---|---:|---:|---:|---:|---:|---:|---|
"""
    for row in read_predictions(d["results"] / "cv_metrics.csv"):
        text += f"| {row['fold']} | {row['n']} | {float(row['balanced_accuracy']):.4f} | {float(row['f1_macro']):.4f} | {float(row['roc_auc']):.4f} | {float(row['recall_invalid']):.4f} | {float(row['f1_valid']):.4f} | {row['matrix']} |\n"
    text += f"""
### D final: validation y test

| Métrica | Validation (67) | Test exploratorio (67) |
|---|---:|---:|
"""
    for key, display in names:
        text += f"| {display} | {d_val[key]:.4f} | {d_test[key]:.4f} |\n"
    text += f"""| Matriz | {d_val['matrix']} | {d_test['matrix']} |

D test: 35 inválidos + 32 válidos; 9 errores: **6 FP** (inválidos aceptados) y **3 FN** (válidos rechazados). Checkpoint seleccionado por mínima pérdida de validation en época {d_val['best_epoch']}; test no intervino en ello. Frente a K10 referencia (test acc 0.7910, F1 válido 0.7812, AUC 0.8938, matriz [[28,7],[7,25]]) mejora observada en el mismo test, pero **no confirmatoria** por consulta previa de ese test.

**Desglose descriptivo de códigos en test D** (no utilizado para seleccionar; un vídeo puede tener varios códigos):

| Código | N apariciones | Correctos | Errores | Videos errados |
|---|---:|---:|---:|---|
{chr(10).join(front_defects)}

### Sensibilidad a umbral en validation de D (ilustrativa, umbral fijo 0.5 mantenido)

| Umbral | Matriz validation | Recall inválido (evitar aceptar nulos) | Recall válido (evitar rechazar válidos) | Balanced acc |
|---:|---|---:|---:|---:|
{chr(10).join(thresholds)}

No se optimizó ni seleccionó un umbral nuevo: se reporta el intercambio sobre validation únicamente. No se usa el fold externo de CV ni test para elegir threshold.

## `sq_left` v2: recuperación **sin reentrenamiento**

`training/models/sq/left/posecnn/runs/baseline_t170_v2/`: checkpoint SHA-256 `{left_data['checkpoint_sha256']}`, scaler `{left_data['scaler_sha256']}`, split congelado `{left_data['split_sha256']}`. Scaler vio exactamente **258** muestras train originales, 2550 features; input (170,5,3), PoseCNN 102017 parámetros. Predicciones validation recuperadas por inferencia desde checkpoint+scaler+split en `validation_predictions_recovered.csv`; su matriz coincide con la validation reportada originalmente. Test y CV se recalcularon a partir de predicciones guardadas; folds/IDs contrastados con `provenance/`. `complete_metrics_recovered.json` contiene resultados y errores verificables.

**Distribución:** dataset utilizable 370 (un duplicado excluido); train **{len(left_split['train'])}**, validation **{len(left_split['validation'])}**, test **{len(left_split['test'])}**. Validation 19 inválidos/37 válidos; test 19/37. En CV solo development (314 vídeos), folds de 63/63/63/63/62.

| Métrica | Validation | Test |
|---|---:|---:|
"""
    for key, display in names:
        text += f"| {display} | {left_data['validation'][key]:.4f} | {left_data['test'][key]:.4f} |\n"
    text += f"""| Matriz [[TN,FP],[FN,TP]] | {left_data['validation']['matrix']} | {left_data['test']['matrix']} |

**CV por fold** (columnas V = válido, I = inválido):

| Fold | N | I | V | Accuracy | Balanced acc | Prec V | Rec V | F1 V | Prec I | Rec I | F1 I | F1 macro | ROC-AUC | Matriz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(fold_lines)}

| Métrica | CV media ± std |
|---|---:|
{aggregate}

**Test left v2: errores por vídeo** (p = probabilidad de válido):

| Video | Real | Predicho | p válido |
|---|---:|---:|---:|
{errors}

**Códigos de falta no exclusivos**: un intento puede tener varios, por lo que los conteos por código **no deben sumarse como número de vídeos distintos**.

| Código | N apariciones en test | Correctos | Errores | Videos errados |
|---|---:|---:|---:|---|
{defects}

Curva left v2: {left_data['history']['n_epochs']} épocas, mejor checkpoint época {left_data['history']['best_epoch']}; train loss {left_data['history']['first_train_loss']:.4f}→{left_data['history']['last_train_loss']:.4f}, validation loss {left_data['history']['first_val_loss']:.4f}→{left_data['history']['last_val_loss']:.4f}. CSV completo en `training/models/sq/left/posecnn/runs/baseline_t170_v2/history.csv`.

## Comparación final (cada red es independiente; tamaños/splits distintos)

| Modelo | Test N | Acc | Balanced acc | Prec válido | Rec válido | F1 válido | F1 inválido | F1 macro | AUC | Matriz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SQ front D (exploratorio) | 67 | {d_test['accuracy']:.4f} | {d_test['balanced_accuracy']:.4f} | {d_test['precision_valid']:.4f} | {d_test['recall_valid']:.4f} | {d_test['f1_valid']:.4f} | {d_test['f1_invalid']:.4f} | {d_test['f1_macro']:.4f} | {d_test['roc_auc']:.4f} | {d_test['matrix']} |
| SQ left v2 (vigente, sin reentrenar) | 56 | {left_data['test']['accuracy']:.4f} | {left_data['test']['balanced_accuracy']:.4f} | {left_data['test']['precision_valid']:.4f} | {left_data['test']['recall_valid']:.4f} | {left_data['test']['f1_valid']:.4f} | {left_data['test']['f1_invalid']:.4f} | {left_data['test']['f1_macro']:.4f} | {left_data['test']['roc_auc']:.4f} | {left_data['test']['matrix']} |

Comparación **descriptiva**: las vistas no representan necesariamente el mismo intento, ni sus test son pareados. No mezclar predictions ni usar ID igual como prueba de evento común.

## Comandos reproducibles y controles

```powershell
python -m training.sq.run_analysis --stage freeze
python -m training.sq.run_analysis --stage left
python -m training.sq.diagnose_front
python -m unittest training.tests.test_sq_controlled_front training.tests.test_sq_posecnn training.tests.test_bp_side_posecnn
python -m training.sq.controlled_front --variant A --stage prepare
python -m training.sq.controlled_front --variant A --stage cv
# Repetir prepare/cv con B, C, D. Cada variante se guarda en su propio run.
python -m training.sq.select_front
python -m training.sq.controlled_front --variant D --stage fit-final
python -m training.sq.controlled_front --variant D --stage evaluate-test
python -m training.sq.complete_report
```

`freeze`, `prepare`, `cv`, `fit-final`, `select_front` y `evaluate-test` protegen artefactos existentes contra reescrituras; usar run-id nuevo para repetir. Verificación: 15 pruebas OK; shapes e IDs de desarrollo, folds exactos, scaler fit solo en train original, checkpoint/scaler left v2, predicciones y config. `git diff --check` no detectó errores; Git mostró avisos de conversión LF/CRLF en archivos ajenos.

## Pendientes reales

1. Revisar **outliers geométricos** y potenciales inversiones de lados a mano en vídeos de desarrollo identificables en `development_sequences.csv`; la inspección visual de 4 vídeos no cubre todo el conjunto.
2. Auditoría de la procedencia de etiquetas alrededor de vídeos inexistentes `sq_148/sq_313`; no inferir mapeo entre vistas. Mantener labels como están hasta evidencia independiente.
3. Repetir confirmación futura con otro holdout nunca consultado / nuevos levantamientos. D test actual es exploratorio.
4. Investigar fases de espera mediante segmentación temporal validada solo en desarrollo; no recortar basado en test.
5. Right: no hay CSV; ninguna red derecha entrenada.
"""
    target = sq.ROOT / "training" / "docs" / "SQ_POSECNN_controlled_front_left_v2.md"
    target.write_text(text, encoding="utf-8")
    print("Reporte:", target)


if __name__ == "__main__":
    main()
