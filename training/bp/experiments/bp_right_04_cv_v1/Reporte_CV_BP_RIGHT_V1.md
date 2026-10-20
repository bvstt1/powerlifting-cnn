# Validación cruzada estratificada BP_RIGHT — cinco folds sobre development

**CV_RETROSPECTIVE_VALIDATION_COMPLETE_RIGHT**. Fuente: Train+Validation históricos = **524 videos**; Test reservado (93 IDs) **no se abrió ni evaluó**. Variante SegCNN fija: `sin_relacion`; input C=90, según contrato histórico. Preselección previa para Test: **SegCNN SIN relación seed1337**, checkpoint protegido SHA-256 `28d1ddc1a1408d88bb7617b97d9cb7f86ef7b783fa735fd1d8326ee7ac25458b`. Esta CV NO cambió la preselección.

## Identidad, folds y ausencia de leakage

- Desarrollo procedente exclusivamente de 431 Train + 93 Validation originales; RIGHT excluye `bp_354` (MP4 duplicado de `bp_027`) antes de crear folds. La auditoría de hashes MP4 en development no encontró ningún par idéntico restante. No existen grupos físicos acreditados en BP lateral: se utilizó **StratifiedKFold(n_splits=5, shuffle=True, random_state=42) por video**, no mapeos por numeración. No puede afirmarse independencia de atleta, sesión o levantamiento físico; `fold_assignments.csv` y `fold_class_distributions.csv` conservan IDs/etiquetas y roles.
- Outer evaluation exclusivamente para el cálculo externo. En cada Outer Train se separó inner Validation ≈15 % con estratificación y seed 42+fold, común a PoseCNN/SegCNN/cabeza. Los checkpoints y scheduler se seleccionaron **solo con inner Validation**; no se seleccionaron seeds según Outer. Scaler Pose/Seg/embeddings fit **solo inner Train original**. Pose Train aumentado ×4 exclusivamente allí, sin flip/rotación. Todos los pesos, BatchNorm y estados de optimizer se inicializaron desde cero por fold. Los backbones de fusión se congelaron en `eval()`; embeddings inner Train son **in-sample**, embeddings Outer externos.
- Canales SegCNN idénticos al input congelado histórico (C=90); topología mantenida si alguna feature resulta constante/vacía en un inner Train, con política fija mediana=0 si vacía y std=1 si no hay dispersión. Ningún estadístico del Outer (ni de la unión de development) se ajustó. Insumos originales temporalmente remuestreados a 170 con las mismas reglas de ausencia/procedencia ya congeladas.
- `source_manifest.json` y `cv_integrity.json` contienen hashes de sources/checkpoints finales antes/después: intactos. Hashes individuales Train/Validation de NPY Pose y NPZ Seg, sin apertura de archivos Test. Archivos CV por fold (scalers, parámetros, curvas, checkpoints y probabilidades) están bajo `fold_01/` … `fold_05/`.

## Métricas por fold externo (una predicción por levantamiento)

| Fold | Modelo | N | Acc | F1 | Macro-F1 | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP | Mejor época Pose/Seg/Cabeza |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | pose | 105 | 0.9429 | 0.9362 | 0.9422 | 0.9710 | 0.9591 | 0.9483 | 0.0517 | 0.0638 | 55/3/3/44 | 17/56/fixed |
| 1 | seg_sin_relacion | 105 | 0.9524 | 0.9462 | 0.9518 | 0.9791 | 0.9516 | 0.9655 | 0.0345 | 0.0638 | 56/2/3/44 | 17/56/fixed |
| 1 | late_sin_relacion | 105 | 0.9714 | 0.9684 | 0.9712 | 0.9938 | 0.9934 | 0.9655 | 0.0345 | 0.0213 | 56/2/1/46 | 17/56/fixed |
| 2 | pose | 105 | 0.9143 | 0.8966 | 0.9117 | 0.9598 | 0.9547 | 0.9661 | 0.0339 | 0.1522 | 57/2/7/39 | 33/15/fixed |
| 2 | seg_sin_relacion | 105 | 0.8857 | 0.8605 | 0.8818 | 0.9057 | 0.9233 | 0.9492 | 0.0508 | 0.1957 | 56/3/9/37 | 33/15/fixed |
| 2 | late_sin_relacion | 105 | 0.9048 | 0.8810 | 0.9008 | 0.9506 | 0.9515 | 0.9831 | 0.0169 | 0.1957 | 58/1/9/37 | 33/15/fixed |
| 3 | pose | 105 | 0.9143 | 0.9032 | 0.9132 | 0.9510 | 0.9472 | 0.9153 | 0.0847 | 0.0870 | 54/5/4/42 | 35/47/fixed |
| 3 | seg_sin_relacion | 105 | 0.8952 | 0.8842 | 0.8943 | 0.9223 | 0.8805 | 0.8814 | 0.1186 | 0.0870 | 52/7/4/42 | 35/47/fixed |
| 3 | late_sin_relacion | 105 | 0.9048 | 0.8936 | 0.9037 | 0.9510 | 0.9433 | 0.8983 | 0.1017 | 0.0870 | 53/6/4/42 | 35/47/fixed |
| 4 | pose | 105 | 0.9333 | 0.9278 | 0.9329 | 0.9742 | 0.9635 | 0.8983 | 0.1017 | 0.0217 | 53/6/1/45 | 15/3/fixed |
| 4 | seg_sin_relacion | 105 | 0.9429 | 0.9333 | 0.9417 | 0.9856 | 0.9857 | 0.9661 | 0.0339 | 0.0870 | 57/2/4/42 | 15/3/fixed |
| 4 | late_sin_relacion | 105 | 0.9619 | 0.9574 | 0.9615 | 0.9934 | 0.9928 | 0.9492 | 0.0508 | 0.0217 | 56/3/1/45 | 15/3/fixed |
| 5 | pose | 104 | 0.8558 | 0.8315 | 0.8527 | 0.9100 | 0.9221 | 0.8966 | 0.1034 | 0.1957 | 52/6/9/37 | 20/28/fixed |
| 5 | seg_sin_relacion | 104 | 0.9231 | 0.9091 | 0.9212 | 0.9516 | 0.9588 | 0.9655 | 0.0345 | 0.1304 | 56/2/6/40 | 20/28/fixed |
| 5 | late_sin_relacion | 104 | 0.9135 | 0.8989 | 0.9116 | 0.9363 | 0.9483 | 0.9483 | 0.0517 | 0.1304 | 55/3/6/40 | 20/28/fixed |

Los CSV por fold incluyen Precision, Recall, Balanced Accuracy y sensibilidad de válidos; FPR=FP/(FP+TN), FNR=FN/(FN+TP), positivo=1 válido, FP=inválido aprobado. Todos los folds tuvieron ambas clases; si faltara alguna, ROC-AUC/AP se declararía N/D. Las matrices de confusión están también guardadas como `fold_XX/confusion_<model>.npy`.

## Media ± desviación poblacional de cinco folds (NO pooled)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | Especificidad inválidos | FP por fold |
|---|---:|---:|---:|---:|---:|---:|---|
| pose | 0.9121±0.0303 | 0.8990±0.0369 | 0.9105±0.0312 | 0.9104±0.0329 | 0.9532±0.0231 | 0.9249±0.0277 | 3/2/5/6/6 |
| seg_sin_relacion | 0.9199±0.0260 | 0.9067±0.0314 | 0.9182±0.0268 | 0.9164±0.0271 | 0.9489±0.0311 | 0.9455±0.0327 | 2/3/7/2/2 |
| late_sin_relacion | 0.9313±0.0292 | 0.9199±0.0358 | 0.9298±0.0302 | 0.9288±0.0324 | 0.9650±0.0239 | 0.9489±0.0283 | 2/1/6/3/3 |

`five_fold_stability.csv` también ofrece AP y FPR media±desviación. Estas medias de folds **no** se confunden con las métricas pooled siguientes.

## Pooled OOF (únicamente Outer Validation de cada fold, un resultado por video)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.9122 | 0.9000 | 0.9109 | 0.9105 | 0.9485 | 0.9414 | 0.9249 | 0.0751 | 0.1039 | 271/22/24/207 |
| seg_sin_relacion | 0.9198 | 0.9071 | 0.9183 | 0.9164 | 0.9425 | 0.9157 | 0.9454 | 0.0546 | 0.1126 | 277/16/26/205 |
| late_sin_relacion | 0.9313 | 0.9211 | 0.9301 | 0.9289 | 0.9575 | 0.9513 | 0.9488 | 0.0512 | 0.0909 | 278/15/21/210 |

`oof_predictions.csv` y `oof_paired_by_video.csv` permiten reproducir estas métricas sin mezclar encoders de distintos folds. Los F1 y AUC pooled no son la media de métricas de fold. Pooled FP = [('pose', 22), ('seg_sin_relacion', 16), ('late_sin_relacion', 15)].

## Errores pareados en OOF

- pose vs seg_sin_relacion: compartidos **24**, exclusivos pose **22** y seg_sin_relacion **18**, desacuerdos **40**. Por fold: [(1, 1, 5, 4), (2, 7, 2, 5), (3, 9, 0, 2), (4, 1, 6, 5), (5, 6, 9, 2)]. FP pose: 22; FP seg_sin_relacion: 16.
- seg_sin_relacion vs late_sin_relacion: compartidos **29**, exclusivos seg_sin_relacion **13** y late_sin_relacion **7**, desacuerdos **20**. Por fold: [(1, 2, 3, 1), (2, 9, 3, 1), (3, 10, 1, 0), (4, 1, 5, 3), (5, 7, 1, 2)]. FP seg_sin_relacion: 16; FP late_sin_relacion: 15.
- pose vs late_sin_relacion: compartidos **31**, exclusivos pose **15** y late_sin_relacion **5**, desacuerdos **20**. Por fold: [(1, 2, 4, 1), (2, 8, 1, 2), (3, 9, 0, 1), (4, 4, 3, 0), (5, 8, 7, 1)]. FP pose: 22; FP late_sin_relacion: 15.

Los IDs compartidos y exclusivos, FP/FN y discrepancias por fold constan en `oof_paired_comparisons.json`; no se aplicaron reglas por video ni se calculó un oráculo. Para RIGHT el promedio tardío 50/50 es comparador **secundario**, sin cabeza, pesos ni umbral ajustados; incluso si varía favorablemente entre folds, el candidato principal sigue siendo SegCNN SIN. 

## Interpretación para la tesis

La arquitectura, el esquema de features y los hiperparámetros fueron escogidos antes usando desarrollo histórico. Esta CV es un análisis **retrospectivo condicionado por esas decisiones**, no una selección anidada totalmente independiente. Los modelos Test finales preseleccionados siguen intactos, y Test se reserva para encargo posterior. Además de falta de grupos físicos verificados, la rama SegCNN lateral depende parcialmente de Pose YOLO y de proyecciones 2D barra–torso; el entrenamiento YOLO previo no demuestra separación por atletas/eventos respecto a este conjunto. Estas limitaciones impiden interpretar un promedio CV alto como prueba externa de juicio de competición.
