# Validación cruzada estratificada BP_LEFT — cinco folds sobre development

**CV_RETROSPECTIVE_VALIDATION_COMPLETE_LEFT**. Fuente: Train+Validation históricos = **513 videos**; Test reservado (91 IDs) **no se abrió ni evaluó**. Variante SegCNN fija: `full`; input C=138, según contrato histórico. Preselección previa para Test: **Fusión embeddings Pose+Seg FULL seed1337**, checkpoint protegido SHA-256 `1467c5b11df73d76d773ff8f8abc77b9ffdf2d0e5853957f3c4e1cf03cbaa9ad`. Esta CV NO cambió la preselección.

## Identidad, folds y ausencia de leakage

- Desarrollo procedente exclusivamente de 422 Train + 91 Validation originales; Los duplicados conocidos LEFT quedan fuera de development. La auditoría de hashes MP4 en development no encontró ningún par idéntico restante. No existen grupos físicos acreditados en BP lateral: se utilizó **StratifiedKFold(n_splits=5, shuffle=True, random_state=42) por video**, no mapeos por numeración. No puede afirmarse independencia de atleta, sesión o levantamiento físico; `fold_assignments.csv` y `fold_class_distributions.csv` conservan IDs/etiquetas y roles.
- Outer evaluation exclusivamente para el cálculo externo. En cada Outer Train se separó inner Validation ≈15 % con estratificación y seed 42+fold, común a PoseCNN/SegCNN/cabeza. Los checkpoints y scheduler se seleccionaron **solo con inner Validation**; no se seleccionaron seeds según Outer. Scaler Pose/Seg/embeddings fit **solo inner Train original**. Pose Train aumentado ×4 exclusivamente allí, sin flip/rotación. Todos los pesos, BatchNorm y estados de optimizer se inicializaron desde cero por fold. Los backbones de fusión se congelaron en `eval()`; embeddings inner Train son **in-sample**, embeddings Outer externos.
- Canales SegCNN idénticos al input congelado histórico (C=138); topología mantenida si alguna feature resulta constante/vacía en un inner Train, con política fija mediana=0 si vacía y std=1 si no hay dispersión. Ningún estadístico del Outer (ni de la unión de development) se ajustó. Insumos originales temporalmente remuestreados a 170 con las mismas reglas de ausencia/procedencia ya congeladas.
- `source_manifest.json` y `cv_integrity.json` contienen hashes de sources/checkpoints finales antes/después: intactos. Hashes individuales Train/Validation de NPY Pose y NPZ Seg, sin apertura de archivos Test. Archivos CV por fold (scalers, parámetros, curvas, checkpoints y probabilidades) están bajo `fold_01/` … `fold_05/`.

## Métricas por fold externo (una predicción por levantamiento)

| Fold | Modelo | N | Acc | F1 | Macro-F1 | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP | Mejor época Pose/Seg/Cabeza |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | pose | 103 | 0.8350 | 0.8172 | 0.8334 | 0.9528 | 0.9540 | 0.9057 | 0.0943 | 0.2400 | 48/5/12/38 | 20/15/3 |
| 1 | seg_full | 103 | 0.8932 | 0.8932 | 0.8932 | 0.9766 | 0.9763 | 0.8679 | 0.1321 | 0.0800 | 46/7/4/46 | 20/15/3 |
| 1 | embedding_full | 103 | 0.9223 | 0.9167 | 0.9220 | 0.9815 | 0.9825 | 0.9623 | 0.0377 | 0.1200 | 51/2/6/44 | 20/15/3 |
| 2 | pose | 103 | 0.8738 | 0.8660 | 0.8734 | 0.9649 | 0.9644 | 0.9057 | 0.0943 | 0.1600 | 48/5/8/42 | 18/10/3 |
| 2 | seg_full | 103 | 0.9320 | 0.9263 | 0.9316 | 0.9943 | 0.9943 | 0.9811 | 0.0189 | 0.1200 | 52/1/6/44 | 18/10/3 |
| 2 | embedding_full | 103 | 0.9320 | 0.9263 | 0.9316 | 0.9909 | 0.9910 | 0.9811 | 0.0189 | 0.1200 | 52/1/6/44 | 18/10/3 |
| 3 | pose | 103 | 0.8447 | 0.8367 | 0.8443 | 0.9189 | 0.9241 | 0.8679 | 0.1321 | 0.1800 | 46/7/9/41 | 17/23/12 |
| 3 | seg_full | 103 | 0.8932 | 0.8932 | 0.8932 | 0.9528 | 0.9627 | 0.8679 | 0.1321 | 0.0800 | 46/7/4/46 | 17/23/12 |
| 3 | embedding_full | 103 | 0.8932 | 0.8911 | 0.8932 | 0.9457 | 0.9464 | 0.8868 | 0.1132 | 0.1000 | 47/6/5/45 | 17/23/12 |
| 4 | pose | 102 | 0.8235 | 0.8125 | 0.8229 | 0.9288 | 0.9313 | 0.8491 | 0.1509 | 0.2041 | 45/8/10/39 | 12/4/3 |
| 4 | seg_full | 102 | 0.8922 | 0.8842 | 0.8916 | 0.9449 | 0.9308 | 0.9245 | 0.0755 | 0.1429 | 49/4/7/42 | 12/4/3 |
| 4 | embedding_full | 102 | 0.9118 | 0.9072 | 0.9116 | 0.9549 | 0.9509 | 0.9245 | 0.0755 | 0.1020 | 49/4/5/44 | 12/4/3 |
| 5 | pose | 102 | 0.8431 | 0.8367 | 0.8429 | 0.9365 | 0.9325 | 0.8491 | 0.1509 | 0.1633 | 45/8/8/41 | 19/38/4 |
| 5 | seg_full | 102 | 0.9118 | 0.9072 | 0.9116 | 0.9480 | 0.9501 | 0.9245 | 0.0755 | 0.1020 | 49/4/5/44 | 19/38/4 |
| 5 | embedding_full | 102 | 0.8922 | 0.8866 | 0.8919 | 0.9634 | 0.9649 | 0.9057 | 0.0943 | 0.1224 | 48/5/6/43 | 19/38/4 |

Los CSV por fold incluyen Precision, Recall, Balanced Accuracy y sensibilidad de válidos; FPR=FP/(FP+TN), FNR=FN/(FN+TP), positivo=1 válido, FP=inválido aprobado. Todos los folds tuvieron ambas clases; si faltara alguna, ROC-AUC/AP se declararía N/D. Las matrices de confusión están también guardadas como `fold_XX/confusion_<model>.npy`.

## Media ± desviación poblacional de cinco folds (NO pooled)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | Especificidad inválidos | FP por fold |
|---|---:|---:|---:|---:|---:|---:|---|
| pose | 0.8440±0.0167 | 0.8338±0.0189 | 0.8434±0.0168 | 0.8430±0.0168 | 0.9404±0.0166 | 0.8755±0.0256 | 5/5/7/8/8 |
| seg_full | 0.9045±0.0156 | 0.9008±0.0147 | 0.9042±0.0155 | 0.9041±0.0151 | 0.9633±0.0191 | 0.9132±0.0424 | 7/1/7/4/4 |
| embedding_full | 0.9103±0.0158 | 0.9056±0.0150 | 0.9100±0.0156 | 0.9096±0.0152 | 0.9673±0.0167 | 0.9321±0.0350 | 2/1/6/4/5 |

`five_fold_stability.csv` también ofrece AP y FPR media±desviación. Estas medias de folds **no** se confunden con las métricas pooled siguientes.

## Pooled OOF (únicamente Outer Validation de cada fold, un resultado por video)

| Modelo | Acc | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Spec inválidos | FPR | FNR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.8441 | 0.8340 | 0.8435 | 0.8430 | 0.9382 | 0.9358 | 0.8755 | 0.1245 | 0.1895 | 232/33/47/201 |
| seg_full | 0.9045 | 0.9006 | 0.9043 | 0.9042 | 0.9587 | 0.9611 | 0.9132 | 0.0868 | 0.1048 | 242/23/26/222 |
| embedding_full | 0.9103 | 0.9053 | 0.9101 | 0.9096 | 0.9625 | 0.9584 | 0.9321 | 0.0679 | 0.1129 | 247/18/28/220 |

`oof_predictions.csv` y `oof_paired_by_video.csv` permiten reproducir estas métricas sin mezclar encoders de distintos folds. Los F1 y AUC pooled no son la media de métricas de fold. Pooled FP = [('pose', 33), ('seg_full', 23), ('embedding_full', 18)].

## Errores pareados en OOF

- pose vs seg_full: compartidos **36**, exclusivos pose **44** y seg_full **13**, desacuerdos **57**. Por fold: [(1, 6, 11, 5), (2, 5, 8, 2), (3, 9, 7, 2), (4, 8, 10, 3), (5, 8, 8, 1)]. FP pose: 33; FP seg_full: 23.
- seg_full vs embedding_full: compartidos **40**, exclusivos seg_full **9** y embedding_full **6**, desacuerdos **15**. Por fold: [(1, 6, 5, 2), (2, 7, 0, 0), (3, 9, 2, 2), (4, 9, 2, 0), (5, 9, 0, 2)]. FP seg_full: 23; FP embedding_full: 18.
- pose vs embedding_full: compartidos **39**, exclusivos pose **41** y embedding_full **7**, desacuerdos **48**. Por fold: [(1, 8, 9, 0), (2, 5, 8, 2), (3, 8, 8, 3), (4, 8, 10, 1), (5, 10, 6, 1)]. FP pose: 33; FP embedding_full: 18.

Los IDs compartidos y exclusivos, FP/FN y discrepancias por fold constan en `oof_paired_comparisons.json`; no se aplicaron reglas por video ni se calculó un oráculo. Para LEFT la cabeza usa seed fijo 1337 y no se utilizaron pesos finales originales en ningún fold. 

## Interpretación para la tesis

La arquitectura, el esquema de features y los hiperparámetros fueron escogidos antes usando desarrollo histórico. Esta CV es un análisis **retrospectivo condicionado por esas decisiones**, no una selección anidada totalmente independiente. Los modelos Test finales preseleccionados siguen intactos, y Test se reserva para encargo posterior. Además de falta de grupos físicos verificados, la rama SegCNN lateral depende parcialmente de Pose YOLO y de proyecciones 2D barra–torso; el entrenamiento YOLO previo no demuestra separación por atletas/eventos respecto a este conjunto. Estas limitaciones impiden interpretar un promedio CV alto como prueba externa de juicio de competición.
