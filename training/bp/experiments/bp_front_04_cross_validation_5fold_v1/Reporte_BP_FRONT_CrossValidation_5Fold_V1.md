# BP_FRONT — Validación cruzada estratificada 5-Fold V1



**Estado: FRONT_5FOLD_CV_COMPLETE.** 15 entrenamientos nuevos completos; 561 videos development; 2.805 predicciones OOF (561 por configuración).

## Protocolo y alcance

Cohorte obligatoria filtrada a Train (462) + Validation (99): 339 válidos y 222 inválidos. Exclusiones bp_349 y bp_352 preservadas. No se abrieron secuencias ni predicciones Test. No se ejecutó YOLO ni se reutilizaron pesos históricos.

StratifiedKFold(5, shuffle=True, random_state=42), orden por video_id. Sin group_id acreditado: no se inventaron correspondencias físicas. Auditoría de duplicados exactos del contenido de keypoints development: ninguno. Esta comprobación no equivale a acreditar independencia de atletas/sesiones.

Dentro de cada outer train: StratifiedShuffleSplit(test_size=0.15, random_state=42). Escalado/imputación ajustados solo con inner train; early stopping y scheduler solo con inner validation. Outer validation se usa exclusivamente tras seleccionar el checkpoint. Unidad: levantamiento completo.

Semillas fijas: Pose=42, FULL=1337, SIN=42. Pose: Adam, BCELoss, FP32, batch 16; Seg: AdamW, BCEWithLogitsLoss, AMP CUDA, batch 32. Todas: lr=0.001, weight_decay=1e-4, máximo 200 épocas, patience 15; scheduler factor 0.5/patience 5. Determinismo estricto activado. Sin búsqueda de semillas ni hiperparámetros.

PoseCNN: (3,170,6), 118.529 parámetros; normalización por hombros, ausencia neutral/confianza cero, remuestreo entre vecinos observados y augmentation histórico con máscara (original + 3). Scaler por timestep×keypoint×canal, fit original antes del aumento. Seg FULL: (79,170), 56.449 parámetros; SIN: (65,170), 54.209 parámetros. Se congelaron los índices históricos (incluida exclusión de profile_ambiguous); no se recalcularon canales constantes para seleccionar features. Remuestreo V1 sin cruzar ausencias/cambios de procedencia o identidad. Mediana/media/std solo sobre observaciones de inner train. Máscaras/procedencias conservadas; no se aumentó SegCNN.

Fusiones: 0.5×Pose + 0.5×Seg del mismo fold. Umbral fijo 0.5. Clase positiva válido; FP = inválido aprobado incorrectamente.

## Distribución de particiones

| fold | role | n | valid | invalid |
| --- | --- | --- | --- | --- |
| 1 | inner_train | 380 | 230 | 150 |
| 1 | inner_validation | 68 | 41 | 27 |
| 1 | outer_validation | 113 | 68 | 45 |
| 2 | inner_train | 381 | 230 | 151 |
| 2 | inner_validation | 68 | 41 | 27 |
| 2 | outer_validation | 112 | 68 | 44 |
| 3 | inner_train | 381 | 230 | 151 |
| 3 | inner_validation | 68 | 41 | 27 |
| 3 | outer_validation | 112 | 68 | 44 |
| 4 | inner_train | 381 | 230 | 151 |
| 4 | inner_validation | 68 | 41 | 27 |
| 4 | outer_validation | 112 | 68 | 44 |
| 5 | inner_train | 381 | 231 | 150 |
| 5 | inner_validation | 68 | 41 | 27 |
| 5 | outer_validation | 112 | 67 | 45 |

## Media ± desviación entre folds

Desviación muestral (ddof=1), no intervalo de confianza. Folds con conjuntos de entrenamiento solapados; no son cinco experimentos independientes.

| model | accuracy | f1 | macro_f1 | roc_auc | specificity |
| --- | --- | --- | --- | --- | --- |
| posecnn | 0.8485 ± 0.0221 | 0.8761 ± 0.0193 | 0.8398 ± 0.0236 | 0.9282 ± 0.0185 | 0.7881 ± 0.0801 |
| segcnn_full | 0.8663 ± 0.0394 | 0.8919 ± 0.0275 | 0.8576 ± 0.0460 | 0.9264 ± 0.0553 | 0.8058 ± 0.1128 |
| segcnn_sin_relacion | 0.8716 ± 0.0628 | 0.8995 ± 0.0447 | 0.8601 ± 0.0734 | 0.9404 ± 0.0423 | 0.7692 ± 0.1406 |
| late_fusion_full | 0.8841 ± 0.0255 | 0.9066 ± 0.0196 | 0.8766 ± 0.0285 | 0.9495 ± 0.0203 | 0.8149 ± 0.0717 |
| late_fusion_sin_relacion | 0.8770 ± 0.0145 | 0.9002 ± 0.0104 | 0.8695 ± 0.0173 | 0.9534 ± 0.0141 | 0.8149 ± 0.0750 |

Todas las métricas, incluida Precision, Recall, Balanced Accuracy, AP, sensibilidad, FPR/FNR y conteos, están en cv_metrics_by_fold.csv y cv_metrics_summary.csv.

## Falsos positivos

| model | 1 | 2 | 3 | 4 | 5 | total |
| --- | --- | --- | --- | --- | --- | --- |
| posecnn | 7 | 5 | 11 | 14 | 10 | 47 |
| segcnn_full | 7 | 4 | 16 | 11 | 5 | 43 |
| segcnn_sin_relacion | 5 | 11 | 20 | 10 | 5 | 51 |
| late_fusion_full | 6 | 5 | 12 | 11 | 7 | 41 |
| late_fusion_sin_relacion | 8 | 5 | 12 | 11 | 5 | 41 |

## Pooled OOF

Una única predicción externa por video y modelo. AP = average_precision_score (integración escalonada precision–recall), no área trapezoidal. El pooled AUC compara scores de cinco redes diferentes; no debe confundirse con la media de AUC por fold.

| model | n | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | average_precision | sensitivity | specificity | fpr | fnr | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| posecnn | 561 | 0.8485 | 0.8649 | 0.8879 | 0.8763 | 0.8404 | 0.8381 | 0.9250 | 0.9424 | 0.8879 | 0.7883 | 0.2117 | 0.1121 | 175 | 47 | 38 | 301 |
| segcnn_full | 561 | 0.8663 | 0.8771 | 0.9056 | 0.8911 | 0.8590 | 0.8560 | 0.9271 | 0.9262 | 0.9056 | 0.8063 | 0.1937 | 0.0944 | 179 | 43 | 32 | 307 |
| segcnn_sin_relacion | 561 | 0.8717 | 0.8618 | 0.9381 | 0.8983 | 0.8622 | 0.8542 | 0.9421 | 0.9570 | 0.9381 | 0.7703 | 0.2297 | 0.0619 | 171 | 51 | 21 | 318 |
| late_fusion_full | 561 | 0.8841 | 0.8848 | 0.9292 | 0.9065 | 0.8771 | 0.8723 | 0.9471 | 0.9578 | 0.9292 | 0.8153 | 0.1847 | 0.0708 | 181 | 41 | 24 | 315 |
| late_fusion_sin_relacion | 561 | 0.8770 | 0.8835 | 0.9174 | 0.9001 | 0.8700 | 0.8664 | 0.9512 | 0.9610 | 0.9174 | 0.8153 | 0.1847 | 0.0826 | 181 | 41 | 28 | 311 |

![Confusiones](figures/pooled_confusion.png)

![ROC y PR](figures/pooled_roc_pr.png)

## Estabilidad y complementariedad

Fusión SIN presenta la menor desviación de Accuracy, F1, Macro-F1 y ROC-AUC. No es universal: otras métricas pueden ordenar la estabilidad de otra manera. Fusión FULL logra mayor Accuracy/F1/Macro-F1 promedio; SIN logra mayor AUC promedio. Estas observaciones no alteran el candidato preseleccionado: **Pose + SIN**.

FULL mejora descriptivamente frente a Pose y Seg FULL. SIN mejora frente a Pose; frente a Seg SIN, su ganancia media en F1 es pequeña, con menor variabilidad y menos FP pooled (41 frente a 51), a costa de más FN (28 frente a 21). No se declara superioridad estadística ni se ajusta ningún umbral.

En la tabla, corregido/introducido se refiere a pasar de A a B. Errores exclusivos A son fallos de A acertados por B; exclusivos B, lo contrario. No se presenta rendimiento de oráculo.

| model_a | model_b | shared_errors | exclusive_a | exclusive_b | fp_corrected | fp_introduced | fn_corrected | fn_introduced | probability_pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| posecnn | segcnn_full | 40 | 45 | 35 | 21 | 17 | 24 | 18 | 0.8221 |
| posecnn | segcnn_sin_relacion | 37 | 48 | 35 | 19 | 23 | 29 | 12 | 0.8110 |
| posecnn | late_fusion_full | 52 | 33 | 13 | 13 | 7 | 20 | 6 | 0.9523 |
| posecnn | late_fusion_sin_relacion | 62 | 23 | 7 | 10 | 4 | 13 | 3 | 0.9565 |
| segcnn_full | late_fusion_full | 53 | 22 | 12 | 10 | 8 | 12 | 4 | 0.9566 |
| segcnn_sin_relacion | late_fusion_sin_relacion | 44 | 28 | 25 | 19 | 9 | 9 | 16 | 0.9464 |

IDs completos por categoría: oof/paired_error_ids.csv. Errores individuales: oof/errors_by_video.csv.

![Variabilidad F1](figures/folds_f1.png)

![Variabilidad Macro-F1](figures/folds_macro_f1.png)

![Variabilidad AUC](figures/folds_roc_auc.png)

## Comparación documental A/B/C

| model | evaluation | n | accuracy | f1 | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| posecnn | A: Validation original | 99 | 0.8990 | 0.9219 | 0.8895 | 0.9641 |
| posecnn | B: CV media externa | 561 | 0.8485 | 0.8761 | 0.8398 | 0.9282 |
| posecnn | C: Test histórico (fuente ausente) | 100 | N/D | N/D | N/D | N/D |
| segcnn_full | A: Validation original | 99 | 0.8687 | 0.8943 | 0.8605 | 0.9500 |
| segcnn_full | B: CV media externa | 561 | 0.8663 | 0.8919 | 0.8576 | 0.9264 |
| segcnn_full | C: Test histórico (fuente ausente) | 100 | N/D | N/D | N/D | N/D |
| segcnn_sin_relacion | A: Validation original | 99 | 0.9192 | 0.9355 | 0.9137 | 0.9462 |
| segcnn_sin_relacion | B: CV media externa | 561 | 0.8716 | 0.8995 | 0.8601 | 0.9404 |
| segcnn_sin_relacion | C: Test histórico (fuente ausente) | 100 | N/D | N/D | N/D | N/D |
| late_fusion_full | A: Validation original | 99 | 0.9192 | 0.9375 | 0.9116 | 0.9679 |
| late_fusion_full | B: CV media externa | 561 | 0.8841 | 0.9066 | 0.8766 | 0.9495 |
| late_fusion_full | C: Test histórico (fuente ausente) | 100 | N/D | N/D | N/D | N/D |
| late_fusion_sin_relacion | A: Validation original | 99 | 0.9293 | 0.9449 | 0.9231 | 0.9594 |
| late_fusion_sin_relacion | B: CV media externa | 561 | 0.8770 | 0.9002 | 0.8695 | 0.9534 |
| late_fusion_sin_relacion | C: Test histórico (fuente ausente) | 100 | N/D | N/D | N/D | N/D |

**Limitación documental:** no existe el reporte autorizado training/bp_front_final_test_v1/Reporte_Final_BP_FRONT_PoseCNN_SegCNN_Fusion_Test.md. Se deja Test como N/D; no se sustituyó por otros archivos Test, por predicciones individuales ni por números del prompt. N=100 es contexto del protocolo solicitado, no un conteo nuevo de Test.

Validation original usa checkpoints seleccionados previamente y 99 videos; CV usa 15 redes nuevas, entrenamiento interno de 380–381 videos y evaluación externa de 112–113. Test histórico corresponde al protocolo anterior. Las diferencias pueden deberse al tamaño de entrenamiento, particiones, inicializaciones independientes, variabilidad de muestras y validación interna/externa. No son el mismo checkpoint ni la misma evaluación.

## Integridad y reproducibilidad

Se recalcularon independientemente todas las métricas por fold/pooled, fórmulas de fusión, pertenencia de OOF, ausencia de solapamientos y los 15 scalers exclusivamente con inner train. Checkpoints CV seleccionados: 15. Hashes de fuentes históricas, secuencias development y manifiestos sin cambios. Ningún checkpoint histórico se cargó para inferencia CV. Los archivos Test no se abrieron; por esa misma restricción no se calculó un nuevo hash de sus secuencias. No se acredita mediante esta ejecución una auditoría universal de archivos fuera de las fuentes registradas.

Incidencia operativa: timeout del primer comando tras completar folds 1–3 y comenzar Pose fold 4. Se preservó el intento incompleto en posecnn/fold_4_interrupted_timeout; se reinició determinísticamente con idéntica semilla/partición (las pérdidas registradas de épocas 1/10/20/30 coinciden). Hubo también un error de lectura de IDs object del cache local al reanudar, corregido sin cambiar datos ni entrenamiento. 15 ejecuciones completas + 1 intento interrumpido; el intento interrumpido no produjo OOF. Cambios de runner documentados en config/resume_revision.json; logs en execution.log.

Artefactos: config/source_hashes_before.json, config/protocol.json, config/frozen_channels.json, fold_assignments.csv, folds/fold_*_split.csv, scalers/checkpoints/curvas por fold, cv_*csv, cv_integrity_verification.json. El cache conserva identidad, T original, valores, máscaras y procedencias remuestreadas; los NPZ originales permanecen intactos.

## Limitaciones metodológicas y conclusiones para la tesis

1. CV retrospectiva: arquitecturas definidas usando development; Test ya conocido. Es una comprobación complementaria de robustez, no selección anidada completa anterior a Test. No restaura el carácter ciego ni elimina la exposición previa a development.

2. Independencia física no acreditada por atleta/sesión/evento: estratificación por video y deduplicación exacta no resuelven esa limitación.

3. SegCNN está parcialmente asistida por Pose durante la extracción geométrica; las modalidades no son totalmente independientes.

4. Geometría 2D: relaciones barra–silueta no demuestran contacto barra-pecho.

5. Independencia de entrenamiento YOLO frente a evaluación posterior no plenamente acreditada.

6. Resultados descriptivos de cinco folds con semillas fijadas por arquitectura; no separan exhaustivamente variabilidad de partición e inicialización. Tampoco permiten atribuir toda diferencia FULL/SIN exclusivamente a los canales, pues las semillas congeladas difieren.

Conclusión: la fusión tardía reduce la variabilidad principal y muestra complementariedad útil, con mejoras descriptivas sobre Pose. El control de FP sigue siendo relevante: ambas fusiones aprueban erróneamente 41/222 inválidos OOF. Pose + SIN conserva su condición de candidato histórico; no se seleccionan nuevos modelos ni se reentrenan modelos finales.
