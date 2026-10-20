# Evaluación final BP_RIGHT Test V1

**RIGHT_FINAL_TEST_COMPLETE**. Protocolo congelado 2026-10-15T14:54:53.303363+00:00; SHA-256 `d2fe98221cf717b69718e7d2a3a77e81d9d09856e43a46cbce3bf0bd1d332498`.

Evaluación única: 93 videos, 41 válidos y 52 inválidos. Una predicción por levantamiento. Principal invariable: **seg_sin_relacion**. Umbral 0.5; promedio tardío 50/50. Sin entrenamiento ni ajustes.

## Fuentes y reproducción

Ver `source_manifest.json`, `duplicate_audit.json`, `validation_reproducibility.json` e `integrity_verification.json`. Coincidencia de IDs, etiquetas, decisiones y métricas Validation comprobada, probabilidades dentro de tolerancias históricas. Los pesos originales no fueron reemplazados por los de CV.

## Tabla A — Validation original

| model | n | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | average_precision | sensitivity | specificity | fpr | fnr | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 93 | 0.8817 | 0.8750 | 0.8537 | 0.8642 | 0.8797 | 0.8788 | 0.9250 | 0.9134 | 0.8537 | 0.9038 | 0.0962 | 0.1463 | 47 | 5 | 6 | 35 |
| seg_sin_relacion | 93 | 0.9247 | 0.9250 | 0.9024 | 0.9136 | 0.9235 | 0.9224 | 0.9402 | 0.9427 | 0.9024 | 0.9423 | 0.0577 | 0.0976 | 49 | 3 | 4 | 37 |
| late_sin_relacion | 93 | 0.9032 | 0.8810 | 0.9024 | 0.8916 | 0.9021 | 0.9031 | 0.9353 | 0.9398 | 0.9024 | 0.9038 | 0.0962 | 0.0976 | 47 | 5 | 4 | 37 |

## Tabla B — CV 5-Fold (media ± desviación poblacional)

| model | n_folds | accuracy_mean | accuracy_std | f1_mean | f1_std | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | balanced_accuracy_std | roc_auc_mean | roc_auc_std | specificity_invalid_mean | specificity_invalid_std | FPR_mean | FPR_std | pr_auc_mean | pr_auc_std | false_positives_by_fold | false_negatives_by_fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 5 | 0.9121 | 0.0303 | 0.8990 | 0.0369 | 0.9105 | 0.0312 | 0.9104 | 0.0329 | 0.9532 | 0.0231 | 0.9249 | 0.0277 | 0.0751 | 0.0277 | 0.9493 | 0.0146 | 3/2/5/6/6 | 3/7/4/1/9 |
| seg_sin_relacion | 5 | 0.9199 | 0.0260 | 0.9067 | 0.0314 | 0.9182 | 0.0268 | 0.9164 | 0.0271 | 0.9489 | 0.0311 | 0.9455 | 0.0327 | 0.0545 | 0.0327 | 0.9400 | 0.0358 | 2/3/7/2/2 | 3/9/4/4/6 |
| late_sin_relacion | 5 | 0.9313 | 0.0292 | 0.9199 | 0.0358 | 0.9298 | 0.0302 | 0.9288 | 0.0324 | 0.9650 | 0.0239 | 0.9489 | 0.0283 | 0.0511 | 0.0283 | 0.9659 | 0.0224 | 2/1/6/3/3 | 1/9/4/1/6 |

### Pooled OOF

| model | n | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | pr_auc | sensitivity_valid | specificity_invalid | FPR | FNR | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 524 | 0.9122 | 0.9039 | 0.8961 | 0.9000 | 0.9109 | 0.9105 | 0.9485 | 0.9414 | 0.8961 | 0.9249 | 0.0751 | 0.1039 | 271 | 22 | 24 | 207 |
| seg_sin_relacion | 524 | 0.9198 | 0.9276 | 0.8874 | 0.9071 | 0.9183 | 0.9164 | 0.9425 | 0.9157 | 0.8874 | 0.9454 | 0.0546 | 0.1126 | 277 | 16 | 26 | 205 |
| late_sin_relacion | 524 | 0.9313 | 0.9333 | 0.9091 | 0.9211 | 0.9301 | 0.9289 | 0.9575 | 0.9513 | 0.9091 | 0.9488 | 0.0512 | 0.0909 | 278 | 15 | 21 | 210 |

### Por fold

| fold | model | n | best_pose_epoch | best_seg_epoch | best_head_epoch | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | pr_auc | sensitivity_valid | specificity_invalid | FPR | FNR | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pose | 105 | 17 | 56 | fixed | 0.9429 | 0.9362 | 0.9362 | 0.9362 | 0.9422 | 0.9422 | 0.9710 | 0.9591 | 0.9362 | 0.9483 | 0.0517 | 0.0638 | 55 | 3 | 3 | 44 |
| 1 | seg_sin_relacion | 105 | 17 | 56 | fixed | 0.9524 | 0.9565 | 0.9362 | 0.9462 | 0.9518 | 0.9508 | 0.9791 | 0.9516 | 0.9362 | 0.9655 | 0.0345 | 0.0638 | 56 | 2 | 3 | 44 |
| 1 | late_sin_relacion | 105 | 17 | 56 | fixed | 0.9714 | 0.9583 | 0.9787 | 0.9684 | 0.9712 | 0.9721 | 0.9938 | 0.9934 | 0.9787 | 0.9655 | 0.0345 | 0.0213 | 56 | 2 | 1 | 46 |
| 2 | pose | 105 | 33 | 15 | fixed | 0.9143 | 0.9512 | 0.8478 | 0.8966 | 0.9117 | 0.9070 | 0.9598 | 0.9547 | 0.8478 | 0.9661 | 0.0339 | 0.1522 | 57 | 2 | 7 | 39 |
| 2 | seg_sin_relacion | 105 | 33 | 15 | fixed | 0.8857 | 0.9250 | 0.8043 | 0.8605 | 0.8818 | 0.8768 | 0.9057 | 0.9233 | 0.8043 | 0.9492 | 0.0508 | 0.1957 | 56 | 3 | 9 | 37 |
| 2 | late_sin_relacion | 105 | 33 | 15 | fixed | 0.9048 | 0.9737 | 0.8043 | 0.8810 | 0.9008 | 0.8937 | 0.9506 | 0.9515 | 0.8043 | 0.9831 | 0.0169 | 0.1957 | 58 | 1 | 9 | 37 |
| 3 | pose | 105 | 35 | 47 | fixed | 0.9143 | 0.8936 | 0.9130 | 0.9032 | 0.9132 | 0.9141 | 0.9510 | 0.9472 | 0.9130 | 0.9153 | 0.0847 | 0.0870 | 54 | 5 | 4 | 42 |
| 3 | seg_sin_relacion | 105 | 35 | 47 | fixed | 0.8952 | 0.8571 | 0.9130 | 0.8842 | 0.8943 | 0.8972 | 0.9223 | 0.8805 | 0.9130 | 0.8814 | 0.1186 | 0.0870 | 52 | 7 | 4 | 42 |
| 3 | late_sin_relacion | 105 | 35 | 47 | fixed | 0.9048 | 0.8750 | 0.9130 | 0.8936 | 0.9037 | 0.9057 | 0.9510 | 0.9433 | 0.9130 | 0.8983 | 0.1017 | 0.0870 | 53 | 6 | 4 | 42 |
| 4 | pose | 105 | 15 | 3 | fixed | 0.9333 | 0.8824 | 0.9783 | 0.9278 | 0.9329 | 0.9383 | 0.9742 | 0.9635 | 0.9783 | 0.8983 | 0.1017 | 0.0217 | 53 | 6 | 1 | 45 |
| 4 | seg_sin_relacion | 105 | 15 | 3 | fixed | 0.9429 | 0.9545 | 0.9130 | 0.9333 | 0.9417 | 0.9396 | 0.9856 | 0.9857 | 0.9130 | 0.9661 | 0.0339 | 0.0870 | 57 | 2 | 4 | 42 |
| 4 | late_sin_relacion | 105 | 15 | 3 | fixed | 0.9619 | 0.9375 | 0.9783 | 0.9574 | 0.9615 | 0.9637 | 0.9934 | 0.9928 | 0.9783 | 0.9492 | 0.0508 | 0.0217 | 56 | 3 | 1 | 45 |
| 5 | pose | 104 | 20 | 28 | fixed | 0.8558 | 0.8605 | 0.8043 | 0.8315 | 0.8527 | 0.8504 | 0.9100 | 0.9221 | 0.8043 | 0.8966 | 0.1034 | 0.1957 | 52 | 6 | 9 | 37 |
| 5 | seg_sin_relacion | 104 | 20 | 28 | fixed | 0.9231 | 0.9524 | 0.8696 | 0.9091 | 0.9212 | 0.9175 | 0.9516 | 0.9588 | 0.8696 | 0.9655 | 0.0345 | 0.1304 | 56 | 2 | 6 | 40 |
| 5 | late_sin_relacion | 104 | 20 | 28 | fixed | 0.9135 | 0.9302 | 0.8696 | 0.8989 | 0.9116 | 0.9089 | 0.9363 | 0.9483 | 0.8696 | 0.9483 | 0.0517 | 0.1304 | 55 | 3 | 6 | 40 |

La fusión tardía permanece como comparador secundario.

## Tabla C — Test final

| model | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | average_precision | sensitivity | specificity | fpr | fnr | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 0.9247 | 0.8864 | 0.9512 | 0.9176 | 0.9242 | 0.9275 | 0.9559 | 0.9277 | 0.9512 | 0.9038 | 0.0962 | 0.0488 | 47 | 5 | 2 | 39 |
| seg_sin_relacion | 0.9355 | 0.9730 | 0.8780 | 0.9231 | 0.9338 | 0.9294 | 0.9639 | 0.9681 | 0.8780 | 0.9808 | 0.0192 | 0.1220 | 51 | 1 | 5 | 36 |
| late_sin_relacion | 0.9140 | 0.8837 | 0.9268 | 0.9048 | 0.9132 | 0.9153 | 0.9850 | 0.9830 | 0.9268 | 0.9038 | 0.0962 | 0.0732 | 47 | 5 | 3 | 38 |

AP corresponde a Average Precision, no al área trapezoidal de PR. Positivo=1 válido; **FP=inválido aprobado**, FN=válido rechazado.

## Diferencias Test menos Validation

| model | accuracy_test_minus_validation | f1_test_minus_validation | macro_f1_test_minus_validation | roc_auc_test_minus_validation | specificity_test_minus_validation | fpr_test_minus_validation |
|---|---|---|---|---|---|---|
| pose | 0.0430 | 0.0534 | 0.0445 | 0.0310 | -0.0000 | 0.0000 |
| seg_sin_relacion | 0.0108 | 0.0095 | 0.0103 | 0.0237 | 0.0385 | -0.0385 |
| late_sin_relacion | 0.0108 | 0.0132 | 0.0111 | 0.0497 | -0.0000 | 0.0000 |

CV usa nuevos pesos por fold y menos datos en cada inner Train; Test usa checkpoints originales seleccionados. El promedio CV no es rendimiento del mismo checkpoint ni debe coincidir necesariamente con Test.

## Errores pareados (B respecto a A)

| a | b | pearson | shared_errors | exclusive_a | exclusive_b | fp_corrected | fp_introduced | fn_corrected | fn_introduced | disagreements |
|---|---|---|---|---|---|---|---|---|---|---|
| pose | seg_sin_relacion | 0.8856 | 3 | 4 | 3 | 4 | 0 | 0 | 3 | 7 |
| pose | late_sin_relacion | 0.9728 | 7 | 0 | 1 | 0 | 0 | 0 | 1 | 1 |
| seg_sin_relacion | late_sin_relacion | 0.9691 | 4 | 2 | 4 | 0 | 4 | 2 | 0 | 6 |

IDs completos y correlaciones: `paired_analysis.json`; todos los errores: `errors_by_video.csv`; cercanía al umbral |p−0.5|≤0.05: `near_threshold.csv`.

## Incertidumbre exploratoria 95%

| model | metric | lower | upper |
|---|---|---|---|
| pose | accuracy | 0.8710 | 0.9680 |
| pose | f1 | 0.8500 | 0.9714 |
| pose | macro_f1 | 0.8675 | 0.9679 |
| pose | balanced_accuracy | 0.8718 | 0.9750 |
| pose | specificity | 0.8182 | 0.9792 |
| pose | roc_auc | 0.9098 | 0.9912 |
| pose | fpr_wilson | 0.0418 | 0.2061 |
| pose | specificity_wilson | 0.7939 | 0.9582 |
| seg_sin_relacion | accuracy | 0.8817 | 0.9785 |
| seg_sin_relacion | f1 | 0.8525 | 0.9756 |
| seg_sin_relacion | macro_f1 | 0.8781 | 0.9782 |
| seg_sin_relacion | balanced_accuracy | 0.8711 | 0.9778 |
| seg_sin_relacion | specificity | 0.9362 | 1.0000 |
| seg_sin_relacion | roc_auc | 0.9208 | 0.9945 |
| seg_sin_relacion | fpr_wilson | 0.0034 | 0.1012 |
| seg_sin_relacion | specificity_wilson | 0.8988 | 0.9966 |
| late_sin_relacion | accuracy | 0.8495 | 0.9677 |
| late_sin_relacion | f1 | 0.8308 | 0.9630 |
| late_sin_relacion | macro_f1 | 0.8494 | 0.9672 |
| late_sin_relacion | balanced_accuracy | 0.8564 | 0.9682 |
| late_sin_relacion | specificity | 0.8182 | 0.9792 |
| late_sin_relacion | roc_auc | 0.9644 | 0.9972 |
| late_sin_relacion | fpr_wilson | 0.0418 | 0.2061 |
| late_sin_relacion | specificity_wilson | 0.7939 | 0.9582 |

Bootstrap pareado ordinario por video, 10000 réplicas aceptadas de 10000, seed 20261015; 0 descartadas por una sola clase. Percentiles 2.5/97.5; mismos índices para todos los modelos. Wilson para FPR/especificidad. Sin agrupamiento acreditado ni corrección por multiplicidad; no se utiliza para seleccionar modelos.

### seg_sin_relacion minus pose
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | 0.0108 | -0.0430 | 0.0645 |
| macro_f1 | 0.0096 | -0.0458 | 0.0652 |
| specificity | 0.0769 | 0.0179 | 0.1569 |

### late_sin_relacion minus pose
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | -0.0108 | -0.0323 | 0.0000 |
| macro_f1 | -0.0110 | -0.0348 | 0.0000 |
| specificity | 0.0000 | 0.0000 | 0.0000 |

### late_sin_relacion minus seg_sin_relacion
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | -0.0215 | -0.0753 | 0.0323 |
| macro_f1 | -0.0206 | -0.0750 | 0.0327 |
| specificity | -0.0769 | -0.1569 | -0.0179 |

## Interpretación experimental

El principal obtiene Accuracy=0.9355, F1=0.9231, Macro-F1=0.9338, AUC=0.9639. Aprueba incorrectamente 1 de 52 inválidos (FPR=0.0192); esta tasa debe figurar explícitamente como limitación arbitral.

Respecto a Validation, ΔAccuracy=+0.0108, ΔF1=+0.0095, ΔMacro-F1=+0.0103, ΔAUC=+0.0237. Una caída es compatible con optimismo de selección/sobreajuste y variabilidad de muestra; no identifica por sí sola la causa.

Principal frente a pose: ΔAccuracy=+0.0108, ΔMacro-F1=+0.0096, ΔFP=-4. Son diferencias descriptivas; revisar sus IC pareados antes de afirmar superioridad.

Principal frente a late_sin_relacion: ΔAccuracy=+0.0215, ΔMacro-F1=+0.0206, ΔFP=-4. Son diferencias descriptivas; revisar sus IC pareados antes de afirmar superioridad.

No se cambia el candidato según Test. La falta de ventaja de una fusión no acredita equivalencia ni uso autónomo en competición.

## Limitaciones metodológicas

- Test lateral expuesto históricamente a evaluaciones PoseCNN; no totalmente inédito para el proyecto.

- Independencia de atletas, sesiones y eventos físicos no acreditada; vistas no emparejadas por numeración.

- Segmentación parcialmente asistida por keypoints Pose: modalidades no independientes.

- Proyecciones barra–torso 2D no demuestran contacto físico.

- Validation reutilizado para seleccionar checkpoints, variantes y cabezas.

- CV retrospectiva de robustez, no selección completamente anidada previa a exposición Test.

- Independencia de datasets históricos YOLO respecto a evaluación no plenamente acreditada.

- IC exploratorios por video suponen independencia no acreditada; no corrigen selección ni comparaciones múltiples.

## Integridad y reproducibilidad

249 artefactos protegidos y 372 entradas verificadas antes/después, sin cambios. Ver también `protected_inputs_before.json` y `protected_inputs_after.json` para MP4/RAW/keypoints/features completos.

Código congelado: `training/bp_lateral_final_test_v1/run.py` y `analyze.py`. Comandos: `python -B -m training.bp_lateral_final_test_v1.run --view right --stage freeze`, luego `validation`, luego `test`; análisis `python -B -m training.bp_lateral_final_test_v1.analyze --view right`. El evaluador rechaza sobrescribir protocolos o repetir inferencias ya guardadas. Los artefactos existentes permiten recalcular métricas sin volver a ejecutar modelos.

![ROC](roc.png)

![Precision Recall](precision_recall.png)

![pose](confusion_pose.png)

![seg_sin_relacion](confusion_seg_sin_relacion.png)

![late_sin_relacion](confusion_late_sin_relacion.png)

## Respuestas para la tesis

1. **SegCNN SIN mantuvo su rendimiento de Validation:** Accuracy 0.9247→0.9355, F1 0.9136→0.9231 y Macro-F1 0.9235→0.9338. Especificidad 0.9423→0.9808. El recall disminuyó de 0.9024 a 0.8780: mejor control de inválidos no implica mejor sensibilidad a válidos.
2. Superó descriptivamente a PoseCNN en Accuracy (+0.0108) y F1 (+0.0054), con 1 FP frente a 5, pero 5 FN frente a 2. Corrigió cuatro FP (`bp_383`, `bp_443`, `bp_444`, `bp_445`) e introdujo tres FN (`bp_369`, `bp_389`, `bp_392`).
3. **La fusión tardía no mejoró las decisiones a umbral fijo:** Accuracy 0.9140/F1 0.9048, inferiores a ambas bases. Mejoró ROC-AUC (0.9850) y AP (0.9830); esta mejor ordenación no implica mejores decisiones a 0.5 ni autoriza ajustar el umbral.
4. SegCNN aprobó incorrectamente solamente `bp_385`: 1/52 inválidos, FPR 1.92%. Wilson 95% [0.34%,10.12%] muestra incertidumbre importante. Pose y Seg compartieron errores `bp_054`, `bp_385`, `bp_487`.
5. Frente a Seg, la fusión corrigió dos FN (`bp_389`, `bp_392`) pero introdujo cuatro FP (`bp_383`, `bp_443`, `bp_444`, `bp_445`). Frente a Pose conservó los siete errores e introdujo FN `bp_369`.
6. F1 Test principal 0.9231 frente a CV 0.9067 ± 0.0314 (pooled 0.9071); AUC 0.9639 frente a 0.9489 ± 0.0311. El nivel de rendimiento es compatible descriptivamente con CV, pero el orden de los modelos cambió: la ventaja media CV de fusión tardía no persistió en Test a 0.5. Los checkpoints son diferentes.
7. IC95% Accuracy principal [0.8817,0.9785], F1 [0.8525,0.9756], especificidad [0.9362,1.0000]. ΔAccuracy Seg−Pose +0.0108, IC [−0.0430,0.0645]: superioridad global no concluyente. Δespecificidad Seg−Pose +0.0769, IC [0.0179,0.1569]: evidencia exploratoria favorable al control de FP, condicionada por independencia no acreditada y pocos casos discordantes.
8. La arquitectura individual es una solución parsimoniosa respaldada por esta evaluación y no necesita fusión para alcanzar el resultado observado. Esto no prueba suficiencia universal, equivalencia estadística ni autonomía arbitral. SegCNN SIN permanece como candidato principal sin reajustes.

### Integridad y ejecución

La etapa `protect` debe ejecutarse entre `freeze` y `validation`. Se comprobaron 2579 archivos de insumos (MP4, RAW, keypoints y features) antes/después, además de 249 artefactos de modelos, configuración, código y CV. Los hashes binarios de Test se calcularon después del congelamiento; los flags `test_opened=false` de reproducción significan ausencia de carga/decodificación de secuencias Test. Hubo una única inferencia Test y se mantuvo excluido `bp_354`.
