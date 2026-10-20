# Evaluación final BP_LEFT Test V1

**LEFT_FINAL_TEST_COMPLETE**. Protocolo congelado 2026-10-15T14:54:27.570893+00:00; SHA-256 `a0c6d2b52980769ea3f75b1f2a0dc99c0b3e3dc30f03991ae72a29697f6da036`.

Evaluación única: 91 videos, 44 válidos y 47 inválidos. Una predicción por levantamiento. Principal invariable: **embedding_full**. Umbral 0.5; promedio tardío 50/50. Sin entrenamiento ni ajustes.

## Fuentes y reproducción

Ver `source_manifest.json`, `duplicate_audit.json`, `validation_reproducibility.json` e `integrity_verification.json`. Coincidencia de IDs, etiquetas, decisiones y métricas Validation comprobada, probabilidades dentro de tolerancias históricas. Los pesos originales no fueron reemplazados por los de CV.

## Tabla A — Validation original

| model | n | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | average_precision | sensitivity | specificity | fpr | fnr | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 91 | 0.9011 | 0.8723 | 0.9318 | 0.9011 | 0.9011 | 0.9021 | 0.9831 | 0.9839 | 0.9318 | 0.8723 | 0.1277 | 0.0682 | 41 | 6 | 3 | 41 |
| seg_full | 91 | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9937 | 0.9943 | 0.9545 | 0.9362 | 0.0638 | 0.0455 | 44 | 3 | 2 | 42 |
| late_full | 91 | 0.9560 | 0.9545 | 0.9545 | 0.9545 | 0.9560 | 0.9560 | 0.9932 | 0.9933 | 0.9545 | 0.9574 | 0.0426 | 0.0455 | 45 | 2 | 2 | 42 |
| embedding_full | 91 | 0.9670 | 0.9767 | 0.9545 | 0.9655 | 0.9670 | 0.9666 | 0.9937 | 0.9941 | 0.9545 | 0.9787 | 0.0213 | 0.0455 | 46 | 1 | 2 | 42 |

## Tabla B — CV 5-Fold (media ± desviación poblacional)

| model | n_folds | accuracy_mean | accuracy_std | f1_mean | f1_std | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | balanced_accuracy_std | roc_auc_mean | roc_auc_std | specificity_invalid_mean | specificity_invalid_std | FPR_mean | FPR_std | pr_auc_mean | pr_auc_std | false_positives_by_fold | false_negatives_by_fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 5 | 0.8440 | 0.0167 | 0.8338 | 0.0189 | 0.8434 | 0.0168 | 0.8430 | 0.0168 | 0.9404 | 0.0166 | 0.8755 | 0.0256 | 0.1245 | 0.0256 | 0.9412 | 0.0153 | 5/5/7/8/8 | 12/8/9/10/8 |
| seg_full | 5 | 0.9045 | 0.0156 | 0.9008 | 0.0147 | 0.9042 | 0.0155 | 0.9041 | 0.0151 | 0.9633 | 0.0191 | 0.9132 | 0.0424 | 0.0868 | 0.0424 | 0.9628 | 0.0217 | 7/1/7/4/4 | 4/6/4/7/5 |
| embedding_full | 5 | 0.9103 | 0.0158 | 0.9056 | 0.0150 | 0.9100 | 0.0156 | 0.9096 | 0.0152 | 0.9673 | 0.0167 | 0.9321 | 0.0350 | 0.0679 | 0.0350 | 0.9671 | 0.0174 | 2/1/6/4/5 | 6/6/5/5/6 |

### Pooled OOF

| model | n | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | pr_auc | sensitivity_valid | specificity_invalid | FPR | FNR | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 513 | 0.8441 | 0.8590 | 0.8105 | 0.8340 | 0.8435 | 0.8430 | 0.9382 | 0.9358 | 0.8105 | 0.8755 | 0.1245 | 0.1895 | 232 | 33 | 47 | 201 |
| seg_full | 513 | 0.9045 | 0.9061 | 0.8952 | 0.9006 | 0.9043 | 0.9042 | 0.9587 | 0.9611 | 0.8952 | 0.9132 | 0.0868 | 0.1048 | 242 | 23 | 26 | 222 |
| embedding_full | 513 | 0.9103 | 0.9244 | 0.8871 | 0.9053 | 0.9101 | 0.9096 | 0.9625 | 0.9584 | 0.8871 | 0.9321 | 0.0679 | 0.1129 | 247 | 18 | 28 | 220 |

### Por fold

| fold | model | n | best_pose_epoch | best_seg_epoch | best_head_epoch | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | pr_auc | sensitivity_valid | specificity_invalid | FPR | FNR | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pose | 103 | 20 | 15 | 3 | 0.8350 | 0.8837 | 0.7600 | 0.8172 | 0.8334 | 0.8328 | 0.9528 | 0.9540 | 0.7600 | 0.9057 | 0.0943 | 0.2400 | 48 | 5 | 12 | 38 |
| 1 | seg_full | 103 | 20 | 15 | 3 | 0.8932 | 0.8679 | 0.9200 | 0.8932 | 0.8932 | 0.8940 | 0.9766 | 0.9763 | 0.9200 | 0.8679 | 0.1321 | 0.0800 | 46 | 7 | 4 | 46 |
| 1 | embedding_full | 103 | 20 | 15 | 3 | 0.9223 | 0.9565 | 0.8800 | 0.9167 | 0.9220 | 0.9211 | 0.9815 | 0.9825 | 0.8800 | 0.9623 | 0.0377 | 0.1200 | 51 | 2 | 6 | 44 |
| 2 | pose | 103 | 18 | 10 | 3 | 0.8738 | 0.8936 | 0.8400 | 0.8660 | 0.8734 | 0.8728 | 0.9649 | 0.9644 | 0.8400 | 0.9057 | 0.0943 | 0.1600 | 48 | 5 | 8 | 42 |
| 2 | seg_full | 103 | 18 | 10 | 3 | 0.9320 | 0.9778 | 0.8800 | 0.9263 | 0.9316 | 0.9306 | 0.9943 | 0.9943 | 0.8800 | 0.9811 | 0.0189 | 0.1200 | 52 | 1 | 6 | 44 |
| 2 | embedding_full | 103 | 18 | 10 | 3 | 0.9320 | 0.9778 | 0.8800 | 0.9263 | 0.9316 | 0.9306 | 0.9909 | 0.9910 | 0.8800 | 0.9811 | 0.0189 | 0.1200 | 52 | 1 | 6 | 44 |
| 3 | pose | 103 | 17 | 23 | 12 | 0.8447 | 0.8542 | 0.8200 | 0.8367 | 0.8443 | 0.8440 | 0.9189 | 0.9241 | 0.8200 | 0.8679 | 0.1321 | 0.1800 | 46 | 7 | 9 | 41 |
| 3 | seg_full | 103 | 17 | 23 | 12 | 0.8932 | 0.8679 | 0.9200 | 0.8932 | 0.8932 | 0.8940 | 0.9528 | 0.9627 | 0.9200 | 0.8679 | 0.1321 | 0.0800 | 46 | 7 | 4 | 46 |
| 3 | embedding_full | 103 | 17 | 23 | 12 | 0.8932 | 0.8824 | 0.9000 | 0.8911 | 0.8932 | 0.8934 | 0.9457 | 0.9464 | 0.9000 | 0.8868 | 0.1132 | 0.1000 | 47 | 6 | 5 | 45 |
| 4 | pose | 102 | 12 | 4 | 3 | 0.8235 | 0.8298 | 0.7959 | 0.8125 | 0.8229 | 0.8225 | 0.9288 | 0.9313 | 0.7959 | 0.8491 | 0.1509 | 0.2041 | 45 | 8 | 10 | 39 |
| 4 | seg_full | 102 | 12 | 4 | 3 | 0.8922 | 0.9130 | 0.8571 | 0.8842 | 0.8916 | 0.8908 | 0.9449 | 0.9308 | 0.8571 | 0.9245 | 0.0755 | 0.1429 | 49 | 4 | 7 | 42 |
| 4 | embedding_full | 102 | 12 | 4 | 3 | 0.9118 | 0.9167 | 0.8980 | 0.9072 | 0.9116 | 0.9112 | 0.9549 | 0.9509 | 0.8980 | 0.9245 | 0.0755 | 0.1020 | 49 | 4 | 5 | 44 |
| 5 | pose | 102 | 19 | 38 | 4 | 0.8431 | 0.8367 | 0.8367 | 0.8367 | 0.8429 | 0.8429 | 0.9365 | 0.9325 | 0.8367 | 0.8491 | 0.1509 | 0.1633 | 45 | 8 | 8 | 41 |
| 5 | seg_full | 102 | 19 | 38 | 4 | 0.9118 | 0.9167 | 0.8980 | 0.9072 | 0.9116 | 0.9112 | 0.9480 | 0.9501 | 0.8980 | 0.9245 | 0.0755 | 0.1020 | 49 | 4 | 5 | 44 |
| 5 | embedding_full | 102 | 19 | 38 | 4 | 0.8922 | 0.8958 | 0.8776 | 0.8866 | 0.8919 | 0.8916 | 0.9634 | 0.9649 | 0.8776 | 0.9057 | 0.0943 | 0.1224 | 48 | 5 | 6 | 43 |

LEFT late_full no fue evaluada en la CV original; no se agrega retrospectivamente.

## Tabla C — Test final

| model | accuracy | precision | recall | f1 | macro_f1 | balanced_accuracy | roc_auc | average_precision | sensitivity | specificity | fpr | fnr | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pose | 0.9231 | 0.8936 | 0.9545 | 0.9231 | 0.9231 | 0.9241 | 0.9749 | 0.9728 | 0.9545 | 0.8936 | 0.1064 | 0.0455 | 42 | 5 | 2 | 42 |
| seg_full | 0.8901 | 0.8696 | 0.9091 | 0.8889 | 0.8901 | 0.8907 | 0.9483 | 0.9433 | 0.9091 | 0.8723 | 0.1277 | 0.0909 | 41 | 6 | 4 | 40 |
| late_full | 0.9341 | 0.9130 | 0.9545 | 0.9333 | 0.9341 | 0.9347 | 0.9710 | 0.9660 | 0.9545 | 0.9149 | 0.0851 | 0.0455 | 43 | 4 | 2 | 42 |
| embedding_full | 0.9011 | 0.8723 | 0.9318 | 0.9011 | 0.9011 | 0.9021 | 0.9710 | 0.9687 | 0.9318 | 0.8723 | 0.1277 | 0.0682 | 41 | 6 | 3 | 41 |

AP corresponde a Average Precision, no al área trapezoidal de PR. Positivo=1 válido; **FP=inválido aprobado**, FN=válido rechazado.

## Diferencias Test menos Validation

| model | accuracy_test_minus_validation | f1_test_minus_validation | macro_f1_test_minus_validation | roc_auc_test_minus_validation | specificity_test_minus_validation | fpr_test_minus_validation |
|---|---|---|---|---|---|---|
| pose | 0.0220 | 0.0220 | 0.0220 | -0.0082 | 0.0213 | -0.0213 |
| seg_full | -0.0549 | -0.0549 | -0.0549 | -0.0455 | -0.0638 | 0.0638 |
| late_full | -0.0220 | -0.0212 | -0.0219 | -0.0222 | -0.0426 | 0.0426 |
| embedding_full | -0.0659 | -0.0644 | -0.0659 | -0.0227 | -0.1064 | 0.1064 |

CV usa nuevos pesos por fold y menos datos en cada inner Train; Test usa checkpoints originales seleccionados. El promedio CV no es rendimiento del mismo checkpoint ni debe coincidir necesariamente con Test.

## Errores pareados (B respecto a A)

| a | b | pearson | shared_errors | exclusive_a | exclusive_b | fp_corrected | fp_introduced | fn_corrected | fn_introduced | disagreements |
|---|---|---|---|---|---|---|---|---|---|---|
| pose | seg_full | 0.8806 | 5 | 2 | 5 | 1 | 2 | 1 | 3 | 7 |
| pose | late_full | 0.9687 | 5 | 2 | 1 | 1 | 0 | 1 | 1 | 3 |
| pose | embedding_full | 0.9353 | 5 | 2 | 4 | 1 | 2 | 1 | 2 | 6 |
| seg_full | embedding_full | 0.9854 | 9 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| late_full | embedding_full | 0.9908 | 6 | 0 | 3 | 0 | 2 | 0 | 1 | 3 |

IDs completos y correlaciones: `paired_analysis.json`; todos los errores: `errors_by_video.csv`; cercanía al umbral |p−0.5|≤0.05: `near_threshold.csv`.

## Incertidumbre exploratoria 95%

| model | metric | lower | upper |
|---|---|---|---|
| pose | accuracy | 0.8681 | 0.9670 |
| pose | f1 | 0.8605 | 0.9730 |
| pose | macro_f1 | 0.8654 | 0.9670 |
| pose | balanced_accuracy | 0.8671 | 0.9732 |
| pose | specificity | 0.7959 | 0.9773 |
| pose | roc_auc | 0.9457 | 0.9948 |
| pose | fpr_wilson | 0.0463 | 0.2259 |
| pose | specificity_wilson | 0.7741 | 0.9537 |
| seg_full | accuracy | 0.8242 | 0.9451 |
| seg_full | f1 | 0.8140 | 0.9505 |
| seg_full | macro_f1 | 0.8224 | 0.9451 |
| seg_full | balanced_accuracy | 0.8233 | 0.9491 |
| seg_full | specificity | 0.7674 | 0.9583 |
| seg_full | roc_auc | 0.8950 | 0.9882 |
| seg_full | fpr_wilson | 0.0598 | 0.2517 |
| seg_full | specificity_wilson | 0.7483 | 0.9402 |
| late_full | accuracy | 0.8791 | 0.9780 |
| late_full | f1 | 0.8736 | 0.9792 |
| late_full | macro_f1 | 0.8786 | 0.9780 |
| late_full | balanced_accuracy | 0.8794 | 0.9792 |
| late_full | specificity | 0.8261 | 0.9808 |
| late_full | roc_auc | 0.9362 | 0.9956 |
| late_full | fpr_wilson | 0.0336 | 0.1993 |
| late_full | specificity_wilson | 0.8007 | 0.9664 |
| embedding_full | accuracy | 0.8352 | 0.9560 |
| embedding_full | f1 | 0.8293 | 0.9583 |
| embedding_full | macro_f1 | 0.8339 | 0.9560 |
| embedding_full | balanced_accuracy | 0.8362 | 0.9574 |
| embedding_full | specificity | 0.7674 | 0.9583 |
| embedding_full | roc_auc | 0.9377 | 0.9940 |
| embedding_full | fpr_wilson | 0.0598 | 0.2517 |
| embedding_full | specificity_wilson | 0.7483 | 0.9402 |

Bootstrap pareado ordinario por video, 10000 réplicas aceptadas de 10000, seed 20261015; 0 descartadas por una sola clase. Percentiles 2.5/97.5; mismos índices para todos los modelos. Wilson para FPR/especificidad. Sin agrupamiento acreditado ni corrección por multiplicidad; no se utiliza para seleccionar modelos.

### seg_full minus pose
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | -0.0330 | -0.0879 | 0.0220 |
| macro_f1 | -0.0330 | -0.0898 | 0.0222 |
| specificity | -0.0213 | -0.0980 | 0.0476 |

### late_full minus pose
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | 0.0110 | -0.0220 | 0.0549 |
| macro_f1 | 0.0110 | -0.0225 | 0.0549 |
| specificity | 0.0213 | 0.0000 | 0.0698 |

### embedding_full minus pose
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | -0.0220 | -0.0769 | 0.0330 |
| macro_f1 | -0.0220 | -0.0771 | 0.0330 |
| specificity | -0.0213 | -0.0980 | 0.0476 |

### embedding_full minus seg_full
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | 0.0110 | 0.0000 | 0.0330 |
| macro_f1 | 0.0110 | 0.0000 | 0.0340 |
| specificity | 0.0000 | 0.0000 | 0.0000 |

### embedding_full minus late_full
| metric | delta | lower | upper |
|---|---|---|---|
| accuracy | -0.0330 | -0.0769 | 0.0000 |
| macro_f1 | -0.0330 | -0.0770 | 0.0000 |
| specificity | -0.0426 | -0.1087 | 0.0000 |

## Interpretación experimental

El principal obtiene Accuracy=0.9011, F1=0.9011, Macro-F1=0.9011, AUC=0.9710. Aprueba incorrectamente 6 de 47 inválidos (FPR=0.1277); esta tasa debe figurar explícitamente como limitación arbitral.

Respecto a Validation, ΔAccuracy=-0.0659, ΔF1=-0.0644, ΔMacro-F1=-0.0659, ΔAUC=-0.0227. Una caída es compatible con optimismo de selección/sobreajuste y variabilidad de muestra; no identifica por sí sola la causa.

Principal frente a pose: ΔAccuracy=-0.0220, ΔMacro-F1=-0.0220, ΔFP=+1. Son diferencias descriptivas; revisar sus IC pareados antes de afirmar superioridad.

Principal frente a seg_full: ΔAccuracy=+0.0110, ΔMacro-F1=+0.0110, ΔFP=+0. Son diferencias descriptivas; revisar sus IC pareados antes de afirmar superioridad.

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

250 artefactos protegidos y 364 entradas verificadas antes/después, sin cambios. Ver también `protected_inputs_before.json` y `protected_inputs_after.json` para MP4/RAW/keypoints/features completos.

Código congelado: `training/bp_lateral_final_test_v1/run.py` y `analyze.py`. Comandos: `python -B -m training.bp_lateral_final_test_v1.run --view left --stage freeze`, luego `validation`, luego `test`; análisis `python -B -m training.bp_lateral_final_test_v1.analyze --view left`. El evaluador rechaza sobrescribir protocolos o repetir inferencias ya guardadas. Los artefactos existentes permiten recalcular métricas sin volver a ejecutar modelos.

![ROC](roc.png)

![Precision Recall](precision_recall.png)

![pose](confusion_pose.png)

![seg_full](confusion_seg_full.png)

![late_full](confusion_late_full.png)

![embedding_full](confusion_embedding_full.png)

## Respuestas para la tesis

1. **La fusión embeddings FULL no mantuvo su ventaja de Validation.** Pasó de Accuracy 0.9670/F1 0.9655 a 0.9011/0.9011. La especificidad pasó de 0.9787 a 0.8723; FP aumentó de 1 a 6 en conjuntos con 47 inválidos cada uno.
2. **No superó a PoseCNN:** tuvo 9 errores frente a 7, y 6 FP frente a 5. Frente a SegCNN FULL corrigió solamente un FN (`bp_055`), sin reducir sus seis FP.
3. La fusión tardía tuvo descriptivamente el mejor resultado Test: Accuracy 0.9341/F1 0.9333 y 4 FP. Frente a Pose corrigió FP `bp_501` y FN `bp_411`, e introdujo FN `bp_413`. No reemplaza al principal.
4. Embeddings frente a Pose corrigió FP `bp_501` y FN `bp_411`, pero introdujo FP `bp_421`, `bp_511` y FN `bp_385`, `bp_413`. Pose y Seg compartieron cinco errores: `bp_314`, `bp_375`, `bp_428`, `bp_443`, `bp_475`.
5. El F1 Test del principal (0.9011) se aproxima a CV (0.9056 ± 0.0150; pooled OOF 0.9053), mucho más que a Validation (0.9655). ROC-AUC Test 0.9710 frente a CV 0.9673 ± 0.0167. Esto es compatible con optimismo de selección de Validation y no demuestra por sí solo una causa única de sobreajuste. CV y Test no utilizan el mismo checkpoint.
6. IC95% Accuracy principal [0.8352, 0.9560], F1 [0.8293, 0.9583], especificidad [0.7674, 0.9583]. ΔAccuracy embeddings−Pose: −0.0220, IC [−0.0769, 0.0330]. ΔAccuracy tardía−Pose: +0.0110, IC [−0.0220, 0.0549]. No hay respaldo concluyente para superioridad global de las fusiones. Los intervalos con extremos cero o [0,0] reflejan pocos desacuerdos o decisiones idénticas en la muestra, no equivalencia poblacional.
7. Los seis inválidos aprobados (FPR 12.77%; Wilson 95% [5.98%,25.17%]) son una limitación relevante para arbitraje, pese al AUC alto. El principal se conserva por selección previa; la tesis debe informar su falta de ventaja final.

### Registro técnico

El análisis bootstrap excedió el límite inicial de 120 s. Se reanudó una vez con el mismo código, semilla y predicciones, ampliando solo el tiempo permitido. No se repitió inferencia ni se modificaron protocolos. Ver `analysis_technical_resume.json`.

La etapa `protect` debe ejecutarse entre `freeze` y `validation`. Calcula hashes binarios de MP4, RAW, keypoints y features después del congelamiento; no interpreta secuencias ni ejecuta detectores. Los flags `test_opened=false` de reproducción se refieren a carga/decodificación de secuencias, no al hash binario previo. Se comprobaron 2512 archivos de insumos antes/después, además de 250 artefactos de modelos, configuración, código y CV.
