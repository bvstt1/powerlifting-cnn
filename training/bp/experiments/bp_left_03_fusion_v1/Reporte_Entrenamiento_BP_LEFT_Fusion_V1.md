# BP_LEFT — Fusión PoseCNN + SegCNN temporal V1 (solo Validation)

**LEFT_FUSION_VALIDATION_COMPLETE.** Cohorte 422 Train y 91 Validation; Test se consultó **solo como cantidad de IDs** (91). No se abrió su NPY/NPZ, no se entrenó ni evaluó en Test, no hay clasificador SIDE.

## Integridad previa y contrato congelado

- PoseCNN + ambas SegCNN reprodujeron **exactamente** las probabilidades base de Validation (`base_reproducibility_full.json`, `base_reproducibility_sin_relacion.json`: max diff 0). Sus checkpoints, scalers y contratos, junto con split, labels y cohortes, conservan SHA-256 en `fusion_source_manifest.json`; los hashes antes/después son idénticos (`fusion_integrity_verification.json`). IDs y etiquetas coinciden estrictamente dentro de esta vista. Ninguna muestra de la otra vista se incorporó.
- `PoseEmbeddingAdapter.extract_embeddings`: `(N,64)` desde entrada `(N,3,170,8)`. `SegEmbeddingAdapter.extract_embeddings`: `(N,128)` tras AdaptiveAvgPool1d desde `(N,C,170)`; los forwards de ambos originales se reconstruyeron sin alterar probabilidad/logits. Concatenación `(N,192)`; cabeza Linear(192,32)→ReLU→Dropout(.40)→Linear(32,1): **6209 parámetros entrenables**. Pose y Seg: **0 entrenables**, `eval()` permanente para BatchNorm. Embeddings por vista normalizados con media/desviación fit **solo Train**; `embedding_fusion_*/embedding_scaler_train.npz` y `embeddings_development.npz` no contienen Test.
- Backbones conservados; selección de SegCNN FULL y SIN relación por menor Validation loss previa; Pose histórica compatible seed42. La cabeza se entrenó con embeddings **in-sample** de Train (no OOF): AdamW lr=.001, weight_decay=.001, batch32, 100 epochs máx, scheduler plateau patience5/factor.5 y early stopping patience15. Dos seeds 42/1337 por variante; mejor checkpoint/seed por menor Validation loss, umbral fijo .5. No fine-tuning de backbones.
- Fusión tardía: `p=.5*p_pose+.5*p_seg` en cada video, sin parámetros ni ajuste de pesos; `late_fusion_*/validation_predictions.csv`. Validez positiva 1. FPR=FP/(TN+FP). Fuente SegCNN parcialmente asistida por Pose; las separaciones barra–torso son mayormente proyecciones 2D, no contacto físico confirmado.

## Siete configuraciones por video (Validation 91)

| Configuración | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Sens válido | Spec inválido | FPR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.9011 | 0.8723 | 0.9318 | 0.9011 | 0.9011 | 0.9021 | 0.9831 | 0.9839 | 0.9318 | 0.8723 | 0.1277 | 41/6/3/41 |
| seg_full | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9937 | 0.9943 | 0.9545 | 0.9362 | 0.0638 | 44/3/2/42 |
| seg_sin_relacion | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9923 | 0.9923 | 0.9545 | 0.9362 | 0.0638 | 44/3/2/42 |
| late_full | 0.9560 | 0.9545 | 0.9545 | 0.9545 | 0.9560 | 0.9560 | 0.9932 | 0.9933 | 0.9545 | 0.9574 | 0.0426 | 45/2/2/42 |
| late_sin_relacion | 0.9560 | 0.9545 | 0.9545 | 0.9545 | 0.9560 | 0.9560 | 0.9908 | 0.9907 | 0.9545 | 0.9574 | 0.0426 | 45/2/2/42 |
| embedding_full | 0.9670 | 0.9767 | 0.9545 | 0.9655 | 0.9670 | 0.9666 | 0.9937 | 0.9941 | 0.9545 | 0.9787 | 0.0213 | 46/1/2/42 |
| embedding_sin_relacion | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9927 | 0.9928 | 0.9545 | 0.9362 | 0.0638 | 44/3/2/42 |

Mayor Accuracy: **['embedding_full']**. Mayor F1: **['embedding_full']**. Mayor Macro-F1: **['embedding_full']**. Menos falsos positivos: **['embedding_full']**. La clase positiva es válido, por lo que FP significa un intento inválido aceptado. Listas completas:
- pose: ['bp_174', 'bp_240', 'bp_241', 'bp_417', 'bp_465', 'bp_486'] (6 FP).
- seg_full: ['bp_240', 'bp_241', 'bp_489'] (3 FP).
- seg_sin_relacion: ['bp_240', 'bp_241', 'bp_436'] (3 FP).
- late_full: ['bp_240', 'bp_241'] (2 FP).
- late_sin_relacion: ['bp_240', 'bp_241'] (2 FP).
- embedding_full: ['bp_241'] (1 FP).
- embedding_sin_relacion: ['bp_240', 'bp_241', 'bp_436'] (3 FP).

## Ocho ejecuciones de cabeza entre ambas vistas (cuatro de esta vista)

| Seg variante | Seed cabeza | Mejor época / épocas corridas | Loss Train / Val | F1 Train / Val | Macro-F1 Val | FP Val |
|---|---:|---|---|---|---:|---:|
| full | 42 | 6 / 21 | 0.0571 / 0.1030 | 0.9754 / 0.9438 | 0.9450 | 3 |
| full | 1337 | 3 / 18 | 0.0711 / 0.0983 | 0.9726 / 0.9655 | 0.9670 | 1 |
| sin_relacion | 42 | 3 / 18 | 0.0588 / 0.1157 | 0.9827 / 0.9438 | 0.9450 | 3 |
| sin_relacion | 1337 | 3 / 18 | 0.0584 / 0.1080 | 0.9802 / 0.9438 | 0.9450 | 3 |

Mejores seeds por menor loss Validation: FULL 1337, SIN relación 1337. Los checkpoints de ambas seeds, métricas, curvas, matrices, predicciones y scaler Train están en `embedding_fusion_*/`. La brecha entre pérdidas Train–Validation y la elección de épocas tempranas señalan **sobreajuste potencial de la cabeza**; ninguna búsqueda adicional de hiperparámetros se efectuó. Las representaciones de Train son in-sample respecto de ambos backbones.

## Corrección/introducción de errores (comparación pareada)

| Fusión | Errores base compartidos | Compartidos corregidos / retenidos | Exclusivos Pose / Seg corregidos | Errores nuevos (ambos base correctos) | FP fusión |
|---|---:|---|---|---:|---:|
| late full | 3 | 0 / 3 | 6 / 1 | 0 | 2 |
| embedding full | 3 | 1 / 2 | 6 / 1 | 0 | 1 |
| late sin_relacion | 3 | 0 / 3 | 6 / 1 | 0 | 2 |
| embedding sin_relacion | 3 | 0 / 3 | 6 / 0 | 0 | 3 |

Los IDs se preservan en `fusion_validation_predictions.csv` y `fusion_pairwise_analysis.json`. No se presenta un oráculo de elección entre modelos como rendimiento de fusión ni se crean reglas ad hoc. La reproducción posterior desde NPY Pose + NPZ Seg sobre Validation confirmó **todas las decisiones iguales**: máximo error absoluto de probabilidad 1.51e-05 por variación numérica de lote; las ocho cabezas también reproducen sus probabilidades con su propia matriz de embeddings. Ningún Test abierto.

## Candidato preliminar y limitaciones

- Regla congelada **antes** del entrenamiento: mayor Macro-F1 Validation, luego especificidad de inválidos, luego Balanced Accuracy; empates exactos favorecen menor complejidad. Las cabezas eligen primero seed por menor Validation loss. `fusion_config.json` documenta pesos tardíos, umbral y criterio. **Preselección BP_LEFT: embedding_full**, checkpoint `C:\Users\basti\MediapipePythonProjects\training\bp_left_fusion_v1\embedding_fusion_full\seed_1337\best_validation_loss.pt`, SHA `1467c5b11df73d76d773ff8f8abc77b9ffdf2d0e5853957f3c4e1cf03cbaa9ad`; `fusion_selected_candidate.json` incluye ranking, hashes de backbones y scaler si aplica. Esta elección se hizo **antes de abrir Test**.
- Validation ya se utilizó antes para checkpoints y variantes, y aquí se reutiliza: **no es un estimador externo independiente**. La PoseCNN lateral histórica ya expuso Test en un proyecto anterior, aunque aquí no se abrió. No está comprobada la independencia por levantamiento físico, atleta o sesión entre LEFT/RIGHT. La SegCNN lateral deriva parte de sus features desde pose, por lo que complementariedad no equivale a modalidades estadísticamente independientes. No interpretar extrapolaciones 2D como contacto físico.
