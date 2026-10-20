# BP_RIGHT — Fusión PoseCNN + SegCNN temporal V1 (solo Validation)

**RIGHT_FUSION_VALIDATION_COMPLETE.** Cohorte 431 Train y 93 Validation; Test se consultó **solo como cantidad de IDs** (93). No se abrió su NPY/NPZ, no se entrenó ni evaluó en Test, no hay clasificador SIDE.

## Integridad previa y contrato congelado

- PoseCNN + ambas SegCNN reprodujeron **exactamente** las probabilidades base de Validation (`base_reproducibility_full.json`, `base_reproducibility_sin_relacion.json`: max diff 0). Sus checkpoints, scalers y contratos, junto con split, labels y cohortes, conservan SHA-256 en `fusion_source_manifest.json`; los hashes antes/después son idénticos (`fusion_integrity_verification.json`). IDs y etiquetas coinciden estrictamente dentro de esta vista. Ninguna muestra de la otra vista se incorporó.
- `PoseEmbeddingAdapter.extract_embeddings`: `(N,64)` desde entrada `(N,3,170,8)`. `SegEmbeddingAdapter.extract_embeddings`: `(N,128)` tras AdaptiveAvgPool1d desde `(N,C,170)`; los forwards de ambos originales se reconstruyeron sin alterar probabilidad/logits. Concatenación `(N,192)`; cabeza Linear(192,32)→ReLU→Dropout(.40)→Linear(32,1): **6209 parámetros entrenables**. Pose y Seg: **0 entrenables**, `eval()` permanente para BatchNorm. Embeddings por vista normalizados con media/desviación fit **solo Train**; `embedding_fusion_*/embedding_scaler_train.npz` y `embeddings_development.npz` no contienen Test.
- Backbones conservados; selección de SegCNN FULL y SIN relación por menor Validation loss previa; Pose reentrenada en 431 Train seed1337. La cabeza se entrenó con embeddings **in-sample** de Train (no OOF): AdamW lr=.001, weight_decay=.001, batch32, 100 epochs máx, scheduler plateau patience5/factor.5 y early stopping patience15. Dos seeds 42/1337 por variante; mejor checkpoint/seed por menor Validation loss, umbral fijo .5. No fine-tuning de backbones.
- Fusión tardía: `p=.5*p_pose+.5*p_seg` en cada video, sin parámetros ni ajuste de pesos; `late_fusion_*/validation_predictions.csv`. Validez positiva 1. FPR=FP/(TN+FP). Fuente SegCNN parcialmente asistida por Pose; las separaciones barra–torso son mayormente proyecciones 2D, no contacto físico confirmado.

## Siete configuraciones por video (Validation 93)

| Configuración | Acc | Prec | Recall/sens válido | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | Sens válido | Spec inválido | FPR | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.8817 | 0.8750 | 0.8537 | 0.8642 | 0.8797 | 0.8788 | 0.9250 | 0.9134 | 0.8537 | 0.9038 | 0.0962 | 47/5/6/35 |
| seg_full | 0.8817 | 0.8947 | 0.8293 | 0.8608 | 0.8790 | 0.8762 | 0.9369 | 0.9144 | 0.8293 | 0.9231 | 0.0769 | 48/4/7/34 |
| seg_sin_relacion | 0.9247 | 0.9250 | 0.9024 | 0.9136 | 0.9235 | 0.9224 | 0.9402 | 0.9427 | 0.9024 | 0.9423 | 0.0577 | 49/3/4/37 |
| late_full | 0.9247 | 0.9250 | 0.9024 | 0.9136 | 0.9235 | 0.9224 | 0.9376 | 0.9285 | 0.9024 | 0.9423 | 0.0577 | 49/3/4/37 |
| late_sin_relacion | 0.9032 | 0.8810 | 0.9024 | 0.8916 | 0.9021 | 0.9031 | 0.9353 | 0.9398 | 0.9024 | 0.9038 | 0.0962 | 47/5/4/37 |
| embedding_full | 0.9032 | 0.9000 | 0.8780 | 0.8889 | 0.9016 | 0.9006 | 0.9400 | 0.9388 | 0.8780 | 0.9231 | 0.0769 | 48/4/5/36 |
| embedding_sin_relacion | 0.9032 | 0.8810 | 0.9024 | 0.8916 | 0.9021 | 0.9031 | 0.9381 | 0.9415 | 0.9024 | 0.9038 | 0.0962 | 47/5/4/37 |

Mayor Accuracy: **['seg_sin_relacion', 'late_full']**. Mayor F1: **['seg_sin_relacion', 'late_full']**. Mayor Macro-F1: **['seg_sin_relacion', 'late_full']**. Menos falsos positivos: **['seg_sin_relacion', 'late_full']**. La clase positiva es válido, por lo que FP significa un intento inválido aceptado. Listas completas:
- pose: ['bp_001', 'bp_053', 'bp_180', 'bp_384', 'bp_441'] (5 FP).
- seg_full: ['bp_001', 'bp_384', 'bp_441', 'bp_456'] (4 FP).
- seg_sin_relacion: ['bp_001', 'bp_384', 'bp_387'] (3 FP).
- late_full: ['bp_001', 'bp_384', 'bp_441'] (3 FP).
- late_sin_relacion: ['bp_001', 'bp_180', 'bp_384', 'bp_387', 'bp_441'] (5 FP).
- embedding_full: ['bp_001', 'bp_053', 'bp_384', 'bp_441'] (4 FP).
- embedding_sin_relacion: ['bp_001', 'bp_180', 'bp_384', 'bp_387', 'bp_441'] (5 FP).

## Ocho ejecuciones de cabeza entre ambas vistas (cuatro de esta vista)

| Seg variante | Seed cabeza | Mejor época / épocas corridas | Loss Train / Val | F1 Train / Val | Macro-F1 Val | FP Val |
|---|---:|---|---|---|---:|---:|
| full | 42 | 1 / 16 | 0.2130 / 0.2963 | 0.9290 / 0.8889 | 0.9016 | 4 |
| full | 1337 | 1 / 16 | 0.2153 / 0.3019 | 0.9315 / 0.9000 | 0.9123 | 3 |
| sin_relacion | 42 | 1 / 16 | 0.1960 / 0.3000 | 0.9340 / 0.9048 | 0.9132 | 5 |
| sin_relacion | 1337 | 2 / 17 | 0.1551 / 0.2998 | 0.9333 / 0.8916 | 0.9021 | 5 |

Mejores seeds por menor loss Validation: FULL 42, SIN relación 1337. Los checkpoints de ambas seeds, métricas, curvas, matrices, predicciones y scaler Train están en `embedding_fusion_*/`. La brecha entre pérdidas Train–Validation y la elección de épocas tempranas señalan **sobreajuste potencial de la cabeza**; ninguna búsqueda adicional de hiperparámetros se efectuó. Las representaciones de Train son in-sample respecto de ambos backbones.

## Corrección/introducción de errores (comparación pareada)

| Fusión | Errores base compartidos | Compartidos corregidos / retenidos | Exclusivos Pose / Seg corregidos | Errores nuevos (ambos base correctos) | FP fusión |
|---|---:|---|---|---:|---:|
| late full | 7 | 0 / 7 | 4 / 4 | 0 | 3 |
| embedding full | 7 | 0 / 7 | 3 / 3 | 0 | 4 |
| late sin_relacion | 5 | 0 / 5 | 3 / 1 | 0 | 5 |
| embedding sin_relacion | 5 | 0 / 5 | 3 / 1 | 0 | 5 |

Los IDs se preservan en `fusion_validation_predictions.csv` y `fusion_pairwise_analysis.json`. No se presenta un oráculo de elección entre modelos como rendimiento de fusión ni se crean reglas ad hoc. La reproducción posterior desde NPY Pose + NPZ Seg sobre Validation confirmó **todas las decisiones iguales**: máximo error absoluto de probabilidad 5.76e-06 por variación numérica de lote; las ocho cabezas también reproducen sus probabilidades con su propia matriz de embeddings. Ningún Test abierto.

## Candidato preliminar y limitaciones

- Regla congelada **antes** del entrenamiento: mayor Macro-F1 Validation, luego especificidad de inválidos, luego Balanced Accuracy; empates exactos favorecen menor complejidad. Las cabezas eligen primero seed por menor Validation loss. `fusion_config.json` documenta pesos tardíos, umbral y criterio. **Preselección BP_RIGHT: seg_sin_relacion**, checkpoint `C:\Users\basti\MediapipePythonProjects\training\bp_right_segcnn_v1\sin_relacion\seed_1337\best_validation_loss.pt`, SHA `28d1ddc1a1408d88bb7617b97d9cb7f86ef7b783fa735fd1d8326ee7ac25458b`; `fusion_selected_candidate.json` incluye ranking, hashes de backbones y scaler si aplica. Esta elección se hizo **antes de abrir Test**.
- Validation ya se utilizó antes para checkpoints y variantes, y aquí se reutiliza: **no es un estimador externo independiente**. La PoseCNN lateral histórica ya expuso Test en un proyecto anterior, aunque aquí no se abrió. No está comprobada la independencia por levantamiento físico, atleta o sesión entre LEFT/RIGHT. La SegCNN lateral deriva parte de sus features desde pose, por lo que complementariedad no equivale a modalidades estadísticamente independientes. No interpretar extrapolaciones 2D como contacto físico.
