# Reentrenamiento BP_RIGHT PoseCNN cohorte común V1

**RIGHT_POSECNN_COMMON_TRAINED_VALIDATION_COMPLETE**. Dos inicializaciones completamente nuevas; 431 Train, 93 Validation, Test=93 **solo IDs**, 12 sin split excluidos. `bp_354` (MP4 duplicado exacto de `bp_027`) sigue en el split histórico Train pero fue excluido *solo de la cohorte derivada*, sin modificar splits/etiquetas/NPY. Ningún NPY Test abierto o evaluado; no se entrenó ninguna fusión.

## Fuentes, protocolo y procedencia

- `source_manifest.json` documenta coincidencia Train/Validation por ID/etiqueta y hashes de los **524 NPY** usados, sus digest históricos, centro/escala y configuración. Se comprobó `(T,8,3)` contra frames auditados para cada archivo, sin Inf; secuencias normalizadas `(170,8,3)`, entrada `(N,3,170,8)`. Orden: rodilla/tobillo/cadera/hombro/mano/codo/talón/punta; canales `(xyn_x,xyn_y,confianza YOLO)`. Grid `linspace(0,T_original-1,170)` igual al histórico y SegCNN; ausencias YOLO pose: interpolación XY interior ≤5, confianza faltante 0. SegCNN puede abstenerse sobre los mismos instantes; no se fuerza imputación conjunta.
- `StandardScaler` nuevo fit **solo** sobre 431 Train originales × 4080 coordenadas aplanadas, antes del aumento; `scaler_train_original.pkl`, SHA `28ac3bc7c6bf13faf378b0d68793f4e104c7259e75b5f42a5d59dd566506d65e`. Histórico tenía 432 Train según `historical_right_audit.json` y scaler SHA `d9fd0125b1036d1b3222145ce7debdfeb3eef6fedbe4f3baf1967260a76c90ec`. Ningún peso, estado BatchNorm, optimizador o scaler históricos se transfirió. `development_original.npz` almacena **solo Train/Validation**.
- Se reprodujo `training.common.train_utils.PoseCNN` (102.017 parámetros): Conv2D 3×3 same 32/64/128 + BN/ReLU, MaxPool(2,1), Dropout2D .40, GAP128, Linear128→64 ReLU/Dropout .50/Linear64→1/Sigmoid. BCELoss + Adam lr .001 decay 1e-4, batch16, scheduler plateau patience4/factor.5, early stopping patience8, máx 200. Train ×4 (original+3 augmentations) con translación/escala/ruido XY/shift temporal, sin flip/rotación; por seed 1.724 secuencias derivadas de 431 videos, no levantamientos físicos nuevos. Threshold 0,5 fijo. Selección mejor época y seed por **menor loss Validation**.
- Baseline histórico RIGHT conservado para referencia: checkpoint SHA `b008ecb44818f5dc9f331a66f394342805b71b6132e62646dd8e7802a3f5e31a`, estado compatible con arquitectura, scaler fit 432, mejor época histórica de `history.csv` 18. No se leyeron métricas ni predicciones Test históricas. El Test lateral fue evaluado históricamente para PoseCNN y no es completamente nuevo para todo el proyecto.

## Dos ejecuciones desde cero (Validation por video, 1=válido)

| Seed | Mejor época / corridas | Loss Train original / Val | Acc | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP | TN/FP/FN/TP |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 8 / 16 | 0.2174 / 0.3596 | 0.8710 | 0.8718 | 0.8293 | 0.8500 | 0.8684 | 0.8666 | 0.9170 | 0.9142 | 47/5/7/34 |
| 1337 | 13 / 21 | 0.2026 / 0.3398 | 0.8817 | 0.8750 | 0.8537 | 0.8642 | 0.8797 | 0.8788 | 0.9250 | 0.9134 | 47/5/6/35 |

**Seed seleccionada: 1337**; checkpoint `seed_1337/best_model.pth`, SHA `712ab35f7ec36bf8b6bf55fbd213d860a511dcdd760b52eedc894e74cd9a7ecf`, loss Validation 0.3398. El FP significa intento inválido declarado válido. Falsos positivos:
- Seed 42: ['bp_001', 'bp_053', 'bp_180', 'bp_384', 'bp_441'].
- Seed 1337: ['bp_001', 'bp_053', 'bp_180', 'bp_384', 'bp_441'].

Las brechas loss Validation menos Train original fueron [(42, 0.1422), (1337, 0.1372)]; hay señal de generalización imperfecta, sin ajustes adicionales. Curvas y matrices en `seed_*/`; métricas y predicciones en archivos por semilla. Las probabilidades de Validation se reprodujeron al recargar ambos checkpoints con máximo error **0**.
- `PoseEmbeddingAdapter.extract_embeddings(x)` obtiene **64** dimensiones (post-Linear128→64/ReLU); la reconstrucción de probabilidades fue exacta, sin modificar la arquitectura ni los pesos. No se exportaron embeddings Test.

## Comparación preliminar con SegCNN RIGHT (idénticos 93 Validation)

| Modelo | Accuracy | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.8817 | 0.8750 | 0.8537 | 0.8642 | 0.8797 | 0.8788 | 0.9250 | 0.9134 | 47/5/6/35 |
| seg_full | 0.8817 | 0.8947 | 0.8293 | 0.8608 | 0.8790 | 0.8762 | 0.9369 | 0.9144 | 48/4/7/34 |
| seg_sin_relacion | 0.9247 | 0.9250 | 0.9024 | 0.9136 | 0.9235 | 0.9224 | 0.9402 | 0.9427 | 49/3/4/37 |

- seg_full: 7 errores compartidos ['bp_001', 'bp_278', 'bp_363', 'bp_384', 'bp_408', 'bp_441', 'bp_462']; 4 solo Pose ['bp_053', 'bp_180', 'bp_309', 'bp_323']; 4 solo Seg ['bp_382', 'bp_393', 'bp_414', 'bp_456']; 8 desacuerdos; correlación Pearson de probabilidades 0.9010.
- seg_sin_relacion: 5 errores compartidos ['bp_001', 'bp_278', 'bp_363', 'bp_384', 'bp_408']; 6 solo Pose ['bp_053', 'bp_180', 'bp_309', 'bp_323', 'bp_441', 'bp_462']; 2 solo Seg ['bp_387', 'bp_477']; 8 desacuerdos; correlación Pearson de probabilidades 0.8816.

`validation_pose_seg_paired.csv` contiene probabilidades por ID, flags de error/FP, desacuerdos y comparación; `validation_pose_seg_comparison.json` incluye errores compartidos y exclusivos. Es evidencia **descriptiva**, no resultado de una fusión ni prueba de independencia de modalidades. Esta PoseCNN está ya en la cohorte común necesaria para diseñar posteriormente la fusión RIGHT mediante un protocolo de desarrollo que no ajuste hiperparámetros con Test. La identidad física entre vistas y atletas/sesiones sigue sin verificarse.
