# Compatibilidad BP_LEFT PoseCNN histórica / SegCNN V1

**LEFT_POSECNN_HISTORICAL_COMPATIBLE**. Se conserva `training/models/bp/left/posecnn/runs/baseline_t170_v1/best_model.pth`; no se entrenó PoseCNN LEFT. Test NPY/SegCNN Test no abiertos ni evaluados.

## Cohorte e integridad

- Coincidencia **por video_id, label y split**: Train=422, Validation=91, Test=91 (solo IDs del split), sin asignación=7; `historical_minus_current={'train': [], 'validation': []}`. Sin muestras adicionales en pesos según split y scaler (`n_samples_seen_=422`). Los siete no asignados no aparecen en Train/Validation histórico. `train_validation_source_audit.csv` registra hashes SHA-256 de cada NPY abierto, su digest histórico de contenido `(T,8,3)`, escala/centro y el grid; `integrity.json` registra hashes de fuentes protegidas.
- Duplicados conocidos en la auditoría de origen: solo los excluidos dentro/sin split, sin duplicados exactos documentados entre Train y Validation/Test; no se leyó ningún NPY Test para esta verificación. Identidad física entre vistas/atletas/sesiones no verificada. No emparejar LEFT/RIGHT por número.
- Orden de puntos en la arquitectura histórica: rodilla, tobillo, cadera, hombro, muñeca/mano, codo, talón, punta. Canales x,y normalizados más confianza YOLO; no MediaPipe. Faltantes XY se interpolan solo en huecos interiores ≤5 (sin extrapolar extremos), confianza ausente a 0, centro fijo mediano del hombro índice 3, escala mediana de distancia hombro3–cadera2. Resample lineal `linspace(0,T_original-1,170)` y entrada `(N,3,170,8)`. SegCNN usa el **mismo grid**, pero abstiene/mantiene máscaras donde PoseCNN interpola breves huecos: misma posición temporal no significa igual política de faltantes.
- Scaler histórico StandardScaler fit sobre Train original `(422,4080)`, antes de aumento x4; máximo |mean/var/scale reconstruidas − guardadas| = {'mean_': 0.0, 'var_': 5.828670879282072e-16, 'scale_': 7.216449660063518e-16}; hash scaler `9b03d5345efabb9621dbfbbcfcf64890ac76034615d864d2978e6e5c8b60bc70`. Aumentos Train: translación, escala, ruido XY y shift temporal, sin flips/rotación. Validación sin aumento; no se comparte scaler con SegCNN.

## Pesos e inferencia reproducible

- Arquitectura inspeccionada `training/common/train_utils.py:PoseCNN`: Conv2D 3×3 padding same 32/64/128, BatchNorm+ReLU, MaxPool(2,1), Dropout2D .40, GAP 128, Dense(128→64), Dropout .50, Dense(64→1), Sigmoid; BCELoss, Adam lr=.001/decay=1e-4, batch16, ReduceLROnPlateau factor .5/patience4, EarlyStopping patience8. 102,017 parámetros; checkpoint SHA `adb2dad4f1b7898b6bb208828472d42cebb4e70fbf535f8da2a4644a9e4e4176`; llaves cargadas estrictamente.
- `history.csv`: mejor época **45**, Validation loss registrada **0.154575216**; reproducida **0.154575218**, diferencia **2.01e-09** con tolerancia predeterminada 0.0002. No existía CSV de probabilidades histórico de este holdout; se reconstruyeron 91 en `validation_predictions.csv` y quedaron ordenadas en `validation_metrics.json`. No se consultaron reportes de Test históricos.
- Embedding externo `training.bp_posecnn_common_v1.core.PoseEmbeddingAdapter.extract_embeddings(x)`: **64 dimensiones**, entrada `(N,3,170,8)`, salida `(N,64)` tras Linear128→64 y ReLU. Reconstrucción exacta de probabilidad original (`max_abs_diff=0.0`), sin modificar pesos ni código histórico. Sin embeddings Test exportados.

## Comparación Validation (mismos 91 levantamientos; clase positiva 1=válido)

| Modelo | Accuracy | Prec | Recall | F1 | Macro-F1 | Bal Acc | ROC-AUC | AP/PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pose | 0.9011 | 0.8723 | 0.9318 | 0.9011 | 0.9011 | 0.9021 | 0.9831 | 0.9839 | 41/6/3/41 |
| seg_full | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9937 | 0.9943 | 44/3/2/42 |
| seg_sin_relacion | 0.9451 | 0.9333 | 0.9545 | 0.9438 | 0.9450 | 0.9454 | 0.9923 | 0.9923 | 44/3/2/42 |

FP=inválido predicho válido. PoseCNN histórica: **6 FP**, IDs **['bp_174', 'bp_240', 'bp_241', 'bp_417', 'bp_465', 'bp_486']**; ver probabilidades en `validation_predictions.csv` y tabla alineada `validation_pose_seg_paired.csv`.

- seg_full: 3 errores compartidos ['bp_240', 'bp_241', 'bp_247']; 6 solo Pose ['bp_174', 'bp_182', 'bp_417', 'bp_462', 'bp_465', 'bp_486']; 2 solo Seg ['bp_050', 'bp_489']; 8 desacuerdos; correlación Pearson de probabilidades 0.9178.
- seg_sin_relacion: 3 errores compartidos ['bp_240', 'bp_241', 'bp_247']; 6 solo Pose ['bp_174', 'bp_182', 'bp_417', 'bp_462', 'bp_465', 'bp_486']; 2 solo Seg ['bp_050', 'bp_436']; 8 desacuerdos; correlación Pearson de probabilidades 0.8980.

Las coincidencias y divergencias son descriptivas, no un resultado de fusión. SegCNN FULL seed42 y SIN relación seed42 se conservaron tal cual; solo se leyeron predicciones Validation. La modalidad SegCNN lateral puede depender de pose en la selección BAR: no asumir independencia estadística completa de entradas. Una fusión futura debe validarse por video en desarrollo sin ajustar usando Test.
