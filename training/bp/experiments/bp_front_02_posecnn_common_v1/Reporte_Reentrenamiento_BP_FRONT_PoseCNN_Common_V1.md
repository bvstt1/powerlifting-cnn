# Reentrenamiento PoseCNN BP_FRONT — cohorte común V1

**Alcance:** únicamente BP_FRONT; dos semillas, pesos del clasificador
inicializados desde cero, sin inferencia YOLO, sin reentrenar SegCNN,
sin fusión y **sin abrir ningún keypoint ni ejecutar inferencia del Test**.

## 1–3. Cohorte, split e integridad

Fuente exclusiva: `segmentation/bp/front/temporal_features_v1/
bp_front_features_common_cohort.csv` (663 IDs únicos). Cohorte incluida:
**661** videos, Train **462** (279 válidos/183 inválidos), Validation **99**
(60/39), Test **100** reservado. `bp_349` (RAW segmentación inválido) y
`bp_352` (duplicado de bp_351) permanecen fuera. La auditoría comparó solo
el **inventario** antiguo y común: **19 IDs del Test común figuraban como
Train histórico Pose-only**. Por ello no se cargaron pesos, scaler ni split
históricos para el nuevo entrenamiento. Positivo: `1=válido`.

## 4. Procedencia, shape y ausencias

Fuente de entrenamiento: `keypoints/bp/yolo/front/bp_NNN.npy`, sin mezclas
MediaPipe, y extractor `bp_front_skeleton_v8.pt`. K=6 en orden
`codo izq / hombro izq / hombro der / codo der / muñeca der / muñeca izq`;
canales `x,y` normalizados + confianza YOLO. En los **561** videos abiertos
(Train+Validation), `(T,6,3)` coincide con T del inventario: **0** Inf,
T incorrectos o archivos keypoints inexistentes. Hay 162 NaN en **bp_406** (Train):
9 frames originales consecutivos completos sin pose, índices 7–15.

El archivo de auditoría histórico `training/data/bp/front/
bp_front_pose_audit.csv` registra `keypoints/bp/front`, mientras el **código
histórico actual** `training/bp/bp_front_posecnn.py` apunta a
`keypoints/bp/yolo/front`. Al contrastar ambas carpetas solo en
Train+Validation, la **única** diferencia de contenido está en `bp_406`:
la copia del audit histórico tiene 0 NaN, el YOLO original solicitado
tiene 162. Esta ejecución utiliza exclusivamente el segundo, sin copiar
landmarks de la carpeta histórica.

## 5–6. Arquitectura original y diferencias

El código original se localizó en `training/bp/bp_front_posecnn.py`
(no en la antigua ruta `training/bp_front_posecnn.py`). Se reprodujo **sin
cambiar capas ni parámetros**:

```text
Entrada: (N,3,170,6)
Conv2D(3→32,k3,padding=same) → BN → ReLU
Conv2D(32→64,k3,padding=same) → BN → ReLU
MaxPool2D(2,1) → Dropout2D(0,40)
Conv2D(64→128,k3,padding=same) → BN → ReLU
AdaptiveAvgPool2D(1,1) → Flatten
Linear(128→128) → ReLU → Dropout(0,50)
Linear(128→64) → ReLU → Dropout(0,50)
Linear(64→1) → Sigmoid
```

**118.529 parámetros entrenables.** Se conservó `Sigmoid + BCELoss` de la
implementación histórica para comparabilidad estructural (sin AMP, FP32);
no se mezcló Sigmoid con `BCEWithLogitsLoss`. La única adaptación del
preprocesamiento afecta a la ausencia real de bp_406: el hueco largo
permanece ausente, no se rellenó con landmarks inventados. También se
usa exclusivamente el split común. Se añadieron métodos de lectura
de embeddings y checkpoints **sin introducir nuevas capas**.

## 7–9. Preprocesamiento, remuestreo y augmentations

Por video: centro espacial = mediana de puntos medios entre hombros válidos;
escala = mediana de distancia interhombros de frames válidos. Se normalizan
solo x,y; confianza permanece cruda hasta el scaler. El grid original de
PoseCNN `linspace(0,1,T) → linspace(0,1,170)` representa los mismos
instantes que SegCNN `linspace(0,T−1,170)`; una prueba numérica sobre
bp_001 verifica la equivalencia. En frames completos se reutiliza la
interpolación lineal histórica; en bp_406 no se interpola a través de
ninguno de los 9 frames perdidos. El vector `(x,y,conf)=(0,0,0)` en esos
instantes es un **sentinel de ausencia, no una observación**; se conserva
`availability` aparte (174 celdas keypoint×paso ausentes en Train tras
remuestreo). El modelo original recibe exactamente 3 canales; la máscara
sidecar no se añade a la arquitectura.

`StandardScaler` histórico de 3×170×6 posiciones se ajusta con **462
videos Train originales normalizados**, nunca Validation/Test. Para cada
semilla se aplicó únicamente a Train un factor de augmentation ×3:
**462 levantamientos físicos → 1.848 secuencias de optimización**.
Cinco transformaciones independientes, cada una con probabilidad 0,5:
shift x/y uniforme ±0,05; escala x/y uniforme 0,9–1,1;
ruido gaussiano x/y σ=0,01; rotación uniforme ±10°;
desplazamiento temporal entero uniforme −3…+3 con relleno de borde.
Sin flips. Orden histórico preservado: fit scaler Train original →
augmentar Train sin escalar → transformar Train aumentado y Validation.
Solo en la ausencia de bp_406 se restablece el sentinel tras la
augmentation (para no fabricar pose). El scaler incluye la confianza,
por lo que después de escalar esta es una **feature estandarizada**, no
una probabilidad YOLO directamente interpretable; sus valores crudos y
disponibilidad se conservaron en la fuente/manifest.

## 10–16. Entrenamiento y resultados por semilla

Adam(lr=0,001, weight_decay=1e−4), batch=16, máximo 200 épocas,
ReduceLROnPlateau(mode=min, factor=0,5, patience=5), early stopping
patience=15. Checkpoint = menor loss Validation de cada semilla; selección
entre semillas = menor loss Validation; umbral binario fijo **0,5**.
Entrenamiento FP32/CUDA RTX 5060 Laptop GPU; PyTorch, NumPy, sklearn,
CUDA y hashes en `source_manifest.json`. Train reporta métricas sobre
los 462 levantamientos **originales** (no sobre 1.848 augmentations).

| Seed | Época mejor / ejecutadas | Duración | Loss Train / Val | Train F1 | Val Acc | Val Prec | Val Recall | Val F1 | Val Macro-F1 | Val Bal Acc | Val ROC-AUC | Val PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **42** | **69 / 84** | **156,8 s** | **0,1150 / 0,2275** | 0,9591 | 0,8990 | 0,8676 | 0,9833 | 0,9219 | 0,8895 | 0,8763 | **0,9641** | **0,9677** |
| 1337 | 49 / 64 | 30,4 s | 0,1308 / 0,2398 | 0,9571 | 0,8990 | 0,8676 | 0,9833 | 0,9219 | 0,8895 | 0,8763 | 0,9504 | 0,9633 |

Ambas semillas tienen la misma matriz Validation **[[TN=30, FP=9],
[FN=1, TP=59]]**, sensibilidad para válidos 0,9833 y especificidad
para inválidos 0,7692. Precisión para inválidos (NPV) 0,9677. La clase
mayoritaria en Train es válido: predecir siempre 1 habría acertado
60/99 (Accuracy 0,6061, especificidad 0). Historial y curvas por seed:
`seed_*/learning_curves.csv/.png`, probabilidades
`seed_*/validation_probabilities.png`, matrices `.npy/.png`, predicciones
individuales en `seed_*/validation_predictions.csv`.
`evaluate_validation.py` recargó el checkpoint seed 42 y el scaler Train y
reprodujo las 99 probabilidades en la misma GPU con diferencia máxima
**0,0** (`validation_recheck.json`), sin abrir Test.

## 17. Selección, distribución y sobreajuste

Elegido por menor Validation loss: **`seed_42/best_validation_loss.pt`**,
época 69, loss=0,2275. El fichero contiene solo pesos aprendidos en esta
ejecución nueva, seed, época y umbral; `selected_checkpoint.json`
documenta la regla. Probabilidades por video en Validation y casos
erróneos: 9 inválidos falsamente válidos, 1 válido falsamente inválido.
Train F1=0,9591 frente a Validation F1=0,9219;
loss Train=0,1150 frente a Val=0,2275 (**brecha 0,1125**).
Hay indicios de sobreajuste y volatilidad temprana de la seed 42
(época 20). Early stopping guarda la mejor época, no la última.
No se infiere generalización final a partir de Validation.

## 18–20. Comparación preliminar con SegCNN (mismos 99 videos)

Checkpoints SegCNN **ya congelados**, sin reentrenar: FULL seed 1337 y
SIN relación seed 42. Se cotejaron los mismos 99 IDs, las mismas etiquetas
positivas y umbral 0,5 (`paired_validation_predictions.csv`).

| Modelo | Acc | Prec válido | Recall válido | F1 | Macro-F1 | Balanced Acc | ROC-AUC | PR-AUC | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PoseCNN Common seed 42 | 0,8990 | 0,8676 | **0,9833** | 0,9219 | 0,8895 | 0,8763 | **0,9641** | **0,9677** | 30/9/1/59 |
| SegCNN FULL seed 1337 | 0,8687 | 0,8730 | 0,9167 | 0,8943 | 0,8605 | 0,8558 | 0,9500 | 0,9664 | 31/8/5/55 |
| SegCNN SIN relación seed 42 | **0,9192** | **0,9062** | 0,9667 | **0,9355** | **0,9137** | **0,9064** | 0,9462 | 0,9580 | 33/6/2/58 |

- Pose vs FULL: **6 errores compartidos**, 4 solo Pose,
  7 solo FULL; 11 predicciones distintas; corr(prob)=0,8600.
- Pose vs SIN: **6 compartidos**, 4 solo Pose, 2 solo SIN;
  6 predicciones distintas; corr(prob)=0,9156.
- **5 errores compartidos por los tres**: bp_386, bp_387, bp_475,
  bp_564, bp_570. Consulte IDs completos por pareja en
  `comparison_validation.json`.

Existe complementariedad **descriptiva** porque cada pareja acierta
algunos errores de la otra (hasta 93/99 con un oráculo que eligiera
perfectamente por video). Esta cifra **no es un resultado de fusión**:
un sistema real no conoce qué modelo acierta en un caso dado.

## 21–24. Embeddings, integridad, limitaciones y siguiente etapa

`model.extract_embeddings(x)` devuelve un tensor **(N,128)** del
`AdaptiveAvgPool2D((1,1))` justo antes de `classifier.Flatten`; el modelo
recibe (N,3,170,6) y `forward` devuelve (N,1). La interfaz
`model.cargar_inferencia(checkpoint)` / `model.cargar_checkpoint(checkpoint)`
solo acepta checkpoints de este experimento; `data.preprocesar_video_para_inferencia`
aplica la geometría y el `train_scaler.npz` congelado. Se probó el ciclo
de carga e inferencia con **bp_001 Train** (embedding shape 1×128) sin
guardar embeddings de Test. El `preprocessing_development_manifest.csv`
conserva identificadores, T, ausencias, centro y escala por video.

`source_manifest.json`: hashes SHA-256 de cohort CSV, código histórico,
checkpoint de YOLO Pose consultado sin inferencia, checkpoints/predicciones
SegCNN existentes y **561 archivos YOLO Pose Train/Validation**.
`integrity_after_training.json` verifica **0 cambios de fuentes**,
contiene hashes de ambos nuevos checkpoints y `test_npys_opened=0`.

Limitaciones: la probabilidad SegCNN se construye en parte a partir de
Pose, por lo que la complementariedad no equivale a independencia modal;
9 frames de bp_406 usan un sentinel explícito, una diferencia mínima pero
real frente al artefacto histórico; las confidencias estandarizadas no
son scores YOLO crudos. El split común debe seguir congelado. La fusión
posterior podrá utilizar embeddings, pero su entrenamiento necesitará
un protocolo Train/Validation sin evaluar Test ni reutilizar de manera
optimista predicciones in-sample como si fueran out-of-fold. **Esta fase
no entrenó fusión y no evaluó Test**.
