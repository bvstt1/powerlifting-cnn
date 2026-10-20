# Proyecto de Tesis: Juez Virtual de Powerlifting

## Descripción General

Este proyecto es una **tesis** cuyo objetivo es crear un **juez virtual para powerlifting** utilizando visión por computador y redes neuronales. Se analizan tres levantamientos compuestos:

- **Sentadilla (sq)** — Squat
- **Press de Banca (bp)** — Bench Press
- **Peso Muerto (dl)** — Deadlift

Cada levantamiento se graba desde **3 vistas sincronizadas**: frontal (front), lateral izquierda (left) y lateral derecha (right).

### Pipeline completo

1. **Detección de esqueletos** → YOLO pose / MediaPipe
2. **Extracción de keypoints** → Archivos `.npy` con forma `(T, N, 3)`
3. **Etiquetado manual** → Archivos **CSV** con válido/nulo + códigos de falta
4. **Preprocesamiento** → Limpieza, interpolación, normalización espacial y temporal
5. **Split/normalización/aumentación** → split por evento físico; scaler fit solo
   con train original; aumento solo de train; transform posterior de train/val/test
6. **Clasificación** → **PoseCNN** (CNN 2D sobre keypoints) u otras arquitecturas

---

## Estructura del Proyecto

```
MediapipePythonProjects/
├── AGENTS.md                              # Este archivo
├── script.py                              # Organizador de dataset
├── .gitignore
│
├── extract_keypoints/                     # Scripts de extracción de keypoints
│   ├── bp/                                # Extractores MediaPipe y YOLO BP
│   ├── dl/
│   │   ├── extract_keypoints_front_dl.py  # MediaPipe: 22 body + 2 hand centroids + bar-knee dist = 25 kps
│   │   ├── extract_keypoints_left_dl.py   # YOLO: shoulder+hip lateral (2 kps)
│   │   └── extract_keypoints_right_dl.py  # YOLO: shoulder+hip lateral (2 kps)
│   └── sq/
│       ├── extract_keypoints_front_sq.py  # MediaPipe: 33 landmarks → (T, 22, 3)
│       ├── extract_keypoints_left_sq.py   # MediaPipe: cadera/rodilla/tobillo izq → (T, 3, 3)
│       ├── extract_keypoints_right_sq.py  # MediaPipe: cadera/rodilla/tobillo der → (T, 3, 3)
│       └── extract_bar_features.py        # Tracking de barra con CSRT + velocidad/aceleración
│
├── keypoints/                             # Keypoints extraídos (.npy)
│   ├── bp/{yolo,mediapipe}/              # Pose BP frontal y lateral según extractor
│   ├── dl/mediapipe/front/               # Pose DL frontal
│   ├── dl/yolo/{left,right,side}/         # Pose DL lateral; left puede ser parcial localmente
│   └── sq/                                # Consultar extractor para versión y esquema
│
├── training/                              # Ver training/README.md y docs/EXPERIMENTOS.md
│   ├── bp/                                # PoseCNN BP monovista y frontal histórico
│   ├── dl/                                # 01-12: pipeline y baselines DL V2/V3
│   ├── sq/                                # PoseCNN y estudios controlados SQ
│   ├── common/train_utils.py              # Arquitectura PoseCNN y utilidades
│   ├── configs/training_config.py         # Perfiles DL y physical_event()
│   ├── tests/test_event_mapping.py        # Pruebas de mapping físico DL
│   ├── legacy/fix_csv.py                  # Reparación histórica de CSV
│   ├── docs/                              # Histórico, inventario y guía de experimentos
│   ├── bp/experiments/bp_*_v1/           # Experimentos BP migrados; por vista numerados bp_<vista>_<NN>_<tipo>_v1; hashes en README
│   ├── data/{bp,dl,sq}/                   # Entradas intermedias, splits, auditorías
│   ├── models/{bp,dl,sq}/                 # Checkpoints y scalers por ejecución
│   └── results/{bp,dl,sq}/                # Métricas, predicciones y reportes
│
├── etiquetado/                            # Datos etiquetados en CSV
│   ├── bp/
│   │   ├── bp_laterales.csv               # 1,483 filas (left+right combinados)
│   │   ├── bp_left.csv                    # 495 filas (vista izquierda)
│   │   └── bp_right.csv                   # 989 filas (vista derecha)
│   ├── dl/
│   │   ├── dl_front.csv                   # 441 filas (CSV frontal histórico reparado)
│   │   ├── dl_left.csv                    # 412 filas + auditoría/exclusión manual
│   │   ├── dl_right.csv                   # 413 filas + auditoría/exclusión manual
│   │   └── dl_side.csv                    # 825 filas consolidadas y sincronizadas
│   ├── historico/
│   │   ├── README.md                      # Procedencia y fechas de los respaldos
│   │   ├── dl/*_pre_manual_audit.csv      # CSV antes de la auditoría manual
│   │   └── sq/*.csv.bak                   # CSV antes de retirar filas sin vídeo
│   └── sq/
│       ├── sq_front.csv                   # 444 filas (vista frontal)
│       ├── sq_laterales.csv               # 522 filas (laterales combinados)
│       ├── sq_left.csv                    # 371 filas (vista izquierda)
│       └── sq_right.csv                   # 385 filas (vista derecha)
│
├── display/                               # Visualización y demos
│   ├── bp/
│   │   ├── mediapipe_skeleton_bp_front.py      # MP BP frontal: cabeza + hombros/codos/muñecas
│   │   ├── mediapipe_skeleton_bp_lateral.py    # MP BP lateral: cuerpo inferior con suavizado EMA
│   │   ├── yolo_skeleton_bp_front.py          # YOLO pose BP frontal (6 kps, modelo v6)
│   │   ├── yolo_skelton_bp_side.py            # YOLO pose BP lateral (8 kps, modelo v1)
│   │   ├── yolo_seg_model.py                  # YOLO segmentación BP frontal
│   │   ├── recording_processed_bp.py          # Grabación multi-view sincronizada + MediaPipe
│   │   ├── label_press_frame.py               # Herramienta interactiva para etiquetar frame "Press"
│   │   ├── bp_laterales.csv                   # (copia en display/)
│   │   └── bp_press_lateral_command.csv       # Timestamps de comando Press (221 registros)
│   ├── dl/
│   │   ├── mediapipe_skeleton_dl_front.py     # MP DL frontal: cuerpo completo + barra simulada
│   │   ├── mediapipe_skeleton_dl_left.py      # MP DL izquierda: hombro+cadera+pierna (6 kps)
│   │   ├── mediapipe_skeleton_dl_right.py     # MP DL derecha: hombro+cadera (2 kps)
│   │   └── yolo_seg_dl_front.py              # YOLO segmentación + MediaPipe combinados (DL frontal)
│   └── sq/
│       ├── mediapipe_skeleton_sq_front.py     # MP SQ frontal: 33 landmarks
│       ├── mediapipe_skeleton_sq__left.py     # MP SQ izquierda: cadera/rodilla/tobillo izq
│       ├── mediapipe_skeleton_sq_right.py     # MP SQ derecha: cadera/rodilla/tobillo der
│       ├── yolo_skeleton_sq_front.py          # YOLO pose SQ frontal (v4, 14 kps + EMA smooth)
│       ├── yolo_seg_sq_front.py              # YOLO segmentación SQ frontal
│       ├── testing.py                        # Máquina de estados para validar profundidad SQ
│       └── bar_grafic.py                     # Gráfico de movimiento vertical de barra
│
├── live/                                # Cámara en vivo
│   ├── live.py                          # Cámara índice 3
│   ├── live2.py                         # Cámara índice 1
│   └── live3.py                         # Cámara índice 2
│
├── models/                              # Modelos organizados por ejercicio
│   ├── bp/
│   │   ├── bp_front_seg_v1.pt           # YOLO segmentación BP frontal
│   │   ├── bp_front_skeleton_v[3-6].pt  # YOLO pose BP frontal (varias versiones)
│   │   ├── bp_side_skeleton_v1.pt       # YOLO pose BP lateral
│   │   └── bp_object_model.pt           # YOLO detección objetos BP
│   ├── dl/
│   │   ├── dl_front_seg_v1.pt           # YOLO segmentación DL frontal
│   │   └── dl_side_skeleton_v1.pt       # YOLO pose DL lateral
│   ├── sq/
│   │   ├── sq_front_seg_v1.pt           # YOLO segmentación SQ frontal
│   │   ├── sq_front_skeleton_v[1-4].pt  # YOLO pose SQ frontal (varias versiones)
│   │   └── sq_side_skeleton_v[1-2].pt   # YOLO pose SQ lateral (varias versiones)
│   └── common/
│       └── pose_landmarker_heavy.task    # MediaPipe Pose Landmarker compartido
│
└── runs/pose/                           # Entrenamientos YOLO
    ├── train/
    ├── train-2/
    └── train-3/
```

---

## Dataset de Video

| Ejercicio | Vista Front | Vista Left | Vista Right | Vista Side | Total |
|-----------|-------------|------------|-------------|------------|-------|
| BP (Press Banca) | 663 | 611 | 600 | 0 | 1,874 |
| DL (Peso Muerto) | 441 | 412 | 413 | 0 | 1,266 |
| SQ (Sentadilla) | 444 | 371 | 384 | 0 | 1,199 |
| **Total** | **1,548** | **1,394** | **1,397** | **0** | **4,339** |

Los videos están en `dataset/<ejercicio>/<vista>/` (ignorados por git). En DL,
right conserva `dl_001`-`dl_413` y left usa la secuencia corregida
`dl_001`-`dl_412` después de eliminar el hueco del video original `dl_310`
(auditoría contra conteo de frames/T de keypoints: `left[N]==right[N]` hasta
N=309 y `left[N]==right[N+1]` desde N=310; el evento 401 sí tiene ambas
vistas). La consolidación `dl_001`-`dl_825` se mantiene únicamente en
`keypoints/dl/yolo/side`. Fuente de verdad del mapping: `physical_event()` en
`training/configs/training_config.py`.

### Mapping físico definitivo de Deadlift lateral

Mapping auditado contra filenames, conteo de frames de video y longitud `T` de
los `.npy` (no inferido desde la propia función):

| Rango físico | Right | Left | Side consolidado |
|---|---|---|---|
| Eventos 001-309 | `right N` | `left N` | right `N`; left `N+413` |
| Evento 310 | `right 310` | **sin muestra** | `side 310` (right) |
| Eventos 311-413 | `right N` | `left N-1` | right `N`; left `N+412` |

- El hueco físico real es **310**, no 401.
- El evento 401 tiene ambas vistas: `right 401 ↔ left 400 ↔ side 813`.
- Frontera verificada: `left 309 == right 309`; `left 310 == right 311`.
- Cola verificada: `left 412 == right 413`.
- `physical_event(video_id, view)` es la **única fuente de verdad** para
  `dl_left`, `dl_right` y `dl_side`; no duplicar la fórmula en otros scripts.
- Tests: `python -B -m training.tests.test_event_mapping`.

### Auditoría manual de labels DL

Se revisaron manualmente los 41 eventos que originalmente tenían
`left_label != right_label`:

- 15 labels left modificados.
- 18 labels right modificados.
- Evento físico 302 descartado explícitamente en ambas vistas, sin borrar los
  registros ni videos: `manual_exclude=1`,
  `exclude_reason=manual_review_discard`.
- Resultado de los 41 casos: 15 pares `0/0`, 16 pares `1/1`, 3 pares `0/1`,
  6 pares `1/0` y 1 descartado.
- Quedan 9 discordancias válidas: eventos 171, 172, 174, 277, 278, 282, 283,
  285 y 351. No deben corregirse automáticamente.
- Backups pre-auditoría en `etiquetado/historico/dl/*_pre_manual_audit.csv`.
- Log reproducible en `training/data/dl/shared/manual_label_audit.csv`.
- Export vigente en `training/data/dl/shared/discordant_view_labels.csv`.

La fila manual que decía `236/236` fue confirmada como typo y aplicada al
evento `235` (`dl_235/dl_235 → 0/0`). No se modificaron códigos, criterios,
IDs, filenames ni metadata ajena a la auditoría.

### Auditoría de calidad DL

Los flags manuales son metadata descriptiva y **nunca son input de PoseCNN**:

- `*L`, `*R`, `*LR`: pérdida de frames en left/right/ambas.
- `+L`, `+R`: detección deficiente de esqueleto/keypoints.
- `+'L`, `+'R`: detección deficiente y label original incorrecto.
- Los flags no excluyen automáticamente; se aplican las reglas objetivas ya
  existentes: shape, mínimo 10 frames, sin Inf, interpolación de gaps ≤5 y
  ratio de frames con NaN en x/y ≤0.30.

Estado de las 82 muestras (41 eventos × 2 vistas) auditadas:

| Estado | N | Detalle |
|---|---:|---|
| `QUALITY_OK` | 66 | Sin flag y pasa reglas objetivas |
| `QUALITY_WARNING` | 11 | Con flag, pero pasa preprocessing |
| `QUALITY_EXCLUDED` | 5 | 196L, 197L y 373R por NaN>30%; 302L/302R manual |

Flags contabilizados: frame loss left=3, right=6, ambas=1 evento; skeleton
issue left=3, right=2; label original incorrecto left=1, right=1. Métricas
detalladas (`total_frames_original`, `valid_frames`, `nan_ratio`, gaps y
confidence) están en `training/data/dl/shared/manual_quality_audit.csv`.

## Keypoints Extraídos

| Ejercicio | Vista | Archivos .npy | Forma | Landmarks |
|-----------|-------|---------------|-------|-----------|
| Peso Muerto | Front | 441 | (T, 25, 3) | 22 body + 2 hand centroids + bar-knee distance |
| Peso Muerto | Left | 412 | (T, 2, 3) | YOLO: shoulder + hip lateral |
| Peso Muerto | Right | 413 | (T, 2, 3) | YOLO: shoulder + hip lateral |
| Peso Muerto | Side | 825 | (T, 2, 3) | Right 001-413 + left 414-825 |
| Sentadilla | Front | 316 | (T, 22, 3) | 33 landmarks sin rostro |
| Sentadilla | Left | 245 | (T, 3, 3) | Cadera + rodilla + tobillo izquierdo |
| Sentadilla | Right | 274 | (T, 3, 3) | Cadera + rodilla + tobillo derecho |

## Datos Etiquetados (CSV)

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `etiquetado/bp/bp_laterales.csv` | 1,483 | BP lateral combinado (left+right) |
| `etiquetado/bp/bp_left.csv` | 495 | BP vista izquierda |
| `etiquetado/bp/bp_right.csv` | 989 | BP vista derecha |
| `etiquetado/dl/dl_front.csv` | 441 | DL vista frontal |
| `etiquetado/dl/dl_left.csv` | 412 | DL vista izquierda (auditado) |
| `etiquetado/dl/dl_right.csv` | 413 | DL vista derecha (auditado) |
| `etiquetado/dl/dl_side.csv` | 825 | DL lateral consolidado (auditado) |
| `etiquetado/sq/sq_front.csv` | 270 | SQ vista frontal |
| `etiquetado/sq/sq_laterales.csv` | 522 | SQ lateral combinado |
| `etiquetado/sq/sq_left.csv` | 275 | SQ vista izquierda |
| `etiquetado/sq/sq_right.csv` | 248 | SQ vista derecha |

**Esquema común:** `video_id, movimiento, camara, archivo, resultado, label [1=valido/0=invalido], codigos [BP-10|SQ-01|DL-03], criterios, etiquetador, timestamp`

Los CSV laterales DL añaden `manual_exclude` y `exclude_reason`; solo el evento
302 tiene `manual_exclude=1`. No usar quality flags como features del modelo.

---

## Pipeline de Entrenamiento (CNN)

### Arquitectura: PoseCNN

```
Input: (batch, 3, 170, K)  →  (N, C, T, K), K=25 front / K=2 lateral
  └─ Conv2D(32, 3×3) → BatchNorm → ReLU
  └─ Conv2D(64, 3×3) → BatchNorm → ReLU
  └─ MaxPool2D(2×1) + Dropout2D(40%)
  └─ Conv2D(128, 3×3) → BatchNorm → ReLU
  └─ GlobalAvgPool2D(1×1)
  └─ Dense(64) → ReLU → Dropout(50%)
  └─ Dense(1) → Sigmoid
```

### Flujo

| Paso | Script | Descripción |
|------|--------|-------------|
| 1 | `01_preprocessing.py` | Carga `.npy`, valida (min 10 frames, <30% NaN), interpola gaps (max 5), normaliza espacialmente (centro cadera, escala torso), cruza con CSV |
| 2 | `02_temporal_normalization.py` | Interpola todas las secuencias a **170 frames** (scipy.interpolate lineal) |
| 3 | `03_split_scale.py` | Split **70/15/15 por physical_event**, común para left/right; guarda datos originales sin escalar |
| 4 | `04_augmentation.py` | Fit `StandardScaler` solo con train original `(N,1020)`; aumento train x4 sin escalar; transform train aumentado/val/test |
| 5 | `05_train_cnn.py` | PoseCNN: Adam(lr=0.001), BCELoss, weight decay 1e-4, patience=8, max 200 épocas |
| 6 | `09/11_*_cv.py` | **5-Fold CV solo en development**; test excluido; folds comunes por evento físico |
| 7 | `07_report.py` | Genera reporte Markdown automático con métricas, matriz de confusión, FP/FN, curvas |

Orden obligatorio vigente:

```text
preprocessing → temporal normalization → grouped physical-event split
→ scaler.fit(train original) → augmentation(train original únicamente)
→ scaler.transform(train aumentado / validation / test) → PoseCNN
```

`StandardScaler` recibe `(N,170,2,3) → reshape (N,1020)`: cada combinación
`timestep × keypoint × canal` se escala como feature independiente. Mantener
esta estrategia para comparabilidad con Baseline V2/V3.

### Resultados (DL Front)

| Métrica | Test | 5-Fold CV (promedio) |
|---------|------|---------------------|
| Accuracy | 0.8871 | 0.728 |
| Precision | 0.9429 | — |
| Recall | 0.8684 | — |
| **F1-Score** | **0.9041** | **0.819** |
| **ROC-AUC** | **0.9254** | **0.891** |

**Matriz de confusión:** TN=22, FP=2, FN=5, TP=33

**Diagnóstico:** Overfitting (mejor época = 10, divergencia desde época ~3). 7 misclasificados (2 FP, 5 FN).

### Deadlift Lateral: Baseline V2

Arquitectura y features sin cambios (`x,y,confidence`, T=170, K=2, C=3).
V2 incorporó mapping gap-310, split físico común y orden correcto del scaler.

| Perfil | Test Accuracy | Test F1 | Test AUC | 5-Fold CV F1 | 5-Fold CV AUC |
|---|---:|---:|---:|---:|---:|
| `dl_left` | 0.7705 | 0.7667 | 0.8820 | 0.8335±0.0508 | 0.8964±0.0353 |
| `dl_right` | 0.8226 | 0.8308 | 0.8942 | 0.8218±0.0453 | 0.8639±0.0611 |
| `dl_side` | 0.7967 | 0.8120 | 0.8682 | 0.7815±0.0602 | 0.8559±0.0439 |

Paired test V2 (61 eventos): left F1=0.7667/AUC=0.8820; right
F1=0.8254/AUC=0.8932. CV V2 usa 351 eventos development y excluye 62
eventos test. Asignación común: `training/data/dl/shared/cv_event_folds.csv`.

### Deadlift Lateral: Baseline V3 post auditoría manual

V3 mantiene **exactamente la misma PoseCNN e hiperparámetros** de V2. Cambian
ground truth, exclusión del evento 302 y, por ello, split/folds regenerados.

Dataset efectivo tras reglas objetivas + descarte manual:

| Perfil | Total | Válidos | Inválidos | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|
| `dl_left` | 403 | 222 | 181 | 283 | 60 | 60 |
| `dl_right` | 411 | 226 | 185 | 287 | 62 | 62 |
| `dl_side` | 814 | 448 | 366 | 570 | 122 | 122 |

Resultados V3:

| Perfil | Test Acc | Test F1 | Test AUC | CV Acc | CV F1 | CV AUC |
|---|---:|---:|---:|---:|---:|---:|
| `dl_left` | 0.8167 | 0.8358 | 0.8923 | 0.8278±0.0441 | 0.8445±0.0395 | 0.8924±0.0403 |
| `dl_right` | 0.8226 | 0.8451 | 0.8645 | 0.8193±0.0514 | 0.8399±0.0413 | 0.8727±0.0521 |
| `dl_side` | 0.7869 | 0.8219 | 0.8179 | 0.7905±0.0556 | 0.8108±0.0446 | 0.8757±0.0476 |

- Paired test V3 (60 eventos): left F1=0.8358/AUC=0.8923; right
  F1=0.8358/AUC=0.8627.
- Paired CV V3: left F1=0.8445±0.0395/AUC=0.8917±0.0403; right
  F1=0.8388±0.0397/AUC=0.8738±0.0508.
- CV V3 usa 350 eventos development, test completamente excluido y folds
  comunes en `training/data/dl/shared/cv_event_folds_v3.csv`.
- No interpretar V2→V3 como cambio arquitectónico: refleja labels auditados,
  descarte 302 y redistribución del split.

### Análisis CLEAN vs QUALITY_WARNING (test V3)

| Perfil/subset | N | Accuracy | F1 | AUC | Errores |
|---|---:|---:|---:|---:|---:|
| Left CLEAN | 60 | 0.8167 | 0.8358 | 0.8923 | 11 |
| Left WARNING | 0 | N/D | N/D | N/D | 0 |
| Right CLEAN | 61 | 0.8197 | 0.8451 | 0.8638 | 11 |
| Right WARNING | 1 | 1.0000 | 0.0000 | N/D | 0 |
| Side CLEAN | 121 | 0.7934 | 0.8276 | 0.8195 | 25 |
| Side WARNING | 1 | 0.0000 | 0.0000 | N/D | 1 |

La única muestra WARNING en test fue `event_301/right`: correcta en el modelo
right e incorrecta en side. Es análisis descriptivo; no se seleccionaron
modelos ni thresholds con estos subconjuntos.

---

## Dependencias Principales

| Paquete | Versión | Uso |
|---------|---------|-----|
| mediapipe | 0.10.31 | Detección de pose landmarks |
| ultralytics | 8.4.51 | YOLOv8/26 para pose y segmentación |
| torch | 2.13.0.dev+cu130 | PyTorch con CUDA 13 |
| opencv-python | 4.13.0 | Procesamiento de video |
| opencv-contrib-python | 4.13.0 | CSRT tracker (barra) |
| numpy | 2.2.6 | Arrays numéricos y .npy |
| matplotlib | 3.10.8 | Gráficos |
| scipy | 1.15.3 | Interpolación, gradientes |
| scikit-learn | — | Split, scaler, métricas |
| roboflow | 1.3.8 | Gestión de datasets |
| tqdm | 4.67.3 | Barras de progreso |

---

## Estado Actual

### ✅ Completado
- [x] Dataset de 4,339 videos en 3 ejercicios × 3 vistas
- [x] Extracción de keypoints con MediaPipe para **sentadilla** (front, left, right)
- [x] Extracción de keypoints para **peso muerto** (MediaPipe front: 25 kps; YOLO left/right: 2 kps)
- [x] Tracking de barra con CSRT (sentadilla)
- [x] **11 CSV principales etiquetados** para los 3 ejercicios (incluye vistas combinadas)
- [x] Datos etiquetados integrados con pipeline de entrenamiento (DL front)
- [x] **Pipeline completo de preprocesamiento** (limpieza, normalización espacial/temporal, split, aumento)
- [x] **PoseCNN entrenada** para clasificación DL front (F1=0.904, ROC-AUC=0.925)
- [x] **PoseCNN entrenada** para DL side con split agrupado (F1=0.822, ROC-AUC=0.840)
- [x] **5-Fold CV agrupada DL side** (F1=0.796 ± 0.046, ROC-AUC=0.848 ± 0.031)
- [x] Pipeline parametrizado con perfiles `dl_front`, `dl_left`, `dl_right` y `dl_side`
- [x] **5-Fold Cross-Validation** con análisis de estabilidad
- [x] Mapping físico DL lateral auditado (hueco real evento 310) y centralizado
- [x] Split común left/right por `physical_event`, sin leakage
- [x] Scaler fit solo en train original; augmentation solo train antes de transform
- [x] Baseline V2 left/right/side + CV común solo sobre development
- [x] Auditoría manual de 41 labels, 33 cambios y descarte del evento 302
- [x] Auditoría objetiva de quality flags (OK/WARNING/EXCLUDED)
- [x] Baseline V3 post auditoría + CV común + evaluación pareada
- [x] Análisis CLEAN vs QUALITY_WARNING y comparación V2/V3
- [x] **Reporte automático** con métricas, matriz de confusión, casos FP/FN
- [x] Modelos YOLO entrenados para **pose** (BP frontal v3-v6, BP lateral v1, SQ frontal v1-v4)
- [x] Modelos YOLO entrenados para **segmentación** (BP frontal v1, SQ frontal v1, DL frontal v1)
- [x] Visualizaciones de esqueleto con MediaPipe y YOLO para los 3 ejercicios
- [x] Herramienta de grabación multi-view sincronizada para BP
- [x] Herramienta interactiva de etiquetado de frame "Press" (`label_press_frame.py`)
- [x] Máquina de estados para validación de profundidad en sentadilla
- [x] Visualización combinada YOLO segmentación + MediaPipe skeleton (DL front)
- [x] Especificación técnica histórica de la CNN (el archivo ya no está en la raíz; consultar la documentación vigente en `training/README.md`)

### 🔄 En Progreso
- [ ] Mejorar generalización de PoseCNN (reducir overfitting)

### ❌ Pendiente
- [x] Extracción de keypoints para **press de banca** (bp) con MediaPipe/YOLO
- [x] Pipeline de entrenamiento para **BP** (front, left, right)
- [x] Pipeline de entrenamiento para **SQ** (front, left, right)
- [ ] Clasificación **multi-clase** (por tipo de falta, no solo binario)
- [ ] Arquitecturas alternativas (LSTM, Transformer)
- [ ] Extender los perfiles reutilizables a BP y SQ
- [ ] Documentación final de la tesis (guía técnica en `training/README.md`)
- [ ] requirements.txt o pyproject.toml

---

## Convenciones del Código

- **Idioma**: Comentarios y variables en **español**
- **Estilo**: Sin tipado explícito, Python simple con numpy/opencv/mediapipe
- **Keypoints guardados como**: `.npy` con forma `(T, N, 3)` donde:
  - `T` = frames
  - `N` = landmarks seleccionados
  - `3` = (x, y, visibility)
- **Rutas de modelos**: Relativas desde el script (`../../models/<ejercicio>/...` o `../../models/common/...`)
- **Parámetros de entrenamiento YOLO**: img=960, batch=8, AdamW(lr=0.001), pose_weight=12.0
- **Parámetros de entrenamiento CNN**: 170 frames, keypoints según perfil (25 front/2 side), Adam(lr=0.001), EarlyStopping(patience=8 en V2/V3), weight decay 1e-4, Dropout2D=0.40, ReduceLROnPlateau

---

## Próximos Pasos (Hoja de Ruta)

1. Revisar generalización y trazabilidad de las cohortes BP y SQ existentes
2. Mejorar generalización de PoseCNN (regularización, más datos, hyperparameter tuning)
3. Comparar los resultados vigentes de BP front/lateral y SQ sin mezclar cohortes
4. Clasificación multi-clase por tipo de falta (no solo válido/nulo)
5. Probar arquitecturas LSTM/Transformer para comparar con CNN
6. Extender la parametrización existente a BP y SQ
7. Evaluar y documentar resultados para la tesis

---

## Actualización de estructura y reglas vigentes (octubre 2026)

La documentación detallada del entrenamiento se mantiene en
[`training/README.md`](training/README.md); el README de raíz es la introducción
para usuarios. Ante diferencias entre este resumen histórico y el código o la
documentación específica actual, inspeccionar el experimento vigente y no asumir
que los nombres/rutas de la estructura antigua siguen existiendo.

### Layout vigente de entrenamiento

- Código de entrenamiento: `training/{bp,dl,sq}/`; utilidades compartidas en
  `training/common/`, configuración en `training/configs/` y pruebas en
  `training/tests/`.
- Artefactos por ejercicio/vista/ejecución:
  `training/data/<ejercicio>/<vista>/`,
  `training/models/<ejercicio>/<vista>/<modelo>/runs/<run-id>/` y
  `training/results/<ejercicio>/<vista>/<modelo>/runs/<run-id>/`.
- Los 20 experimentos BP históricos están en `training/bp/experiments/`.
  No moverlos de vuelta a la raíz ni tratar ese archivo como destino de
  experimentos nuevos. La migración conservó los bytes de 949 archivos; los
  reportes/manifiestos congelados retienen referencias y hashes de su ubicación
  y código originales.
- `training/docs/migration_artifact_manifest.json` registra rutas y hashes
  posteriores a migración. No actualizar evidencia histórica para hacerla
  parecer una ejecución nueva. Una re-ejecución requiere un protocolo y
  manifiesto nuevos.
- Los pesos SegCNN y de fusión de BP pueden vivir dentro de los directorios de
  `training/bp/experiments/`; no asumir que todos los checkpoints están bajo
  `training/models/`. Métricas deben atribuirse a la arquitectura indicada por
  el reporte, nunca por el nombre de la cohorte.

### Reglas para cambios y experimentos

- Ejecutar módulos Python desde la raíz, normalmente como
  `python -m training.<paquete>.<modulo>`.
- Para trabajos nuevos, respetar el layout por `run-id`; no sobrescribir
  artefactos existentes. Guardar configuración, split, hashes de entrada y
  versión de código junto a la ejecución.
- Revisar primero `training/README.md` y la documentación específica del perfil
  antes de cambiar scripts, rutas, protocolos o métricas.
- Los documentos históricos congelados son evidencia, no documentación para
  reformatear durante una migración. Si cambia código o protocolo, registrar una
  nueva ejecución de forma independiente.
- Datos, vídeos y artefactos grandes pueden estar ignorados por Git. No inferir
  que una ruta local ausente implica que se deba reconstruir, eliminar o
  reemplazar el artefacto documentado.
- Los scripts BP parametrizados usan vistas independientes (`left`/`right`);
  SQ incluye estudios controlados front/left/right. DL tiene el pipeline
  parametrizado y baselines V2/V3. Revisar comandos y etapas en
  `training/README.md` antes de lanzar etapas que generen resultados o evalúen
  el test reservado.

### Referencias

- README del repositorio: `README.md`.
- Índice y comandos de entrenamiento: `training/README.md`.
- Inventario, informes y trazabilidad por experimento: `training/docs/` y
  `training/bp/experiments/`.
- Mapping físico DL lateral: `physical_event()` en
  `training/configs/training_config.py`; nunca duplicar su fórmula.
