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
5. **Aumentación** → Traslación, escala, ruido, rotación, shift temporal (x4)
6. **Clasificación** → **PoseCNN** (CNN 2D sobre keypoints) u otras arquitecturas

---

## Estructura del Proyecto

```
MediapipePythonProjects/
├── AGENTS.md                              # Este archivo
├── Especificacion_CNN_Deadlift_Front.md   # Especificación técnica de la CNN
├── script.py                              # Organizador de dataset
├── .gitignore
│
├── extract_keypoints/                     # Scripts de extracción de keypoints
│   ├── bp/                                # (vacío - pendiente)
│   ├── dl/
│   │   ├── extract_keypoints_front_dl.py  # MediaPipe: 22 body + 2 hand centroids + bar-knee dist = 25 kps
│   │   ├── extract_keypoints_left_dl.py   # MediaPipe: shoulder+hip izquierdo (2 kps)
│   │   └── extract_keypoints_right_dl.py  # MediaPipe: shoulder+hip derecho (2 kps)
│   └── sq/
│       ├── extract_keypoints_front_sq.py  # MediaPipe: 33 landmarks → (T, 22, 3)
│       ├── extract_keypoints_left_sq.py   # MediaPipe: cadera/rodilla/tobillo izq → (T, 3, 3)
│       ├── extract_keypoints_right_sq.py  # MediaPipe: cadera/rodilla/tobillo der → (T, 3, 3)
│       └── extract_bar_features.py        # Tracking de barra con CSRT + velocidad/aceleración
│
├── keypoints/                             # Keypoints extraídos (.npy)
│   ├── dl/
│   │   ├── front/   (441 archivos)
│   │   ├── left/    (259 archivos)
│   │   └── right/   (258 archivos)
│   └── sq/
│       ├── front/   (316 archivos)
│       ├── left/    (245 archivos)
│       └── right/   (274 archivos)
│
├── training/                              # Pipeline de entrenamiento de CNN
│   ├── 01_preprocessing.py               # Carga .npy, validación, interpolación NaN, normalización espacial
│   ├── 02_temporal_normalization.py       # Interpolación a 170 frames (scipy lineal)
│   ├── 03_split_scale.py                  # Split estratificado 70/15/15 + StandardScaler
│   ├── 04_augmentation.py                 # Aumentación train x4 (traslación, escala, ruido, rotación, shift)
│   ├── 05_train_cnn.py                    # Entrenamiento PoseCNN (Adam, EarlyStopping, ReduceLROnPlateau)
│   ├── 06_cross_validation.py            # 5-Fold stratified CV con análisis de estabilidad
│   ├── 07_report.py                       # Reporte automático en Markdown con métricas y errores
│   ├── fix_csv.py                         # Reparación de CSV etiquetado corrupto (BOM UTF-8)
│   ├── train_utils.py                     # Clase PoseCNN, loops de entrenamiento, métricas, aumento
│   ├── RESUMEN_SESION.md                  # Resumen completo de resultados del entrenamiento
│   ├── data/
│   │   ├── preprocessed/raw_sequences.npz      # 412 secuencias limpias (T variable)
│   │   ├── normalized/sequences_170.npz        # 412 × 170 × 25 × 3
│   │   ├── split/split_data.npz + scaler.pkl   # Train/Val/Test + escalador
│   │   └── augmented/augmented_data.npz        # Train aumentado x4
│   ├── models/
│   │   ├── best_model.pth                      # PoseCNN entrenado (pesos)
│   │   └── history.npz                         # Historial de entrenamiento
│   └── results/
│       ├── report.md                           # Reporte de evaluación generado
│       ├── predictions.csv                     # Predicciones del test set
│       ├── misclassified.csv                   # Casos FP/FN con detalle
│       ├── train_val_loss.png                  # Curvas de entrenamiento
│       └── cross_validation_boxplot.png        # Boxplot de 5-Fold CV
│
├── etiquetado/                            # Datos etiquetados en CSV
│   ├── bp/
│   │   ├── bp_laterales.csv               # 1,483 filas (left+right combinados)
│   │   ├── bp_left.csv                    # 495 filas (vista izquierda)
│   │   └── bp_right.csv                   # 989 filas (vista derecha)
│   ├── dl/
│   │   └── etiquetado_dl_front.csv        # 442 filas (vista frontal)
│   └── sq/
│       ├── sq_front.csv                   # 270 filas (vista frontal)
│       ├── sq_laterales.csv               # 522 filas (laterales combinados)
│       ├── sq_left.csv                    # 275 filas (vista izquierda)
│       └── sq_right.csv                   # 248 filas (vista derecha)
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
├── models/                              # Modelos entrenados
│   ├── pose_landmarker_heavy.task        # MediaPipe Pose Landmarker
│   ├── bp_front_seg_v1.pt               # YOLO segmentación BP frontal
│   ├── bp_front_skeleton_v[3-6].pt      # YOLO pose BP frontal (varias versiones)
│   ├── bp_side_skeleton_v1.pt           # YOLO pose BP lateral
│   ├── bp_object_model.pt               # YOLO detección objetos BP
│   ├── dl_front_seg_v1.pt              # YOLO segmentación DL frontal
│   ├── sq_front_seg_v1.pt              # YOLO segmentación SQ frontal
│   ├── sq_front_skeleton_v1.pt          # YOLO pose SQ frontal v1
│   ├── sq_front_skeleton_v2.pt          # YOLO pose SQ frontal v2
│   ├── sq_front_skeleton_v3.pt          # YOLO pose SQ frontal v3
│   └── sq_front_skeleton_v4.pt          # YOLO pose SQ frontal v4
│
└── runs/pose/                           # Entrenamientos YOLO
    ├── train/
    ├── train-2/
    └── train-3/
```

---

## Dataset de Video

| Ejercicio | Vista Front | Vista Left | Vista Right | Total |
|-----------|-------------|------------|-------------|-------|
| BP (Press Banca) | 492 | 487 | 481 | 1,460 |
| DL (Peso Muerto) | 259 | 150 | 150 | 559 |
| SQ (Sentadilla) | 375 | 245 | 274 | 894 |
| **Total** | **1,126** | **882** | **905** | **~2,913** |

Los videos están en `dataset/<ejercicio>/<vista>/` (ignorados por git).

## Keypoints Extraídos

| Ejercicio | Vista | Archivos .npy | Forma | Landmarks |
|-----------|-------|---------------|-------|-----------|
| Peso Muerto | Front | 441 | (T, 25, 3) | 22 body + 2 hand centroids + bar-knee distance |
| Peso Muerto | Left | 259 | (T, 2, 3) | Shoulder + hip izquierdo |
| Peso Muerto | Right | 258 | (T, 2, 3) | Shoulder + hip derecho |
| Sentadilla | Front | 316 | (T, 22, 3) | 33 landmarks sin rostro |
| Sentadilla | Left | 245 | (T, 3, 3) | Cadera + rodilla + tobillo izquierdo |
| Sentadilla | Right | 274 | (T, 3, 3) | Cadera + rodilla + tobillo derecho |

## Datos Etiquetados (CSV)

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `etiquetado/bp/bp_laterales.csv` | 1,483 | BP lateral combinado (left+right) |
| `etiquetado/bp/bp_left.csv` | 495 | BP vista izquierda |
| `etiquetado/bp/bp_right.csv` | 989 | BP vista derecha |
| `etiquetado/dl/etiquetado_dl_front.csv` | 442 | DL vista frontal |
| `etiquetado/sq/sq_front.csv` | 270 | SQ vista frontal |
| `etiquetado/sq/sq_laterales.csv` | 522 | SQ lateral combinado |
| `etiquetado/sq/sq_left.csv` | 275 | SQ vista izquierda |
| `etiquetado/sq/sq_right.csv` | 248 | SQ vista derecha |

**Esquema común:** `video_id, movimiento, camara, archivo, resultado, label [1=valido/0=invalido], codigos [BP-10|SQ-01|DL-03], criterios, etiquetador, timestamp`

---

## Pipeline de Entrenamiento (CNN)

### Arquitectura: PoseCNN

```
Input: (batch, 3, 170, 25)  →  (N, C, T, K)
  └─ Conv2D(32, 3×3) → BatchNorm → ReLU
  └─ Conv2D(64, 3×3) → BatchNorm → ReLU
  └─ MaxPool2D(2×1) + Dropout2D(25%)
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
| 3 | `03_split_scale.py` | Split estratificado **70/15/15** (288 train, 62 val, 62 test) + StandardScaler |
| 4 | `04_augmentation.py` | Aumenta train **x4**: traslación (±5%), escala (±10%), ruido Gaussiano (σ=0.01), rotación (±10°), shift temporal (±3 frames) |
| 5 | `05_train_cnn.py` | Entrena PoseCNN: Adam(lr=0.001), BCELoss, ReduceLROnPlateau, EarlyStopping(patience=15), max 200 epochs |
| 6 | `06_cross_validation.py` | **5-Fold stratified CV** con pipeline completo por fold |
| 7 | `07_report.py` | Genera reporte Markdown automático con métricas, matriz de confusión, FP/FN, curvas |

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
- [x] Dataset de ~2,913 videos en 3 ejercicios × 3 vistas
- [x] Extracción de keypoints con MediaPipe para **sentadilla** (front, left, right)
- [x] Extracción de keypoints con MediaPipe para **peso muerto** (front: 25 kps, left/right: 2 kps)
- [x] Tracking de barra con CSRT (sentadilla)
- [x] **8 archivos CSV etiquetados** (~4,500+ filas) para los 3 ejercicios
- [x] Datos etiquetados integrados con pipeline de entrenamiento (DL front)
- [x] **Pipeline completo de preprocesamiento** (limpieza, normalización espacial/temporal, split, aumento)
- [x] **PoseCNN entrenada** para clasificación DL front (F1=0.904, ROC-AUC=0.925)
- [x] **5-Fold Cross-Validation** con análisis de estabilidad
- [x] **Reporte automático** con métricas, matriz de confusión, casos FP/FN
- [x] Modelos YOLO entrenados para **pose** (BP frontal v3-v6, BP lateral v1, SQ frontal v1-v4)
- [x] Modelos YOLO entrenados para **segmentación** (BP frontal v1, SQ frontal v1, DL frontal v1)
- [x] Visualizaciones de esqueleto con MediaPipe y YOLO para los 3 ejercicios
- [x] Herramienta de grabación multi-view sincronizada para BP
- [x] Herramienta interactiva de etiquetado de frame "Press" (`label_press_frame.py`)
- [x] Máquina de estados para validación de profundidad en sentadilla
- [x] Visualización combinada YOLO segmentación + MediaPipe skeleton (DL front)
- [x] Especificación técnica de la CNN (`Especificacion_CNN_Deadlift_Front.md`)

### 🔄 En Progreso
- [ ] Mejorar generalización de PoseCNN (reducir overfitting)

### ❌ Pendiente
- [ ] Extracción de keypoints para **press de banca** (bp) — `extract_keypoints/bp/` vacío
- [ ] Pipeline de entrenamiento para **BP** (front, left, right)
- [ ] Pipeline de entrenamiento para **DL lateral** (left, right)
- [ ] Pipeline de entrenamiento para **SQ** (front, left, right)
- [ ] Clasificación **multi-clase** (por tipo de falta, no solo binario)
- [ ] Arquitecturas alternativas (LSTM, Transformer)
- [ ] Hacer el pipeline reutilizable entre ejercicios/vistas (parametrizar)
- [ ] README / documentación para la tesis
- [ ] requirements.txt o pyproject.toml

---

## Convenciones del Código

- **Idioma**: Comentarios y variables en **español**
- **Estilo**: Sin tipado explícito, Python simple con numpy/opencv/mediapipe
- **Keypoints guardados como**: `.npy` con forma `(T, N, 3)` donde:
  - `T` = frames
  - `N` = landmarks seleccionados
  - `3` = (x, y, visibility)
- **Rutas de modelos**: Relativas desde el script (`../../models/...`)
- **Parámetros de entrenamiento YOLO**: img=960, batch=8, AdamW(lr=0.001), pose_weight=12.0
- **Parámetros de entrenamiento CNN**: 170 frames, 25 keypoints, Adam(lr=0.001), EarlyStopping(patience=15), ReduceLROnPlateau

---

## Próximos Pasos (Hoja de Ruta)

1. Extracción de keypoints para **press de banca** (bp) con MediaPipe
2. Mejorar generalización de PoseCNN (regularización, más datos, hyperparameter tuning)
3. Pipeline de entrenamiento para las demás vistas (BP front/lateral, DL lateral, SQ)
4. Clasificación multi-clase por tipo de falta (no solo válido/nulo)
5. Probar arquitecturas LSTM/Transformer para comparar con CNN
6. Parametrizar el pipeline para que sea reutilizable entre ejercicios/vistas
7. Evaluar y documentar resultados para la tesis
