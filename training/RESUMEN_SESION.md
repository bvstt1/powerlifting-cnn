# Resumen de Sesión: Pipeline CNN para Deadlift Frontal

**Fecha:** 2026-07-22
**Objetivo:** Implementar pipeline completo de CNN binaria para clasificar deadlift frontal como válido (1) o inválido (0) usando keypoints de MediaPipe.

---

## Archivos Creados

### Pipeline de Preprocesamiento

| Archivo | Descripción |
|---|---|
| `training/01_preprocessing.py` | Carga keypoints `.npy`, valida integridad, interpola NaN pequeños, normalización espacial (centrado en cadera, escalado por torso). Salida: `raw_sequences.npz` (objetos con T variable) |
| `training/02_temporal_normalization.py` | Re-muestrea todas las secuencias a 170 frames usando interpolación lineal. Salida: `sequences_170.npz` con shape `(N, 170, 25, 3)` |
| `training/03_split_scale.py` | División estratificada 70/15/15 + StandardScaler (solo train). Salida: `split_data.npz` + `scaler.pkl` |
| `training/04_augmentation.py` | Aumenta train x4 con traslación, escalado, ruido, rotación, desplazamiento temporal. Salida: `augmented_data.npz` |

### Entrenamiento y Evaluación

| Archivo | Descripción |
|---|---|
| `training/05_train_cnn.py` | Arquitectura PoseCNN (Conv2D → BN → ReLU → MaxPool → Dropout → GlobalAvgPool → Dense). Trackea loss/accuracy por época, genera curvas `train_val_loss.png`, analiza overfitting automáticamente |
| `training/06_cross_validation.py` | Validación cruzada 5-Fold estratificada. Entrena modelo completo en cada fold. Genera tabla con métricas, boxplot, análisis de estabilidad |
| `training/07_report.py` | Genera informe Markdown automático con: resumen dataset, arquitectura, métricas, matriz de confusión, análisis de errores, FP/FN con video_id |

### Utilidades

| Archivo | Descripción |
|---|---|
| `training/fix_csv.py` | Script único para reparar CSV de etiquetas dañado (comillas dobles) |

---

## Dataset Final

| Métrica | Valor |
|---|---|
| Keypoints totales | 441 (`keypoints/dl/front/dl_001.npy` a `dl_441.npy`) |
| Con etiqueta en CSV | 440 (1 faltante: `dl_350`) |
| Procesados (válidos) | 412 (después de filtro por NaN) |
| Válidos (label=1) | 256 (62.1%) |
| Inválidos (label=0) | 156 (37.9%) |
| Shape por secuencia | `(T, 25, 3)` → normalizado a `(170, 25, 3)` |

**CSV actualizado:** `etiquetado/dl/etiquetado_dl_front_fixed.csv` (440 filas, 264 válidos + 176 inválidos originales, 28 eliminados por NaN)

**Videos eliminados por exceso de NaN:** dl_200, dl_209, dl_247, dl_248, dl_317-335, dl_337, dl_339, dl_340, dl_403, dl_429, dl_430 (28 total)

---

## Arquitectura del Modelo

```
Entrada: (batch, 3, 170, 25)  # (C, T, K) después de permute
  │
  ├─ Conv2D(32, 3×3) → BatchNorm → ReLU
  ├─ Conv2D(64, 3×3) → BatchNorm → ReLU
  ├─ MaxPool2D(2×1) + Dropout(25%)
  ├─ Conv2D(128, 3×3) → BatchNorm → ReLU
  ├─ GlobalAvgPool2D(1×1)
  │
  ├─ Dense(64) → ReLU → Dropout(50%)
  └─ Dense(1) → Sigmoid
```

### Hiperparámetros

| Parámetro | Valor |
|---|---|
| Learning Rate | 0.001 |
| Optimizer | Adam (con ReduceLROnPlateau, factor 0.5, patience 5) |
| Loss | Binary Cross-Entropy |
| Batch Size | 16 |
| Early Stopping | Patience 15 |
| Data Augmentation | 3x (traslación ±5%, escalado ±10%, ruido σ=0.01, rotación ±10°, shift temporal ±3) |
| Normalización temporal | 170 frames |
| Normalización espacial | Centrado en cadera, escalado por torso |

---

## Resultados Finales

### Test Set (62 muestras)

| Métrica | Valor |
|---|---|
| Accuracy | **0.8871** |
| Precision | **0.9429** |
| Recall | **0.8684** |
| F1-score | **0.9041** |
| ROC-AUC | **0.9254** |

### Matriz de Confusión

| | Pred: Inválido | Pred: Válido |
|---|---|---|
| **Real: Inválido** | 22 (TN) | 2 (FP) |
| **Real: Válido** | 5 (FN) | 33 (TP) |

### 5-Fold Cross Validation (promedio)

| Fold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 1 | 0.8313 | 0.8167 | 0.9423 | 0.8750 | 0.8635 |
| 2 | 0.6627 | 0.6456 | 1.0000 | 0.7846 | 0.8768 |
| 3 | 0.6951 | 0.6711 | 1.0000 | 0.8031 | 0.9070 |
| 4 | 0.7683 | 0.7353 | 0.9804 | 0.8403 | 0.8925 |
| 5 | 0.6829 | 0.6712 | 0.9608 | 0.7903 | 0.9152 |
| **Promedio** | **0.7281** | **0.7080** | **0.9767** | **0.8187** | **0.8910** |
| **Std** | ±0.063 | ±0.062 | ±0.023 | ±0.034 | ±0.019 |

### Análisis de Estabilidad

- **ROC-AUC:** 0.891 ± 0.019 (CV=2.1%) → Estable ✅
- **F1:** 0.819 ± 0.034 (CV=4.2%) → Estable ✅
- **Accuracy:** 0.728 ± 0.063 (CV=8.6%) → Moderado ⚠️
- **Recall:** Muy alto (97.7%) pero probablemente sesgado a predecir "válido"
- **Conclusión:** El modelo generaliza aceptablemente pero con cierta variabilidad entre splits

### Análisis de Curvas

- **Diagnóstico automático:** OVERFITTING
- **Mejor época:** 10 (val_loss mínimo: 0.3027)
- **Sobreajuste detectado desde:** época ~3
- Train loss continuó bajando mientras val_loss subía ligeramente después de época 10
- Early stopping en época 25 (15 épocas sin mejora)

### Errores en Test

**Falsos Positivos (5):** predichos inválido cuando eran válidos

| video_id | Probabilidad (ser válido) |
|---|---|
| dl_267 | 0.1604 |
| dl_313 | 0.4574 |
| dl_427 | 0.2787 |
| dl_233 | 0.2116 |
| dl_354 | 0.0576 |

**Falsos Negativos (2):** predichos válido cuando eran inválidos

| video_id | Probabilidad (ser válido) |
|---|---|
| dl_441 | 0.9970 |
| dl_253 | 0.6482 |

---

## Archivos de Salida Generados

```
training/
├── data/
│   ├── preprocessed/raw_sequences.npz      # 412 secuencias limpias
│   ├── normalized/sequences_170.npz        # 412 × 170 × 25 × 3
│   ├── split/split_data.npz + scaler.pkl   # Train/Val/Test escalados
│   └── augmented/augmented_data.npz        # Train aumentado x4
├── models/
│   ├── best_model.pth                      # Mejor modelo (época 10)
│   └── history.npz                         # Loss/Accuracy por época
├── results/
│   ├── train_val_loss.png                  # Curvas de entrenamiento
│   ├── cross_validation_boxplot.png        # Boxplot 5-Fold CV
│   ├── misclassified.csv                   # 5 FP + 2 FN con detalles
│   └── report.md                           # Informe completo
├── 01_preprocessing.py
├── 02_temporal_normalization.py
├── 03_split_scale.py
├── 04_augmentation.py
├── 05_train_cnn.py
├── 06_cross_validation.py
├── 07_report.py
└── fix_csv.py
```

---

## Pendiente para Próxima Sesión

1. **Entrenar modelos para BP y DL lateral** — adaptar pipeline a otras vistas
2. **Mejorar overfitting** — probar más regularización, class weights, o focal loss
3. **Probar arquitecturas alternativas** — LSTM, Transformer, o CNN 1D
4. **Integrar etiquetas multi-clase** — clasificar por tipo de falta (DL-01 a DL-07)
5. **Hacer el código reutilizable** — refactorizar para que funcione para cualquier ejercicio/vista
6. **Subir a GitHub** — documentar README con instrucciones de uso
