# REPORTE TOTAL — Juez Virtual de Powerlifting: Redes PoseCNN (Skeletons)
**Proyecto tesis — Clasificación binaria válido (1) / nulo (0)**
**Fecha: 2026-10-07 | Fuente verdad: `training/results/`, `training/models/`, `AGENTS.md`, `RESUMEN_SESION.md`**

> Documento diseñado para importar en un generador de PowerPoints. Cada sección `## Slide N` = una diapositiva sugerida. Tablas listas para copiar.

---

## Slide 1 — Portada y objetivo

* **Título:** Juez Virtual de Powerlifting con Visión por Computador — Resultados PoseCNN
* **Objetivo:** Clasificar automáticamente levantamientos (Squat, Bench Press, Deadlift) desde skeletons, sin usar imagen cruda.
* **Pipeline:** Detección esqueletos (YOLO / MediaPipe) → keypoints `.npy (T,N,3)` → etiquetado CSV → preprocessing + normalización espacial/temporal (170 frames) → split por evento físico → scaler fit solo train → aumento solo train → PoseCNN.
* **Estado:** Entrenados y evaluados: `DL front`, `DL left`, `DL right`, `DL side`, `BP front`. Todos PoseCNN sobre skeletons.

---

## Slide 2 — Arquitectura común PoseCNN

```
Input: (batch, 3, 170, K)  # (C,T,K), C=3 (x,y,conf), K según vista
  Conv2D(32,3x3) → BN → ReLU
  Conv2D(64,3x3) → BN → ReLU
  MaxPool2D(2x1) + Dropout2D (25% histórico / 40% V2-V3)
  Conv2D(128,3x3) → BN → ReLU
  GlobalAvgPool → Dense(64) → ReLU → Dropout(50%) → Dense(1) → Sigmoid
```

| Hiperparámetro | Valor común |
|---|---|
| Loss / Optimizer | BCELoss / Adam lr=0.001, weight_decay 1e-4 |
| Batch / Épocas / EarlyStopping | 16 / máx 200 / patience 8 (V2/V3) o 15 (histórico) |
| Scheduler | ReduceLROnPlateau factor 0.5 patience 5 |
| Temporal | 170 frames, interpolación lineal scipy (no zero-padding) |
| Scaler | StandardScaler reshape (N, T*K*C), fit solo train original |
| Aumento | Solo train x3-x4: shift ±5%, escala 0.9-1.1, ruido σ=0.01, rot ±10°, shift temporal ±3 |
| **Excepción BP front** | MLP 128→64→1 (resto Dense 64→1). Resto idéntico |

---

## Slide 3 — Datos de entrada por modelo

| Modelo | Fuente keypoints | Forma | N efectivo (válidos/nulos) | Split |
|---|---|---:|---|---|
| DL front | MediaPipe 22 body + 2 hand centroids + bar-knee dist = 25 kps | (170,25,3) | 412 (256 / 156 = 62.1% válidos) | Test 62 |
| DL left V3 vigente | YOLO lateral shoulder+hip = 2 kps | (170,2,3) | 403 (222 / 181) | 283 / 60 / 60 |
| DL right V3 vigente | YOLO lateral 2 kps | (170,2,3) | 411 (226 / 185) | 287 / 62 / 62 |
| DL side V3 vigente | right 001-413 + left 414-825, K=2 | (170,2,3) | 814 (448 / 366) | 570 / 122 / 122 |
| BP front V1 | YOLO `bp_front_skeleton_v8.pt`, 6 kps (hombros/codos/muñecas) | (170,6,3) | 662 (663 menos duplicado exacto bp_352==bp_351; 400 / 263 ≈60/40%) | 462 / 100 / 100 |

Notas metodológicas DL lateral: hueco físico real evento 310 (`physical_event()` en `training/training_config.py` única fuente verdad). Split común left/right por evento físico, sin leakage. V2 = fix metodológico. V3 = V2 + auditoría manual 41 labels (15 left + 18 right cambiados, evento 302 descartado `manual_exclude=1`), 9 discordancias válidas restantes (171,172,174,277,278,282,283,285,351).

---

## Slide 4 — Resultados TEST (tabla comparativa principal)

| Modelo | Acc | Prec | Rec | F1 | AUC | TN / FP / FN / TP | N test |
|---|---:|---:|---:|---:|---:|---|---:|
| DL front | 0.8871 | 0.9429 | 0.8684 | **0.9041** | 0.9243 (0.9254 en RESUMEN) | 22 / 2 / 5 / 33 | 62 |
| DL left V2 | 0.7705 | 0.8519 | 0.6970 | 0.7667 | 0.8820 | 24 / 4 / 10 / 23 | 61* |
| DL right V2 | 0.8226 | 0.9000 | 0.7714 | 0.8308 | 0.8942 | 24 / 3 / 8 / 27 | 62* |
| DL side V2 | 0.7967 | 0.8308 | 0.7941 | 0.8120 | 0.8682 | 44 / 11 / 14 / 54 | 123 |
| DL left V3 | 0.8167 | 0.8235 | 0.8485 | **0.8358** | 0.8923 | 21 / 6 / 5 / 28 | 60 |
| DL right V3 | 0.8226 | 0.8108 | 0.8824 | **0.8451** | 0.8645 | 21 / 7 / 4 / 30 | 62 |
| DL side V3 | 0.7869 | 0.7595 | 0.8955 | 0.8219 | 0.8179 | 36 / 19 / 7 / 60 | 122 |
| BP front | 0.8400 | 0.8548 | 0.8833 | **0.8689** | **0.9354** | 31 / 9 / 7 / 53 | 100 |

*V2 paired n=61 eventos comunes: left F1 0.7667/AUC 0.8820, right F1 0.8254/AUC 0.8932. V3 paired n=60: left=right F1 0.8358; AUC left 0.8923 > right 0.8627.
Fuentes: `training/results/dl/front/report.md`, `training/results/dl/{left,right,side}/baseline_v2/report.md`, `.../baseline_v3_manual_audit/report.md`, `training/results/bp/front/posecnn/bp_front_pose_metrics.json`.

**Lectura ejecutiva:**
1. Mejor F1 puntual: DL front (0.904). Mejor AUC: BP front (0.935).
2. Tarea lateral K=2 más difícil; `side` no supera vistas puras (recall alto 0.896 pero precisión 0.760, 19 FP).
3. V3 mejora left +6.9 pp F1 por limpieza labels; right estable.

---

## Slide 5 — Validación cruzada 5-Fold (estabilidad, solo development)

| Modelo | Acc CV | F1 CV | AUC CV |
|---|---:|---:|---:|
| DL front | 0.728 ±0.063 | 0.819 ±0.034 | 0.891 ±0.019 (recall CV 0.977 sesgado a válido) |
| DL left V2 | 0.825 ±0.052 | 0.8335 ±0.0508 | 0.8964 ±0.0353 |
| DL right V2 | 0.797 ±0.055 | 0.8218 ±0.0453 | 0.8639 ±0.0611 |
| DL side V2 | 0.768 ±0.056 | 0.7815 ±0.0602 | 0.8559 ±0.0439 |
| DL left V3 | 0.827 ±0.044 | 0.8445 ±0.0395 | 0.8924 ±0.0403 |
| DL right V3 | 0.819 ±0.051 | 0.8399 ±0.0413 | 0.8727 ±0.0521 |
| DL side V3 | 0.791 ±0.056 | 0.8108 ±0.0446 | 0.8757 ±0.0476 |
| BP front | **0.866 ±0.038** | **0.891 ±0.029** | **0.944 ±0.021** |

Fuentes: `RESUMEN_SESION.md`, `results/dl/baseline_v2_cv/report.md`, `results/dl/baseline_v3_manual_audit/v2_v3_comparison.json`, `results/bp/front/posecnn/bp_front_pose_cv_results.csv`.
**Lectura:** BP front = más robusto en CV. DL front = mayor gap test→CV (overfitting). V3 reduce desviación vs V2.

---

## Slide 6 — V2 vs V3 (Deadlift lateral, misma arquitectura)

| Perfil | Test F1 V2→V3 | Test AUC V2→V3 | CV F1 V2→V3 | CV AUC V2→V3 |
|---|---:|---:|---:|---:|
| left | 0.7667 → 0.8358 | 0.8820 → 0.8923 | 0.8335±0.051 → 0.8445±0.040 | 0.8964±0.035 → 0.8924±0.040 |
| right | 0.8308 → 0.8451 | 0.8942 → 0.8645 | 0.8218±0.045 → 0.8399±0.041 | 0.8639±0.061 → 0.8727±0.052 |
| side | 0.8120 → 0.8219 | 0.8682 → 0.8179 | 0.7815±0.060 → 0.8108±0.045 | 0.8559±0.044 → 0.8757±0.048 |

**No interpretar como mejora arquitectónica:** mismo PoseCNN e hiperparámetros; cambia ground truth auditado + descarte 302 + nuevo split/folds (`cv_event_folds.csv` vs `cv_event_folds_v3.csv`).

---

## Slide 7 — Diagnóstico entrenamiento y calidad

| Modelo | Mejor época / final | Diagnóstico curvas |
|---|---|---|
| DL front | 10 (RESUMEN) / 20 (report, val_loss 0.328) | OVERFITTING desde ~época 2-3 |
| DL left V2 / right V2 / side V2 | 16/24, 19/27, 10/18 | OVERFITTING |
| DL left V3 / right V3 / side V3 | 13/21, 13/21, 7/15 | NORMAL / NORMAL / POSIBLE OVERFITTING |
| BP front | 68 / 83 (val_loss 0.184, test_loss 0.462) | Gap val→test F1 0.944→0.869, overfitting moderado |

Calidad DL (descriptivo, no es input): 66 QUALITY_OK, 11 WARNING, 5 EXCLUDED sobre 82 muestras auditadas. Test V3 WARNING casi vacío (left 0, right 1, side 1: `event_301/right` correcta en right, falla en side). Fuente: `manual_quality_audit.csv`, `quality_subset_metrics.json`.

---

## Slide 8 — Qué falta por entrenar (roadmap)

### A. En proceso / próxima prioridad
1. **Mejorar BP front skeleton (PoseCNN v2):** más regularización / tuning para cerrar gap val→test (0.93→0.84 acc). Modelo actual `bp_front_pose_best_model.pth` como baseline.
2. **Entrenar BP side skeleton (PoseCNN):** YOLO `bp_side_skeleton_v1.pt` (8 kps lateral) → keypoints `keypoints/bp/{left,right}` + `bp_left.csv` (495) / `bp_right.csv` (989) / `bp_laterales.csv` (1483). Reutilizar pipeline `01-05` con perfil `bp_side` (K=8, T=170).
3. **Entrenar BP side segmentación:** modelo `bp_front_seg_v1.pt` / segmentación lateral + features de barra.
4. **Combinación front + side (BP):** fusión tardía (late fusion) o multivista: promediar probs front/side o MLP conjunto; mismo protocolo paired por evento como DL V2/V3.

### B. Squat (SQ) pendiente
5. **SQ front PoseCNN:** keypoints existentes `keypoints/sq/front` (316, T×22×3, MediaPipe 33 sin rostro) + `sq_front.csv` (270). Crear perfil `sq_front`.
6. **SQ side PoseCNN:** `keypoints/sq/left` (245, 3 kps cadera/rodilla/tobillo izq) + `right` (274) + `sq_left.csv` (275) / `sq_right.csv` (248) / `sq_laterales.csv` (522). Perfil `sq_side` análogo a `dl_side`.
7. **Agregar bar tracking a SQ (y evaluar en BP/DL):** extensor existente `extract_keypoints/sq/extract_bar_features.py` (CSRT + velocidad/aceleración) y `display/sq/bar_grafic.py`. Añadir como canal/feature extra o rama auxiliar a PoseCNN.

### C. Transversal tesis
8. Clasificación multi-clase por código de falta (BP-10, SQ-01, DL-03…), no solo binario.
9. Arquitecturas temporales alternativas (LSTM / Transformer) vs PoseCNN con mismos splits/folds para comparación justa.
10. Extender perfiles reutilizables a BP y SQ + `requirements.txt` + README tesis.

---

## Slide 9 — Comparación lista para otro documento

Copiar esta tabla al documento comparativo externo:

| Tarea | Modelo doc. interno | F1 test | AUC test | F1 CV | AUC CV | Nota vs externo |
|---|---|---:|---:|---:|---|---|
| DL front skeleton | PoseCNN K=25 | 0.9041 | 0.9243 | 0.8187±0.034 | 0.8910±0.019 |  |
| DL left skeleton V3 | PoseCNN K=2 | 0.8358 | 0.8923 | 0.8445±0.040 | 0.8924±0.040 |  |
| DL right skeleton V3 | PoseCNN K=2 | 0.8451 | 0.8645 | 0.8399±0.041 | 0.8727±0.052 |  |
| DL side skeleton V3 | PoseCNN K=2 | 0.8219 | 0.8179 | 0.8108±0.045 | 0.8757±0.048 |  |
| BP front skeleton | PoseCNN K=6 YOLOv8 | 0.8689 | 0.9354 | 0.8914±0.029 | 0.9437±0.021 |  |
| BP side skeleton | — pendiente | — | — | — | — | rellenar al entrenar |
| BP side seg / fusión | — pendiente | — | — | — | — | rellenar al entrenar |
| SQ front / side | — pendiente | — | — | — | — | rellenar al entrenar |

Advertencias: (1) V2→V3 no es cambio de red. (2) Históricos pre-V2 (`results/dl/{left,right,side}/report.md`) obsoletos por mapping/leakage. (3) Comparar siempre test + CV, no solo test.

---

## Apéndice — Rutas de evidencia (para auditoría)

* DL front: `training/results/dl/front/report.md`, `predictions.csv`, `misclassified.csv`, `training/RESUMEN_SESION.md`
* DL V2: `training/results/dl/{left,right,side}/baseline_v2/report.md`, `training/results/dl/baseline_v2/paired_metrics.json`, `training/results/dl/baseline_v2_cv/`
* DL V3: `training/results/dl/{left,right,side}/baseline_v3_manual_audit/report.md`, `training/results/dl/baseline_v3_manual_audit/{paired_metrics.json,v2_v3_comparison.json,paired_predictions.csv}`, `training/results/dl/baseline_v3_manual_audit_cv/`
* BP front: `training/results/bp/front/posecnn/{bp_front_pose_metrics.json,bp_front_pose_cv_results.csv,bp_front_pose_test_predictions.csv}`, `training/models/bp/front/posecnn/{bp_front_pose_config.json,bp_front_pose_best_model.pth}`, `training/data/bp/front/bp_front_pose_audit_summary.json`
* Config mapping: `training/training_config.py::physical_event()`, `training/test_event_mapping.py`
* Labels: `etiquetado/dl/dl_{left,right,side}.csv` (+ `*_pre_manual_audit.csv`), `etiquetado/bp/bp_{front,left,right}.csv`, `etiquetado/sq/sq_{front,left,right}.csv`
