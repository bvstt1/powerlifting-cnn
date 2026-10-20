# Primer entrenamiento SegCNN temporal BP_FRONT — V1

## Alcance, protocolo y fuentes

Se entrenaron **dos redes CNN temporales 1D independientes** sobre características
geométricas preexistentes, sin nueva inferencia YOLO. Solo se usaron secuencias
`train` y `validation` de `bp_front_features_common_cohort.csv`; ningún NPZ Test
se cargó ni se evaluó. No se modificaron el extractor, las secuencias, labels,
particiones ni el checkpoint histórico PoseCNN. El baseline PoseCNN para comparación
final deberá reiniciarse y reentrenarse sobre el split común; según la auditoría
externa aportada, 19 videos del Test común habían estado en el Train histórico
Pose-only. No existe clasificador BP_SIDE combinado.

Entradas: `segmentation/bp/front/temporal_features_v1/`. Salidas reproducibles:
`training/bp_front_segcnn_v1/`. SHA-256 de fuentes, split, esquema, manifiesto
original y **las 561 secuencias abiertas de Train/Validation** registrados en
`source_manifest.json` (`test_sequences_opened=0`). Configuración previa a
entrenar: `training_config.json`.

## Verificaciones iniciales (breves, antes de entrenar)

1. El inventario contiene 663 IDs únicos: **661 incluidos** (Train=462,
   Validation=99, Test=100), `bp_349` fuera por RAW inválido y `bp_352`
   excluido como duplicado de `bp_351` aunque tiene secuencia extraída.
   662 NPZ extraídos, F=38. Labels 1=válido/0=inválido; Train=279/183,
   Validation=60/39 (válido/inválido). Secuencias de desarrollo verificadas:
   `(T,38)` con T registrado, frame_index contiguo y `availability ==
   isfinite(features)`; sin Inf ni secuencias faltantes.
2. `QUALITY_WARNING` aparece en **485/661** incluidos: advertencias en el
   manifiesto RAW, *no exclusiones*. Motivos (pueden coincidir):
   `multiple_instances` 468, `main_track_below_raw_detection` 85,
   `raw_missing_frames` 20, `bench_identity_fallback` 7 y `bar_handoff` 1.
   Denotan ambigüedad/faltantes por frame o tracking histórico; el extractor
   conserva las detecciones ausentes como NaN+máscara y no usa tracking V3.
   No hay fallo crítico de integridad; se mantuvieron los 485.
3. `profile_ambiguous` es **constante (=1) en Train**, detectado usando solo
   Train: eliminado de ambos tensores de entrada y retenido en el esquema
   original como metadato. No se usó Test para seleccionar canales.
4. El esquema original clasificó `q_bar_associated`, `q_bar_unverified`,
   `q_bar_multiple` y `q_projected` como `QUALITY` con
   `dependencia_pose=false`. **Interpretación corregida aquí sin alterar
   el esquema original ni los NPZ:** son indicadores QUALITY **pose-assisted**
   porque la selección BAR utiliza muñecas y el perfil central hombros; una
   fórmula binaria final no elimina esa dependencia. La rama SegCNN FULL y
   su ablación siguen siendo pose-assisted.

## Remuestreo, valores faltantes y escalado

Grid de 170 pasos `linspace(0,T_original−1,170)`, idéntico al utilizado
en el pipeline PoseCNN frontal. Para una posición entera se copia el frame;
para una posición fraccionaria se interpolan dos frames originales **solo si**
ambos observan el canal. BAR y dinámica requieren el mismo track-ID y
`prov_bar=1` en ambos extremos; separación requiere igual identidad y
procedencia directa/proyectada. Si no se cumplen esas condiciones, NaN y
disponibilidad=0. Indicadores binarios y estados de procedencia se asignan
por vecino temporal más cercano, sin interpolar. Esto no remuestrea máscaras
de píxeles ni fabrica movimientos a través de cambios de identidad.

Para cada variante, sobre **Train remuestreado y valores observados solamente**:
mediana de imputación, media y desviación estándar por canal; desviación
mínima 1e-6. Se imputan los faltantes con mediana, luego se escala, pero se
concatena además la **máscara binaria de disponibilidad por canal**. Por
separado se añaden tres estados BAR (observada/no-verificada/ausente) y, en
FULL, dos de separación (directa/proyectada). No se incorporaron estimaciones
temporales `est_*`. Preprocesamiento, índices y orden de canales quedaron
guardados en `*/channels_and_train_preprocessing.json`.

## Experimentos congelados

| Experimento | Features reales | Máscaras | Procedencia | `C` total |
|---|---:|---:|---:|---:|
| FULL geometry | 37 (las 38 menos `profile_ambiguous`) | 37 | 5 | **79** |
| SIN relación barra–silueta | 31 | 31 | 3 BAR | **65** |

En la ablación se retiraron **antes de entrenar** las features
`silhouette_profile_y_norm`, `sep_signed_norm`, `sep_coverage`,
`sep_extrap_dist_norm`, `d_sep_dt`, `q_projected`, sus máscaras correspondientes
y los dos bits `sep_directa/sep_proyectada`. `profile_ambiguous` ya estaba
excluida por constante. Se conservaron los canales de BAR, PERSON, BENCH,
dinámica de BAR y los demás QUALITY. Sigue habiendo dependencia de Pose.

## Arquitectura y entrenamiento

Entrada por video `(C,170)`; `Conv1D(C→32,k5)→BN→ReLU`,
`Conv1D(32→64,k5)→BN→ReLU→MaxPool1D(2)→Dropout(0,20)`,
`Conv1D(64→128,k3)→BN→ReLU→AdaptiveAvgPool1D(1)`,
`Linear(128→64)→ReLU→Dropout(0,40)→Linear(64→1)`.
Logit binario directo, `BCEWithLogitsLoss`, sigmoid solo para evaluación.
Parámetros entrenables: **56.449 FULL / 54.209 ablación**.

PyTorch/CUDA disponible en RTX 5060 Laptop. AdamW(lr=0,001, wd=1e-4),
batch 32, máximo 200 épocas, ReduceLROnPlateau(factor=0,5, patience=5),
early stopping con patience=15, AMP CUDA. Semillas predefinidas **42 y 1337**
para cada variante. Checkpoint y semilla elegidos **por menor loss en
Validation**; umbral binario **0,5 prefijado** (sin optimización de umbral).
La clasificación y métricas son **por video**, no por frame.

## Resultados en Train y Validation

| Experimento / seed | Época mejor / ejecutadas | Loss Train / Val | Train F1 | Val Acc | Val F1 | Val Macro-F1 | Val Balanced Acc | Val ROC-AUC | Val PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL 42 | 8 / 23 | 0,2322 / 0,2779 | 0,9134 | 0,8687 | 0,8889 | 0,8642 | 0,8692 | 0,9498 | 0,9639 |
| **FULL 1337** | **11 / 26** | **0,1842 / 0,2738** | **0,9492** | **0,8687** | **0,8943** | **0,8605** | **0,8558** | **0,9500** | **0,9664** |
| **SIN 42** | **20 / 35** | **0,1242 / 0,2819** | **0,9680** | **0,9192** | **0,9355** | **0,9137** | **0,9064** | **0,9462** | **0,9580** |
| SIN 1337 | 10 / 25 | 0,1948 / 0,2872 | 0,9345 | 0,8889 | 0,9091 | 0,8831 | 0,8814 | 0,9462 | 0,9604 |

**Resultados por clase de checkpoints elegidos (Validation):**

| Modelo | Precisión válido | Sensibilidad válido | Especificidad inválido | Precisión inválido | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|
| FULL seed 1337 | 0,8730 | 0,9167 | 0,7949 | 0,8611 | **31 / 8 / 5 / 55** |
| SIN seed 42 | 0,9062 | 0,9667 | 0,8462 | 0,9429 | **33 / 6 / 2 / 58** |

Referencia clase mayoritaria definida por Train (predecir siempre válido):
Validation Accuracy=0,6061, F1=0,7547, Macro-F1=0,3774,
Balanced Acc=0,5, AUC=0,5, AP=0,6061; TN/FP/FN/TP=0/39/0/60.
En las dos semillas, F1 Validation osciló entre 0,8889–0,8943
para FULL y 0,9091–0,9355 para SIN; ROC-AUC entre 0,9498–0,9500
y 0,9462–0,9462, respectivamente. Dos semillas no estiman una
distribución de incertidumbre estable.

## Comparación y análisis de errores

Los dos checkpoints elegidos discrepan en **9/99** videos Validation;
ambos erran 6, FULL falla adicionalmente 7, SIN falla adicionalmente 2
(`validation_paired_predictions.csv` y
`paired_comparison_validation.json`). La ablación mejora 5,05 puntos de
Accuracy y 5,32 puntos de Macro-F1, mientras FULL tiene 0,38 puntos más
de ROC-AUC y 0,84 de PR-AUC. **Es evidencia exploratoria, no prueba causal**
de que el perfil siempre perjudique: muestra pequeña, solo dos semillas,
selección por loss, sin ajuste de umbral. Las 71 muestras Validation con
QUALITY_WARNING permanecieron incluidas.

Distribución de probabilidad predicha para válido (checkpoint elegido):
en FULL mediana inválidos=0,035 / válidos=0,986; en SIN
inválidos=0,011 / válidos=0,983. Gráficos por semilla en
`*/seed_*/validation_probabilities.png`, matrices PNG y `.npy`, y
predicciones CSV con IDs para revisar FP/FN. Por video en las curvas se
observa que el entrenamiento sigue mejorando tras la mejor época de
Validation: **sobreajuste moderado**. Brecha loss Train–Val 0,0896 para
FULL elegido y 0,1577 para SIN elegido; early stopping evita usar épocas
posteriores (hasta 26 y 35, respectivamente). No se examinó Test.

## Mejor checkpoint y recomendación

- FULL: `full/seed_1337/best_validation_loss.pt`, época 11, Val loss 0,2738.
- SIN: `sin_relacion/seed_42/best_validation_loss.pt`, época 20, Val loss
  0,2819. Para el primer baseline orientado a F1 y balanced accuracy en
  Validation, la ablación es la candidata experimental; para menor loss,
  FULL es marginalmente preferible. **No se declara ganadora final en Test**.

Se puede avanzar a **reentrenar PoseCNN desde cero sobre esta misma cohorte
y split**, y después diseñar fusión BP_FRONT. Antes de cualquier evaluación
final se debe congelar una política única de selección y el protocolo de
fusión sin observar Test. Mantener las separaciones 2D como descriptores de
silueta, no evidencia de contacto físico con el pecho; `profile_ambiguous`
vale 1 en todo Train. Pose ayuda a construir parte de esta rama; una mejora
de SegCNN sobre PoseCNN no demuestra independencia modal.
