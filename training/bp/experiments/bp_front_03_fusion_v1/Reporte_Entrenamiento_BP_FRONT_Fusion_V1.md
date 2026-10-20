# Primera fusión PoseCNN + SegCNN temporal BP_FRONT — V1

**Solo BP_FRONT. Test reservado: ningún `.npy`/`.npz` de Test fue cargado
para inferencia, métricas o selección (`test` solo aparece en conteos del
manifiesto). No se reentrenó PoseCNN ni SegCNN; no se modificó ninguna
fuente (`fusion_integrity_verification.json`: 8/8 hashes base intactos).**

## 1. Punto de partida y reproducibilidad (FASE A)

Checkpoints congelados: PoseCNN Common `seed_42/best_validation_loss.pt`,
SegCNN FULL `seed_1337/...`, SegCNN SIN `seed_42/...`, cada uno con su
scaler/estadísticas Train y sus CSV de Validation (99 videos, umbral 0,5).
Recarga verificada en GPU RTX 5060 (`base_reproducibility.json`):
Pose diff máx 1,6e-6; SegCNN diff **0,0** replicando su inferencia AMP con
batch 128 (en FP32 difiere ~5e-4 por redondeo AMP: diferencia numérica
documentada, predicciones idénticas en los 99). Embeddings reales:
**Z_pose (N,128)** vía `extract_embeddings` (AdaptiveAvgPool2D pre-MLP) y
**Z_seg (N,128)** vía adaptador `temporal(x).flatten(1)` (AdaptiveAvgPool1D
pre-cabeza); se comprobó que el adaptador no altera las probabilidades
originales. Concatenación **(N,256)**. Cohorte 462/99/100 intacta;
preprocesamientos reproducidos bit a bit contra los JSON congelados
(canales 79/65, medianas/medias/std Train).

## 2. Fusión tardía 50/50 (Experimento A, sin entrenamiento)

`p = 0,5·p_pose + 0,5·p_seg`, umbral 0,5 fijo, mismos 99 IDs:

| Modelo | Acc | Prec | Rec | F1 | Macro-F1 | BalAcc | ROC-AUC | PR-AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|---|
| A1 Pose+FULL | 0,9192 | 0,8824 | 1,0000 | 0,9375 | 0,9116 | 0,8974 | 0,9679 | 0,9773 | 31/8/0/60 |
| A2 Pose+SIN | 0,9293 | 0,8955 | 1,0000 | 0,9449 | 0,9231 | 0,9103 | 0,9594 | 0,9681 | 32/7/0/60 |

Ambas eliminan los falsos negativos de Validation (FN=0). Sin pesos
aprendidos: baseline reproducible y honesto.

## 3. Fusión de embeddings (Experimento B, backbones congelados)

Cabeza `Linear(256,32)→ReLU→Dropout(0,40)→Linear(32,1)`, **8.257 parámetros
entrenables**, `BCEWithLogitsLoss`, AdamW(lr 0,001, wd 0,001), batch 32,
máx 100 épocas, early stopping patience 15, ReduceLROnPlateau(0,5/5),
semillas 42 y 1337, checkpoint por menor Val loss. Embeddings
estandarizados con media/std **solo de Train** por experimento. Encoders en
`eval()` sin gradientes; preprocesamientos y checkpoints originales
intactos.

| Cabeza (seleccionada) | Seed | Época | Loss Tr/Val | Acc | F1 | Macro-F1 | ROC-AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| B1 Pose+FULL (s42) | 42 | 2/17 | 0,1706/0,2373 | 0,8788 | 0,9032 | 0,8705 | 0,9556 | 31/8/4/56 |
| B1 (s1337) | 1337 | 3 | –/0,2444 | – | 0,8852 | 0,8505 | 0,9547 | – |
| B2 Pose+SIN (s1337) | 1337 | 3 | 0,1671/0,2602 | 0,8788 | 0,9000 | 0,8731 | 0,9530 | 33/6/6/54 |
| B2 (s42) | 42 | 2 | –/0,2638 | 0,8788 | 0,9000 | 0,8731 | 0,9530 | 33/6/6/54 |

**Sobreajuste inmediato y esperado:** mejor época 2–3 en las 4 ejecuciones;
Train F1 ≈ 0,96 frente a Val F1 ≈ 0,90. Los encoders ya vieron los 462
Train, así que la cabeza ajusta representaciones in-sample: experimento
válido de transferencia, **no stacking out-of-fold** (documentado en
`fusion_config.json` antes de entrenar). Variabilidad entre semillas
mínima en B2 (idénticas métricas) y moderada en B1.

## 4. Tabla comparativa (mismos 99, umbral 0,5)

| Modelo | Acc | F1 | Macro-F1 | BalAcc | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| PoseCNN | 0,8990 | 0,9219 | 0,8895 | 0,8763 | 0,9641 | 9 | 1 |
| SegCNN FULL | 0,8687 | 0,8943 | 0,8605 | 0,8558 | 0,9500 | 8 | 5 |
| SegCNN SIN | 0,9192 | 0,9355 | 0,9137 | 0,9064 | 0,9462 | 6 | 2 |
| **A1 tardía Pose+FULL** | 0,9192 | 0,9375 | 0,9116 | 0,8974 | 0,9679 | 8 | 0 |
| **A2 tardía Pose+SIN** | 0,9293 | 0,9449 | 0,9231 | 0,9103 | 0,9594 | 7 | 0 |
| B1 emb Pose+FULL | 0,8788 | 0,9032 | 0,8705 | 0,8641 | 0,9556 | 8 | 4 |
| B2 emb Pose+SIN | 0,8788 | 0,9000 | 0,8731 | 0,8731 | 0,9530 | 6 | 6 |

Menor FP: **B2 y SegCNN SIN (6)**, seguidos de A2 (7). Mayor ROC-AUC:
A1 (0,9679), luego Pose (0,9641). Mayor Accuracy/F1/Macro-F1: **A2**.
Bootstrap pareado por video (2000 remuestreos, descriptivo): A2 vs Pose
ΔAcc media +0,031 IC95 [0,000; 0,071]; A2 vs SegSIN +0,010 [−0,020; 0,040];
A1 vs SegFULL +0,050 [−0,010; 0,111]. Intervalos anchos: n=99, sin
conclusión estadística.

## 5. Errores pareados y complementariedad

- A1: 8 errores, todos compartidos con Pose; corrige bp_321 y bp_507 de
  Pose; introduce bp_353 y bp_420 (fallos solo-A1 vs SegFULL).
- A2: 7 errores; corrige bp_321, bp_507 y bp_556 de Pose; introduce solo
  bp_353 (vs SegSIN). FN=0 en ambas: corrigen los 5 FN de SegFULL y los 2
  de SegSIN, incluido bp_288.
- B1: 12 errores (5 nuevos vs Pose: bp_288, bp_367, bp_432, bp_514, bp_589).
- B2: 12 errores (6 FN: no corrige ningún FN individual y añade nuevos).
- Núcleo duro: bp_386, bp_387, bp_475, bp_564, bp_570 fallan en los tres
  modelos base; **ninguna fusión los corrige** (5/5 persisten en A1, A2, B1,
  B2). No se afirma que sean imposibles: solo que esta fusión no los
  resuelve. Tablas por video en `fusion_paired_error_analysis.csv` y
  `fusion_pairwise_analysis.json`.

## 6. Selección y recomendación

Criterio prefijado en `fusion_config.json` (Macro-F1 → especificidad
inválidos → BalAcc): ranking `late_sin (0,9231/0,8205)` > seg_sin >
late_full > pose > emb_sin > emb_full > seg_full. **Candidato provisional:
A2 tardía Pose+SIN** (`late_fusion_sin_relacion/`, sin checkpoint que
entrenar). Las cabezas de embeddings **no superan** a los individuales y
muestran sobreajuste in-sample: se reportan como resultado negativo
transparente, no como modelo a congelar. Si se exige fusión paramétrica,
B2-s1337 es la menos mala, pero no es competitiva.

**Congelar antes de Test:** A2 (regla 50/50 + umbral 0,5 + checkpoints
Pose-s42 y SegSIN-s42), el protocolo de evaluación (mismas 7 métricas por
video) y la prohibición de reajustar umbrales con Test. Limitaciones que se
mantienen: SegCNN asistida por Pose (no modalidades independientes),
perfil ≠ pecho, Validation reutilizado en 3 selecciones previas + esta
(no es evaluación externa), e independencia por atleta/sesión no acreditada.
